"""
淘宝广告 CTR 训练入口脚本
==========================

使用 preprocess_taobao.py 预处理后的数据训练 HyFormer CTR 模型。

与 run_taac2026_sample.py 的区别:
  - 直接加载 .pt tensor，不走 HuggingFace 数据加载
  - CTR 二分类任务: 使用 BCEWithLogitsLoss 而非 CrossEntropyLoss
  - 评估指标: AUC + LogLoss + Accuracy

用法:
  python scripts/run_taobao.py --data-dir data/taobao_processed
  python scripts/run_taobao.py --data-dir data/taobao_processed --max-rows 50000 --epochs 3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.taac_hyformer import TAACHyFormerClassifier
from utils.common import set_seed, split_indices, json_ready_args


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------

def binary_auc_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """计算二分类 AUC (从 logits)。"""
    try:
        from sklearn.metrics import roc_auc_score
        probs = torch.sigmoid(logits).numpy()
        y = labels.numpy()
        if len(set(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, probs))
    except ImportError:
        # fallback: 简单排序法
        return _simple_auc(torch.sigmoid(logits), labels)


def _simple_auc(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """无 sklearn 时的简单 AUC 计算。"""
    probs, labels = probs.flatten(), labels.flatten()
    n_pos = int(labels.sum().item())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    sorted_indices = torch.argsort(probs, descending=True)
    sorted_labels = labels[sorted_indices]

    tpr_prev, fpr_prev = 0.0, 0.0
    tp, fp = 0, 0
    auc = 0.0

    for i in range(len(sorted_labels)):
        if sorted_labels[i].item() == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
        tpr_prev, fpr_prev = tpr, fpr

    return auc


def log_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """计算 LogLoss。"""
    probs = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
    loss = -(labels.float() * torch.log(probs) + (1 - labels.float()) * torch.log(1 - probs))
    return float(loss.mean().item())


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """二分类准确率。"""
    preds = (torch.sigmoid(logits) >= 0.5).long()
    return float((preds == labels).float().mean().item())


# ---------------------------------------------------------------------------
# 训练循环
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float, float]:
    """运行一个 epoch，返回 (loss, auc, logloss, accuracy)。"""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_items = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for non_seq_x, seq_x, labels in loader:
        non_seq_x = non_seq_x.to(device)
        seq_x = seq_x.to(device)
        labels = labels.to(device).float()

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(non_seq_x, seq_x).squeeze(-1)  # [batch]
        loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_items += batch_size
        total_loss += loss.item() * batch_size
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu().long())

    if total_items == 0:
        return 0.0, float("nan"), float("nan"), 0.0

    concat_logits = torch.cat(all_logits)
    concat_labels = torch.cat(all_labels)
    auc = binary_auc_from_logits(concat_logits, concat_labels)
    ll = log_loss_from_logits(concat_logits, concat_labels)
    acc = accuracy_from_logits(concat_logits, concat_labels)

    return total_loss / total_items, auc, ll, acc


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="淘宝广告 CTR 训练 (HyFormer)")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "taobao_processed",
                        help="预处理数据目录 (默认: data/taobao_processed)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="限制训练样本数 (默认: 全部)")
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--num-sequences", type=int, default=2)
    parser.add_argument("--global-tokens-per-seq", type=int, default=1)
    parser.add_argument("--num-non-seq-tokens", type=int, default=14)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-hidden", type=int, default=256)
    parser.add_argument("--hyformer-layers", type=int, default=4)
    parser.add_argument("--seq-encoder-type", choices=("longer", "full_transformer", "swiglu"), default="longer")
    parser.add_argument("--short-seq-len", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "taobao_ctr")
    parser.add_argument("--save-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 加载数据 ----
    print("[数据] 加载预处理 tensor ...")
    non_seq_x = torch.load(args.data_dir / "non_seq_x.pt", weights_only=True)
    seq_x = torch.load(args.data_dir / "seq_x.pt", weights_only=True)
    labels = torch.load(args.data_dir / "labels.pt", weights_only=True)

    with open(args.data_dir / "metadata.json", encoding="utf-8") as f:
        data_meta = json.load(f)

    if args.max_rows is not None:
        n = min(args.max_rows, len(labels))
        non_seq_x = non_seq_x[:n]
        seq_x = seq_x[:n]
        labels = labels[:n]

    # ---- 划分 train/val ----
    train_idx, val_idx = split_indices(len(labels), args.val_ratio, args.seed)
    train_dataset = TensorDataset(non_seq_x[train_idx], seq_x[train_idx], labels[train_idx])
    val_dataset = TensorDataset(non_seq_x[val_idx], seq_x[val_idx], labels[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # ---- 构建模型 ----
    num_classes = 1  # CTR 二分类用 1 个 logit + BCEWithLogitsLoss
    model = TAACHyFormerClassifier(
        non_seq_dim=non_seq_x.size(1),
        seq_feature_dim=seq_x.size(3),
        num_classes=num_classes,
        seq_len=args.seq_len,
        num_sequences=seq_x.size(1),
        global_tokens_per_seq=args.global_tokens_per_seq,
        num_non_seq_tokens=args.num_non_seq_tokens,
        d_model=args.d_model,
        num_heads=args.num_heads,
        ffn_hidden=args.ffn_hidden,
        hyformer_layers=args.hyformer_layers,
        seq_encoder_type=args.seq_encoder_type,
        short_seq_len=args.short_seq_len,
    ).to(args.device)

    # BCEWithLogitsLoss 自动处理正负样本不平衡的 pos_weight
    pos_count = int(labels.sum().item())
    neg_count = len(labels) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], dtype=torch.float32).to(args.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"[模型] device={args.device}  samples={len(labels)}")
    print(f"  train={len(train_loader.dataset)}  val={len(val_loader.dataset)}")
    print(f"  non_seq={tuple(non_seq_x.shape)}  seq={tuple(seq_x.shape)}")
    print(f"  pos_rate={pos_count/len(labels)*100:.2f}%  pos_weight={pos_weight.item():.2f}")
    print(f"  d_model={args.d_model}  layers={args.hyformer_layers}  encoder={args.seq_encoder_type}")

    # ---- 训练 ----
    best_val_auc = float("-inf")
    best_epoch = 0
    best_model_state: dict[str, torch.Tensor] | None = None
    args_payload = json_ready_args(args)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_auc, _, train_acc = run_epoch(
            model, train_loader, criterion, args.device, optimizer
        )
        val_loss, val_auc, val_ll, val_acc = run_epoch(
            model, val_loader, criterion, args.device
        )

        print(
            f"[epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} train_auc={train_auc:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_auc={val_auc:.4f} val_logloss={val_ll:.4f} val_acc={val_acc:.4f}"
        )

        if not math.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # ---- 保存 ----
    run_meta = {
        **data_meta,
        "args": args_payload,
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if args.save_checkpoint and best_model_state is not None:
        ckpt_path = args.output_dir / f"best_model_epoch{best_epoch:02d}_auc{best_val_auc:.4f}.pt"
        torch.save({"model": best_model_state, "metadata": run_meta}, ckpt_path)
        print(f"[保存] best checkpoint: {ckpt_path}")

    print(f"\n[完成] best_val_auc={best_val_auc:.4f} @ epoch {best_epoch}")


if __name__ == "__main__":
    main()