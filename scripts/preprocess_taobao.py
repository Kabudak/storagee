"""
淘宝广告点击率(CTR)数据集预处理脚本
=====================================

数据来源: Alimama (Taobao Display Ad Click)
文件:
  - raw_sample.csv: 展示/点击日志  (user, time_stamp, adgroup_id, pid, nonclk, clk)
  - ad_feature.csv: 广告特征       (adgroup_id, cate_id, campaign_id, customer, brand, price)
  - user_profile.csv: 用户画像     (userid, cms_segid, cms_group_id, final_gender_code,
                                     age_level, pvalue_level, shopping_level, occupation,
                                     new_user_class_level)

预处理流程:
  1. 读取三张 CSV 并做基础清洗（缺失值填充、类型转换）
  2. 拼接表: raw_sample LEFT JOIN ad_feature ON adgroup_id
             raw_sample LEFT JOIN user_profile ON user=userid
  3. 按 user 分组，按时间排序，构建两条异构行为序列:
     - 序列0 (click_seq):    用户历史**点击**的 adgroup_id 序列 (clk=1)
     - 序列1 (exposure_seq): 用户历史**曝光未点击**的 adgroup_id 序列 (nonclk=1)
  4. 对当前样本提取非序列特征 (non_seq_features, 17维):
     - 用户ID (1维): user_id
     - 用户画像 (6维): gender, age_level, pvalue_level, shopping_level, occupation, city_level
     - 目标广告ID (1维): adgroup_id
     - 目标广告属性 (4维): campaign_id, customer, brand, price
     - 广告类目 (1维): cate_id
     - 上下文特征 (4维): pid_type, pid_id, hour_of_day, day_of_week
  5. 输出 PyTorch tensor 并保存到 data/taobao_processed/
     - non_seq_x: [N, 17]   (原始非序列特征，维度不限)
     - seq_x:     [N, 2, seq_len, 1]   (2条序列，每步1个adgroup_id特征)
     - labels:    [N]  (0=未点击, 1=点击)

Token 配置 (对齐 HyFormer 论文 16-token 设计):
  - 原始 non_seq_dim=17 → non_seq_tokenizer (Linear) 映射到 14 个 non-seq tokens
  - 每条序列 1 个 query token → 2 个 query tokens
  - 14 non-seq tokens + 2 query tokens = 16 总 token
  - 训练时: --num-sequences 2 --num-non-seq-tokens 14 --global-tokens-per-seq 1
  - 原始特征维度与 token 数量解耦: non_seq_tokenizer 自动完成投影

关键设计:
  - 非序列特征维度不限: 所有有价值的特征都放入 non_seq_x，模型内部做投影
  - 两条异构序列: 点击 vs 曝光未点击，捕获不同行为信号
  - 曝光未点击序列保留稀疏点击场景下的大量负反馈信息
  - 行为序列只取当前样本时间戳之前的记录，防止数据泄漏
  - 广告特征放 non_seq: CTR 预测 P(user 点击 ad)，广告是 query side，
    HyFormer 的 Query Generation 用 non_seq（含广告特征）与历史行为交互
  - 与 taac_data.py 使用相同的 squash/safe_float 函数确保特征处理一致性

用法:
  python scripts/preprocess_taobao.py
  python scripts/preprocess_taobao.py --max-rows 100000 --seq-len 100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from bisect import bisect_left
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 工具函数 (与 taac_data.py 保持一致)
# ---------------------------------------------------------------------------

def safe_float(value) -> float:
    """将任意值转为 float，字符串做 hash 映射。"""
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
            return int(digest[:12], 16) / float(16**12)
    return 0.0


def squash(value: float) -> float:
    """copysign(log1p(abs(x)), x) — 与 taac_data.py 中 squash_numeric 一致。"""
    if value == 0.0:
        return 0.0
    return math.copysign(math.log1p(abs(value)), value)


def squash_array(arr: np.ndarray) -> np.ndarray:
    """向量化版本的 squash。"""
    vals = arr.astype(np.float64)
    mask_invalid = ~np.isfinite(vals)
    vals[mask_invalid] = 0.0
    result = np.where(vals == 0.0, 0.0, np.copysign(np.log1p(np.abs(vals)), vals))
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# 1. 读取 & 基础清洗
# ---------------------------------------------------------------------------

def load_raw_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读取三张 CSV 并做基础清洗。"""
    print("[1/5] 读取原始 CSV ...")

    raw_sample = pd.read_csv(data_dir / "raw_sample.csv")
    ad_feature = pd.read_csv(data_dir / "ad_feature.csv")
    user_profile = pd.read_csv(data_dir / "user_profile.csv")

    # --- raw_sample 清洗 ---
    raw_sample["time_stamp"] = raw_sample["time_stamp"].astype(np.int64)
    raw_sample["user"] = raw_sample["user"].astype(np.int64)
    raw_sample["adgroup_id"] = raw_sample["adgroup_id"].astype(np.int64)
    # 解析 pid 为两个数值: scenario_type, scenario_id
    pid_split = raw_sample["pid"].str.split("_", expand=True)
    raw_sample["pid_type"] = pid_split[0].astype(np.int64)
    raw_sample["pid_id"] = pid_split[1].astype(np.int64)
    raw_sample.drop(columns=["pid"], inplace=True)

    # 标签: clk=1 为点击, clk=0 为未点击; nonclk 与 clk 互补，只保留 clk
    raw_sample.drop(columns=["nonclk"], inplace=True)
    raw_sample.rename(columns={"clk": "label"}, inplace=True)

    # --- ad_feature 清洗 ---
    ad_feature["adgroup_id"] = ad_feature["adgroup_id"].astype(np.int64)
    ad_feature["cate_id"] = ad_feature["cate_id"].astype(np.int64)
    ad_feature["campaign_id"] = ad_feature["campaign_id"].astype(np.int64)
    ad_feature["customer"] = ad_feature["customer"].astype(np.int64)
    # brand 含 NULL 字符串，填充为 0
    ad_feature["brand"] = ad_feature["brand"].replace("NULL", np.nan)
    ad_feature["brand"] = ad_feature["brand"].fillna(0).astype(np.int64)
    ad_feature["price"] = pd.to_numeric(ad_feature["price"], errors="coerce").fillna(0.0)

    # --- user_profile 清洗 ---
    # 列名可能有尾部空格
    user_profile.columns = user_profile.columns.str.strip()
    user_profile["userid"] = user_profile["userid"].astype(np.int64)
    # 缺失值填充为 0
    for col in ["cms_segid", "cms_group_id", "final_gender_code",
                "age_level", "pvalue_level", "shopping_level",
                "occupation", "new_user_class_level"]:
        user_profile[col] = pd.to_numeric(user_profile[col], errors="coerce").fillna(0).astype(np.int64)

    print(f"  raw_sample:  {len(raw_sample):,} 行")
    print(f"  ad_feature:  {len(ad_feature):,} 行")
    print(f"  user_profile: {len(user_profile):,} 行")
    return raw_sample, ad_feature, user_profile


# ---------------------------------------------------------------------------
# 2. 拼接表
# ---------------------------------------------------------------------------

def join_tables(
    raw_sample: pd.DataFrame,
    ad_feature: pd.DataFrame,
    user_profile: pd.DataFrame,
) -> pd.DataFrame:
    """拼接三张表。"""
    print("[2/5] 拼接表 ...")
    df = raw_sample.merge(ad_feature, on="adgroup_id", how="left")
    df = df.merge(user_profile, left_on="user", right_on="userid", how="left")
    # 去掉重复的 userid 列
    if "userid" in df.columns:
        df.drop(columns=["userid"], inplace=True)

    # 填充可能因 join 产生的 NaN
    for col in ["cate_id", "campaign_id", "customer", "brand"]:
        df[col] = df[col].fillna(0).astype(np.int64)
    df["price"] = df["price"].fillna(0.0)
    for col in ["cms_segid", "cms_group_id", "final_gender_code",
                "age_level", "pvalue_level", "shopping_level",
                "occupation", "new_user_class_level"]:
        df[col] = df[col].fillna(0).astype(np.int64)

    # ------------------------------------------------------------------
    # 去重: 数据集说明指出，以 userID+timestamp 为主键会有大量重复记录
    # 原因: 不同类型行为数据来自不同部门，打包时存在微小时间偏差
    # 处理策略: 按 (user, adgroup_id) 去重
    #   - 如果同一用户对同一广告有点击记录，优先保留点击 (label=1)
    #   - 否则保留最新的一条曝光未点击记录 (label=0)
    # ------------------------------------------------------------------
    before = len(df)
    df.sort_values(["user", "adgroup_id", "label", "time_stamp"],
                   ascending=[True, True, False, False], inplace=True)
    df.drop_duplicates(subset=["user", "adgroup_id"], keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)
    after = len(df)
    if before != after:
        print(f"  去重: {before:,} → {after:,} (去掉 {before - after:,} 条重复记录)")

    print(f"  拼接后: {len(df):,} 行, {df.shape[1]} 列")
    return df


# ---------------------------------------------------------------------------
# 3. 构建用户历史行为序列 (防止数据泄漏)
# ---------------------------------------------------------------------------

def build_user_behavior_sequences(
    df: pd.DataFrame,
) -> dict[int, dict[str, list]]:
    """
    为每个用户构建两条异构行为序列:
      - click_seq:    用户历史点击的 adgroup_id (label=1)
      - exposure_seq: 用户历史曝光未点击的 adgroup_id (label=0)

    返回:
      {
        user_id: {
          "click_ts":     [int, ...],     # 点击时间戳 (已排序)
          "click_adids":  [int, ...],     # 点击的 adgroup_id
          "expose_ts":    [int, ...],     # 曝光未点击时间戳 (已排序)
          "expose_adids": [int, ...],     # 曝光未点击的 adgroup_id
        }
      }
    """
    print("[3/5] 构建用户行为序列 (点击 + 曝光未点击) ...")

    user_histories: dict[int, dict[str, list]] = {}

    # --- 点击序列 ---
    click_df = df[df["label"] == 1][["user", "time_stamp", "adgroup_id"]].copy()
    click_df.sort_values(["user", "time_stamp"], inplace=True)
    for uid, group in click_df.groupby("user"):
        uid = int(uid)
        if uid not in user_histories:
            user_histories[uid] = {"click_ts": [], "click_adids": [], "expose_ts": [], "expose_adids": []}
        user_histories[uid]["click_ts"] = group["time_stamp"].values.tolist()
        user_histories[uid]["click_adids"] = group["adgroup_id"].values.tolist()

    # --- 曝光未点击序列 ---
    expose_df = df[df["label"] == 0][["user", "time_stamp", "adgroup_id"]].copy()
    expose_df.sort_values(["user", "time_stamp"], inplace=True)
    for uid, group in expose_df.groupby("user"):
        uid = int(uid)
        if uid not in user_histories:
            user_histories[uid] = {"click_ts": [], "click_adids": [], "expose_ts": [], "expose_adids": []}
        user_histories[uid]["expose_ts"] = group["time_stamp"].values.tolist()
        user_histories[uid]["expose_adids"] = group["adgroup_id"].values.tolist()

    # 统计
    users_with_click = sum(1 for h in user_histories.values() if h["click_adids"])
    users_with_expose = sum(1 for h in user_histories.values() if h["expose_adids"])
    print(f"  总用户数: {len(user_histories):,}")
    print(f"  有点击序列的用户: {users_with_click:,}")
    print(f"  有曝光序列的用户: {users_with_expose:,}")

    return user_histories


# ---------------------------------------------------------------------------
# 4. 特征工程 & 向量化
# ---------------------------------------------------------------------------

def vectorize_dataset(
    df: pd.DataFrame,
    user_histories: dict[int, dict[str, list]],
    seq_len: int,
    num_sequences: int,
    max_rows: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    将 DataFrame 转为模型可用的 tensor。

    非序列特征 (non_seq_features, 17 维):
      - 用户ID (1维): user_id
      - 用户画像 (6维): gender, age_level, pvalue_level, shopping_level, occupation, city_level
      - 目标广告ID (1维): adgroup_id
      - 目标广告属性 (4维): campaign_id, customer, brand, price
      - 广告类目 (1维): cate_id
      - 上下文特征 (4维): pid_type, pid_id, hour_of_day, day_of_week
      原始特征维度与 token 数量解耦: non_seq_tokenizer (Linear) 自动投影
      non_seq_dim=17 → 14 non-seq tokens

    序列特征 (seq_x):
      - 序列0 (click_seq):    用户在当前时间戳之前点击的 adgroup_id 序列
      - 序列1 (exposure_seq): 用户在当前时间戳之前曝光未点击的 adgroup_id 序列
      每个序列 1 维特征，截断/填充到 seq_len
      => seq_x: [N, 2, seq_len, 1]

    Token 配置:
      14 non-seq tokens + 2 query tokens = 16 total (对齐论文)
    """
    print("[4/5] 特征向量化 ...")

    if max_rows is not None:
        df = df.head(max_rows)

    n = len(df)
    print(f"  处理 {n:,} 条样本")

    # ====================================================================
    # A. 非序列特征 — 向量化计算
    # ====================================================================

    # 用户ID (1维)
    id_cols = ["user", "adgroup_id"]
    # 用户画像 (6维)
    user_cols = [
        "final_gender_code", "age_level", "pvalue_level", "shopping_level",
        "occupation", "new_user_class_level",
    ]
    # 目标广告属性 (4维)
    ad_cols = ["campaign_id", "customer", "brand", "price"]
    # 广告类目 (1维)
    cate_cols = ["cate_id"]
    # 上下文特征 (2维: pid)
    ctx_cols = ["pid_type", "pid_id"]

    non_seq_parts = [df[col].values.astype(np.float32) for col in id_cols + user_cols + cate_cols + ad_cols + ctx_cols]

    # 时间特征 — 向量化 (2维)
    timestamps = df["time_stamp"].values.astype(np.int64)
    dt_index = pd.to_datetime(timestamps, unit="s")
    hours = dt_index.hour.values.astype(np.float32) / 23.0
    dows = dt_index.dayofweek.values.astype(np.float32) / 6.0
    non_seq_parts.append(hours)
    non_seq_parts.append(dows)

    non_seq_np = np.column_stack(non_seq_parts)
    non_seq_tensor = torch.from_numpy(non_seq_np)
    non_seq_dim = non_seq_tensor.size(1)

    # ====================================================================
    # B. 序列特征 — 逐样本构建 (需要按时间截断防止数据泄漏)
    # ====================================================================
    seq_feature_dim = 1
    seq_tensor = torch.zeros(n, num_sequences, seq_len, seq_feature_dim, dtype=torch.float32)

    # 预提取 numpy 数组加速
    users_np = df["user"].values
    timestamps_np = df["time_stamp"].values

    # 为每个用户预提取序列值 (分类ID保持原值)
    user_seq_data: dict[int, dict[str, list]] = {}
    for uid, hist in user_histories.items():
        user_seq_data[uid] = {
            "click_adids": [float(v) for v in hist["click_adids"]],
            "click_ts": hist["click_ts"],
            "expose_adids": [float(v) for v in hist["expose_adids"]],
            "expose_ts": hist["expose_ts"],
        }

    for idx in range(n):
        uid = int(users_np[idx])
        ts = int(timestamps_np[idx])

        sq = user_seq_data.get(uid)
        if sq is None:
            continue

        # --- 序列0: 点击序列 (time_stamp < ts) ---
        click_ts = sq["click_ts"]
        if click_ts:
            # bisect_left 找到第一个 >= ts 的位置，即 < ts 的数量
            cut = bisect_left(click_ts, ts)
            if cut > 0:
                start = max(0, cut - seq_len)
                for s, val in enumerate(sq["click_adids"][start:cut]):
                    seq_tensor[idx, 0, s, 0] = val

        # --- 序列1: 曝光未点击序列 (time_stamp < ts) ---
        expose_ts = sq["expose_ts"]
        if expose_ts:
            cut = bisect_left(expose_ts, ts)
            if cut > 0:
                start = max(0, cut - seq_len)
                for s, val in enumerate(sq["expose_adids"][start:cut]):
                    seq_tensor[idx, 1, s, 0] = val

        if idx % 500000 == 0 and idx > 0:
            print(f"    已处理 {idx:,}/{n:,} ...")

    # ====================================================================
    # C. 标签
    # ====================================================================
    labels = torch.tensor(df["label"].values, dtype=torch.long)

    # 统计
    pos = int(labels.sum().item())
    neg = len(labels) - pos
    has_click_seq = (seq_tensor[:, 0].abs().sum(dim=(1, 2)) > 0).sum().item()
    has_expose_seq = (seq_tensor[:, 1].abs().sum(dim=(1, 2)) > 0).sum().item()

    print(f"  正样本: {pos:,}  负样本: {neg:,}  正样本率: {pos/len(labels)*100:.2f}%")
    print(f"  有点击序列的样本: {has_click_seq:,}/{n:,} ({has_click_seq/max(n,1)*100:.1f}%)")
    print(f"  有曝光序列的样本: {has_expose_seq:,}/{n:,} ({has_expose_seq/max(n,1)*100:.1f}%)")

    metadata = {
        "dataset": "taobao_ad_ctr",
        "num_samples": n,
        "non_seq_dim": non_seq_dim,
        "num_sequences": num_sequences,
        "seq_len": seq_len,
        "seq_feature_dim": seq_feature_dim,
        "num_classes": 2,
        "label_mapping": {"0": 0, "1": 1},
        "positive_samples": pos,
        "negative_samples": neg,
        "pos_rate": round(pos / max(n, 1), 6),
        "samples_with_click_seq": int(has_click_seq),
        "samples_with_expose_seq": int(has_expose_seq),
        "non_seq_feature_names": [
            # 用户ID (1维)
            "user_id",
            # 目标广告ID (1维)
            "adgroup_id",
            # 用户画像 (6维)
            "gender", "age_level", "pvalue_level", "shopping_level",
            "occupation", "city_level",
            # 广告类目 (1维)
            "ad_cate_id",
            # 目标广告属性 (4维)
            "ad_campaign_id", "ad_customer", "ad_brand", "ad_price",
            # 上下文特征 (4维)
            "pid_type", "pid_id", "hour_of_day", "day_of_week",
        ],
        "sequence_names": ["click_seq", "exposure_seq"],
        "token_design": {
            "num_non_seq_tokens": 14,
            "global_tokens_per_seq": 1,
            "num_sequences": 2,
            "total_tokens": 16,
        },
    }

    return non_seq_tensor, seq_tensor, labels, metadata


# ---------------------------------------------------------------------------
# 5. 保存
# ---------------------------------------------------------------------------

def save_outputs(
    output_dir: Path,
    non_seq_tensor: torch.Tensor,
    seq_tensor: torch.Tensor,
    labels: torch.Tensor,
    metadata: dict,
) -> None:
    """保存处理后的 tensor 和元信息。"""
    print("[5/5] 保存输出 ...")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(non_seq_tensor, output_dir / "non_seq_x.pt")
    torch.save(seq_tensor, output_dir / "seq_x.pt")
    torch.save(labels, output_dir / "labels.pt")

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"  保存到: {output_dir}")
    print(f"  non_seq_x: {tuple(non_seq_tensor.shape)}")
    print(f"  seq_x:     {tuple(seq_tensor.shape)}")
    print(f"  labels:    {tuple(labels.shape)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="淘宝广告 CTR 数据集预处理")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "archive",
                        help="原始 CSV 数据目录 (默认: data/archive)")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "taobao_processed",
                        help="输出目录 (默认: data/taobao_processed)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="限制处理行数，用于快速调试 (默认: 全部)")
    parser.add_argument("--seq-len", type=int, default=100,
                        help="用户行为序列最大长度 (默认: 100)")
    parser.add_argument("--num-sequences", type=int, default=2,
                        help="行为序列数量: 2条 (点击+曝光未点击) (默认: 2)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = args.output_dir / "_cache"

    # ------------------------------------------------------------------
    # 步骤 1-3 (读取、拼接、构建行为序列) 较慢，结果缓存到磁盘
    # 后续只调 --max-rows 时可跳过，直接加载缓存
    # ------------------------------------------------------------------
    cache_df = cache_dir / "joined_df.parquet"
    cache_hist = cache_dir / "user_histories.json"

    if cache_df.exists() and cache_hist.exists():
        print("[缓存] 发现缓存文件，跳过读取/拼接/构建序列 ...")
        df = pd.read_parquet(cache_df)
        with open(cache_hist, encoding="utf-8") as f:
            user_histories_raw = json.load(f)
        # JSON 的 key 是字符串，转回 int
        user_histories = {int(k): v for k, v in user_histories_raw.items()}
        print(f"  df: {len(df):,} 行")
        print(f"  用户行为序列: {len(user_histories):,} 个用户")
    else:
        # 1. 读取原始数据
        raw_sample, ad_feature, user_profile = load_raw_data(args.data_dir)

        # 2. 拼接
        df = join_tables(raw_sample, ad_feature, user_profile)

        # 3. 构建用户行为序列
        user_histories = build_user_behavior_sequences(df)

        # 保存缓存
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_df, index=False)
        # JSON key 必须是字符串
        user_histories_str = {str(k): v for k, v in user_histories.items()}
        cache_hist.write_text(
            json.dumps(user_histories_str, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[缓存] 已保存到 {cache_dir}")

    # 4. 向量化 (受 --max-rows 影响，每次都重新执行)
    non_seq_tensor, seq_tensor, labels, metadata = vectorize_dataset(
        df, user_histories,
        seq_len=args.seq_len,
        num_sequences=args.num_sequences,
        max_rows=args.max_rows,
    )

    # 5. 保存
    save_outputs(args.output_dir, non_seq_tensor, seq_tensor, labels, metadata)

    print("\n[完成] 预处理结束。可以使用以下命令训练:")
    print(f"  python scripts/run_taobao.py --data-dir {args.output_dir}")
    print(f"\n快速调试:")
    print(f"  python scripts/run_taobao.py --data-dir {args.output_dir} --max-rows 50000 --epochs 3")


if __name__ == "__main__":
    main()
