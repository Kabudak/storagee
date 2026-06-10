# 结构化输入版本命令示例

## 1. 预处理

默认使用 `5` 秒时间窗做事件级去重，并输出结构化 sparse/dense 张量：

```bash
python scripts/preprocess_taobao.py
```

快速调试：

```bash
python scripts/preprocess_taobao.py --max-rows 50000 --seq-len 50
```

只训练 2000 条目标样本、但先用更大的 raw_sample 窗口构造历史：

```bash
python scripts/preprocess_taobao.py --max-raw-rows 100000 --max-rows 2000 --seq-len 50 --output-dir data/taobao_processed_2000_hist100k
```

说明：`--max-rows` 会在完整时间轴上做等距抽样，而不是只截取最早的一段数据，便于保留时间切分调试语义。

显式指定去重时间窗：

```bash
python scripts/preprocess_taobao.py --dedup-window-sec 5
```

## 2. 训练

默认按时间切分 train / val：

```bash
python scripts/run_taobao.py --epochs 5 --batch-size 256
```

快速 CPU 检查：

```bash
python scripts/run_taobao.py --max-rows 50000 --epochs 1 --batch-size 64 --device cpu
```

小样本正例过少时可临时启用自动正例权重；正式 CTR LogLoss/AUC 复现实验默认使用不加权 BCE：

```bash
python scripts/run_taobao.py --data-dir data/taobao_processed_2000_hist100k --epochs 3 --batch-size 128 --split-mode random --pos-weight-mode auto
```

说明：训练脚本里的 `--max-rows` 同样会沿完整时间轴做等距抽样，避免小样本调试时验证集只落在最早几天。

使用随机切分做对比实验：

```bash
python scripts/run_taobao.py --split-mode random --val-ratio 0.2
```

按最后 1 天做验证：

```bash
python scripts/run_taobao.py --split-mode time --val-days 1
```

## 3. 常用参数

```bash
python scripts/run_taobao.py ^
  --epochs 5 ^
  --batch-size 256 ^
  --num-queries-per-seq 8 ^
  --num-non-seq-tokens 9 ^
  --d-model 128 ^
  --field-embed-dim 16 ^
  --num-heads 4 ^
  --ffn-hidden 256 ^
  --hyformer-layers 4 ^
  --seq-encoder-type longer ^
  --short-seq-len 8 ^
  --lr 1e-3 ^
  --weight-decay 1e-4
```

## 4. checkpoint

```bash
python scripts/run_taobao.py --epochs 5 --save-checkpoint
```

## 5. 当前预处理输出

预处理完成后，`data/taobao_processed/` 下应包含：

- `non_seq_sparse.pt`
- `non_seq_dense.pt`
- `seq_sparse.pt`
- `seq_dense.pt`
- `seq_mask.pt`
- `labels.pt`
- `timestamps.pt`
- `metadata.json`
