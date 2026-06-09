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
