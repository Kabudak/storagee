# `run_taac2026_sample.py` 命令行用法

## 基本训练

```bash
python scripts/run_taac2026_sample.py --epochs 5 --batch-size 32
```

## 快速 CPU 检查

```bash
python scripts/run_taac2026_sample.py --max-rows 1000 --epochs 1 --batch-size 32 --device cpu --no-amp
```

## 训练并保存 checkpoint

训练结束时会保存最后一轮 checkpoint；当验证 AUC 出现新高时，还会额外保存一份 best checkpoint。

```bash
python scripts/run_taac2026_sample.py --epochs 5 --batch-size 32 --save-checkpoint
```

## 选择序列编码器

可选值：

- `longer`
- `full_transformer`
- `swiglu`

示例：

```bash
python scripts/run_taac2026_sample.py --epochs 5 --batch-size 32 --seq-encoder-type longer
python scripts/run_taac2026_sample.py --epochs 5 --batch-size 32 --seq-encoder-type full_transformer
python scripts/run_taac2026_sample.py --epochs 5 --batch-size 32 --seq-encoder-type swiglu
```

## 调整 token 配置

```bash
python scripts/run_taac2026_sample.py --epochs 5 --batch-size 32 --num-sequences 3 --global-tokens-per-seq 1 --num-non-seq-tokens 13
```

## 开启 / 关闭混合精度

CUDA 下默认开启 AMP。

```bash
python scripts/run_taac2026_sample.py --epochs 5 --batch-size 32
python scripts/run_taac2026_sample.py --epochs 5 --batch-size 32 --amp-dtype bf16
python scripts/run_taac2026_sample.py --epochs 5 --batch-size 32 --no-amp
```

## 从 checkpoint 继续训练

`--resume` 可以填写：

- `output-dir` 下的 checkpoint 文件名
- checkpoint 的绝对路径

示例：

```bash
python scripts/run_taac2026_sample.py --epochs 10 --batch-size 32 --resume best_model_20260414_164500.pt
python scripts/run_taac2026_sample.py --epochs 10 --batch-size 32 --resume best_model_20260414_164500.pt --save-checkpoint
```

## 读取本地 parquet

```bash
python scripts/run_taac2026_sample.py --local-parquet D:\path\to\demo_1000.parquet --epochs 5 --batch-size 32
```

## 常用调参项

```bash
python scripts/run_taac2026_sample.py ^
  --epochs 5 ^
  --batch-size 32 ^
  --seq-len 16 ^
  --num-sequences 3 ^
  --global-tokens-per-seq 1 ^
  --num-non-seq-tokens 13 ^
  --d-model 128 ^
  --num-heads 4 ^
  --ffn-hidden 256 ^
  --hyformer-layers 4 ^
  --seq-encoder-type longer ^
  --short-seq-len 8 ^
  --lr 1e-3 ^
  --weight-decay 1e-4
```

## 查看完整参数

```bash
python scripts/run_taac2026_sample.py --help
```
