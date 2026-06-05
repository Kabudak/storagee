# storage 文件夹改动总结

日期：2026-06-05  
目录：`storage/`  
目标：把 `storage` 里的 HyFormer 复现代码从“合并序列 / 简化 global token”版本，调整为更贴近论文原文的 multi-sequence HyFormer prototype。

## 1. 核心方向

这次改动的核心不是追求在 1000 条 demo 数据上刷 AUC，而是让代码结构更接近 HyFormer 论文：

- 多条行为序列不再 merge 成一个 `combined_kv`。
- 每条序列拥有自己的 query/global token。
- 每条序列独立做 Query Decoding。
- Query Boosting 阶段再把所有 decoded query tokens 和 non-seq tokens 放在一起做 token mixing。
- non-seq tokens 在每一层参与 Query Boosting，而不是只在 Query Generation 阶段出现一次。

当前更准确的模型链路是：

```text
seq_i -> SequenceEncoder_i -> K/V_i
query_i -> CrossAttention(query_i, K/V_i) -> decoded_query_i

concat(decoded_query_1, decoded_query_2, ..., non_seq_tokens)
  -> QueryBoostMixer
  -> updated query_i + updated non_seq_tokens
  -> next HyFormer layer
```

## 2. `main_pytorch.py` 改动

### 2.1 恢复 per-sequence Query Decoding

`HyFormerLayer` 中把单个 shared decoder 改成了每条序列一个 decoder：

```python
self.query_decoders = nn.ModuleList(
    [CrossAttentionBlock(d_model=d_model, num_heads=num_heads) for _ in range(num_sequences)]
)
```

每条序列单独执行：

```python
decoded_query = self.query_decoders[seq_idx](
    query_tokens[seq_idx],
    encoded_seq,
    key_padding_mask=~ensure_non_empty_mask(encoded_mask),
)
```

这对应论文里多序列场景下的设计：不同序列先保留各自语义，不在 Query Decoding 之前直接 merge。

### 2.2 Query Boosting 混合 query 和 non-seq tokens

Query Boosting 的输入改为：

```python
mixed_tokens = torch.cat(decoded_queries + [non_seq_tokens], dim=1)
boosted_tokens = self.query_boost(mixed_tokens)
```

然后再把 boosted tokens 切回：

```text
updated query tokens per sequence
updated non-seq tokens
```

这样更贴近论文中 Query Boosting 的职责：做 query-to-query、cross-sequence、query-to-feature interaction。

### 2.3 token 数设置更贴近论文

`HyFormerLayer` 现在使用：

```text
total_tokens = num_sequences * num_queries_per_sequence + num_non_seq_tokens
```

默认配置建议：

```text
num_sequences = 3
num_queries_per_sequence = 1
num_non_seq_tokens = 13
total_tokens = 16
```

这与论文实验里“13 个 non-seq tokens + 3 个 global/query tokens = 16 个 mixer tokens”的口径更一致。

### 2.4 sequence state 跨层传递

保留了你新版里更合理的改法：

```python
current_seq_tokens = encoded_sequences
current_seq_masks = encoded_masks
```

也就是下一层接收上一层 sequence encoder 的输出，而不是每层都重新从初始 sequence token 编码。

## 3. `models/taac_hyformer.py` 改动

### 3.1 Query Generator 改为每条序列一个

之前是生成一组 shared global tokens。现在改为：

```python
self.query_generators = nn.ModuleList([... for _ in range(num_sequences)])
```

每个 generator 输出：

```text
[batch_size, global_tokens_per_seq, d_model]
```

返回值是一个 list：

```python
query_tokens = [
    generator(global_info).view(batch, global_tokens_per_seq, d_model)
    for generator in self.query_generators
]
```

### 3.2 加回 non-seq tokenizer

新增/保留：

```python
self.non_seq_tokenizer = nn.Linear(non_seq_dim, num_non_seq_tokens * d_model)
```

forward 中构造：

```python
non_seq_tokens = self.non_seq_tokenizer(non_seq_x).view(
    batch_size,
    self.num_non_seq_tokens,
    self.d_model,
)
```

这样 non-seq features 不只是用于 query generation，还会进入每层 Query Boosting。

## 4. `scripts/run_taac2026_sample.py` 改动

### 4.1 CLI 参数调整

删除容易误解的：

```bash
--num-global-tokens
```

改为论文语义更清晰的：

```bash
--global-tokens-per-seq 1
--num-non-seq-tokens 13
```

推荐默认：

```bash
python scripts/run_taac2026_sample.py ^
  --num-sequences 3 ^
  --global-tokens-per-seq 1 ^
  --num-non-seq-tokens 13
```

含义是：3 条序列，每条序列 1 个 query/global token，再加 13 个 non-seq tokens。

### 4.2 best checkpoint 显存问题修复

原来 best model state 是直接 `clone()`，CUDA 训练时会在 GPU 上额外保留一份模型权重。

现在改成：

```python
best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
```

这样 best checkpoint 权重拷贝到 CPU，不占用额外 GPU 显存。

best checkpoint 中也不再保存 final epoch 的 optimizer/scaler：

```python
"optimizer": None,
"scaler": None,
```

避免出现“best model 权重 + final optimizer state”不匹配的问题。

## 5. 文档改动

同步更新了：

- `storage/README.md`
- `storage/AGENTS.md`
- `storage/Test.md`

主要对齐：

- 不再写 shared global tokens。
- 改成 per-sequence query/global tokens。
- 命令示例改为 `--global-tokens-per-seq`。
- `Test.md` 原来有乱码，已改成可读中文说明。

## 6. 关于 non-seq tokens 的论文理解

论文里的 `13 non-sequential tokens` 不是说原始非序列特征只有 13 条。

更准确是：

- 原始非序列特征可以很多，包括 user/query/document/cross/context 等。
- 这些特征经过 tokenization / semantic grouping 后，被组织成固定数量的 non-seq tokens。
- 论文实验里使用的是 13 个 non-seq tokens。
- 这 13 个 token 不是逐层减少的，而是在每一层 Query Boosting 中和 decoded query tokens 一起更新。
- 论文里提到的 feature selection / query compression 主要是为了控制 query generation 和 serving 成本，不等同于 non-seq token 数逐层递减。

因此当前默认：

```text
num_non_seq_tokens = 13
global_tokens_per_seq = 1
num_sequences = 3
```

是更贴近论文表述的实验口径。

## 7. 当前仍未解决的问题

这版更接近论文结构，但仍不是完整论文复现：

- 当前 TAAC demo 数据只有 1000 条，AUC/ACC 基本没有参考性。 T0  初步解决 找了kuairand和淘宝广告 先进行特征处理
- 当前多序列来自公开样例的自动分组，不是真实 long-term/search/feed 语义序列。T0
- 数据侧还没有真实长行为序列，例如 1k/3k 用户历史。 T0
- 特征侧仍是连续特征 tensorization，没有实现 CTR 工业常见 sparse embedding tokenizer。T1
- metric 仍是普通 AUC / multiclass AUC，不是论文里的 Query-level AUC。T2
- 训练目标仍是样例分类任务的 CrossEntropyLoss，不是 CTR 二分类 BCE。T2
- `longer` 仍是 LONGER-style 近似，不是完整工业 LONGER。T2
- GPU Pooling、Async AllReduce、FLOPs/latency 统计还没有实现。T3

## 8. 验证情况

已做不写 `.pyc` 的 AST 语法检查，核心 Python 文件语法通过。

由于当前 Codex 环境里的 PyTorch DLL 加载失败，未在该环境下完成 forward/runtime 验证。你本机环境已经能跑，因此建议用下面命令做最小回归：

```bash
python scripts/run_taac2026_sample.py --max-rows 1000 --epochs 1 --batch-size 32 --device cpu --no-amp
```

重点检查：

- shape 是否正常。
- loss 是否能下降/反传。
- `--global-tokens-per-seq 1 --num-non-seq-tokens 13` 是否能正常跑。
- `--save-checkpoint` 是否能保存 best checkpoint 且显存不再额外暴涨。
