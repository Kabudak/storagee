# June9 改进记录

本文档记录围绕 HyFormer 论文原文对当前 Taobao 复现项目做过的结构审查、代码改进、数据适配和测试结果。文件名按用户要求保留为 `June9.md`，内容更新到 2026-06-10。

## 1. 改进目标

本轮改造的目标不是追求公开 Taobao 三表数据上的最高指标，而是让项目更像 HyFormer：

- 输入组织要符合 CTR/推荐的 sparse/dense field 结构。
- non-seq features 要形成 feature tokens。
- sequence features 要形成多条序列分支。
- Query Generation 要使用 non-seq context 和 sequence summaries。
- Query Decoding 要逐序列独立完成。
- Query Boosting 要在 decoded queries 和 non-seq tokens 之间做交互。
- 小数据条件下允许参数简化，但要明确记录取舍。

## 2. 论文对齐审查结论

原始项目已经具备 HyFormer 的大致轮廓，但存在几个影响复现质量的问题：

1. Tokenization 过粗。
   - 早期实现把多个 sparse field embedding 直接 mean pooling。
   - 这会抹掉 field identity，不利于 HyFormer 的 feature interaction。

2. Query Generation 上下文不足。
   - 只看当前分支 pooled summary 时，多序列 query 没有完整 global context。
   - HyFormer multi-sequence 思想要求不同序列 query 在生成阶段就能感知全局 non-seq 信息和序列摘要。

3. Query Boosting 残差不够稳。
   - Mixer 内部需要 token-mixing residual 和 channel FFN residual。
   - 当 `d_model` 需要投影到 `mixer_dim` 时，还应该保留模块外部 residual。

4. Prediction head 过度 mean pooling。
   - 直接平均所有 tokens 会混掉 query tokens 与 non-seq tokens 的角色。

5. 数据侧必须承认三表限制。
   - 公开 Taobao 数据通常没有 `raw_behavior_log`。
   - 只能从 `raw_sample` 反推 click/exposure 历史，不应在文档中称为与论文完全同构。

## 3. 已完成代码改进

### 3.1 Field-aware tokenization

文件：

- `models/taac_hyformer.py`

改动：

- `SemanticTokenBuilder` 不再 mean pooling 多个 sparse field embedding。
- 每个 sparse field embedding 加 field offset。
- 多个 field embedding flatten 后进入 MLP。
- dense fields 单独 MLP。
- sparse/dense 表达相加后 LayerNorm。

动机：

- 保留 field identity。
- 更贴近论文中 feature token interaction 的前提。

取舍：

- 没有为每个 field 都单独生成一个 token，因为 Taobao 三表字段较少且样本规模有限。
- 当前按语义组形成 9 个 non-seq tokens，是论文思想和小数据可训练性的折中。

### 3.2 Sequence step field-aware encoder

文件：

- `models/taac_hyformer.py`

改动：

- 历史 step 内的 sparse fields 同样使用 field offset + flatten + MLP。
- dense step features 单独 MLP。
- 加 position embedding 和 sequence type embedding。

动机：

- 避免把 `adgroup_id`、`cate_id`、`campaign_id`、`brand` 等字段平均成一个无字段身份的向量。
- 让 click/exposure 分支即使字段集合相同，也能由 sequence type embedding 区分。

### 3.3 Query Generation 使用全局多序列上下文

文件：

- `models/taac_hyformer.py`

改动：

```text
global_context = flatten(non_seq_tokens) + concat(all_pooled_sequence_summaries)
```

每条序列分支有独立 query generator MLP。

动机：

- HyFormer 的 query/global token 不应只来自单个序列局部 summary。
- multi-sequence 场景下，click query 和 exposure query 都应该看到 non-seq tokens 与所有序列摘要。

取舍：

- 当前使用 branch-specific MLP，而不是完全共享 MLP。
- 这样保留不同序列分支的 query 生成差异。

### 3.4 增强三表下的历史序列特征

文件：

- `scripts/preprocess_taobao.py`

新增 non-seq history summary：

- same ad count
- same cate count
- same brand count
- same campaign count
- same customer count
- click/exposure 分支各自统计

新增 sequence step dense features：

- `price_delta_log`
- `price_ratio_log`
- `recency_rank_log`
- `relative_position`

动机：

- 三表数据没有真实行为日志，只能尽量让反推历史承载更多 target-aware 信息。
- 这些字段能帮助 query decoding 关注与当前广告相关的历史 step。

取舍：

- 这些 target-aware matching features 是推荐系统工程上的适配，不是论文逐字规定。
- 但它们服务于 HyFormer 的核心思想：让 query 从行为序列中解码与当前预测相关的信息。

### 3.5 QueryBoostMixer 残差结构修正

文件：

- `main_pytorch.py`

改动：

- 拆成 `token_norm` 和 `channel_norm`。
- token mixing 使用 residual。
- per-token FFN 使用 residual。
- 模块外部保留输入 residual。

当前形式：

```text
z = in_proj(x)
z = z + token_mix(token_norm(z))
z = z + ffn(channel_norm(z))
out = x + out_proj(z)
```

动机：

- 更接近 Transformer/Mixer 类模块稳定训练的常见结构。
- 当 `d_model=128,total_tokens=25` 时，内部会投影到 `mixer_dim=150`，外部 residual 可以避免投影路径破坏原始 token 表达。

### 3.6 Prediction head 分离 query 与 non-seq 表达

文件：

- `models/taac_hyformer.py`

改动：

```text
query_repr = mean(all_query_tokens)
non_seq_repr = mean(all_non_seq_tokens)
head_input = concat(query_repr, non_seq_repr)
```

动机：

- HyFormer 中 query tokens 承载 sequence-aware interest。
- non-seq tokens 承载静态和上下文特征。
- 分开 readout 比直接 mean 所有 tokens 更能保持角色分工。

### 3.7 训练 loss 默认回到 unweighted BCE

文件：

- `scripts/run_taobao.py`

改动：

- 新增 `--pos-weight-mode none|auto`。
- 默认 `none`。
- `auto` 仅用于小样本不平衡诊断。

动机：

- CTR 复现更应关注原始 BCE/LogLoss 和 AUC。
- 自动 pos_weight 会改变优化目标，可能在小样本上造成校准不稳。

### 3.8 降低默认 field embedding 维度

文件：

- `scripts/run_taobao.py`

改动：

- 新增 `--field-embed-dim`。
- 训练脚本默认 `16`。

参数量对比：

```text
field_embed_dim=16 -> 45,484,617 params
field_embed_dim=24 -> 61,795,169 params
```

动机：

- 高基数 ID embedding 是主要参数来源。
- 公开小数据不适合过大的 embedding 参数。
- 降低 field embedding 维度不改变 HyFormer 主体信息流。

### 3.9 `--max-raw-rows` 快速读取

文件：

- `scripts/preprocess_taobao.py`

改动：

- `raw_sample` 读取改为 `pd.read_csv(..., nrows=max_raw_rows)`。

动机：

- 原始 `raw_sample.csv` 约 1.1GB。
- 之前即使只测 2000 行，也会先全量读取，smoke test 太慢。

### 3.10 移除不必要的 HuggingFace datasets 顶层依赖

文件：

- `utils/common.py`

改动：

- 去掉 `from datasets import Dataset`。
- 使用 `Protocol` 表示可索引 dataset。

动机：

- 当前 Taobao 三表训练不需要 HuggingFace `datasets`。
- 避免环境缺少该包时训练脚本无法启动。

## 4. 数据适配策略

当前公开数据只有：

- `raw_sample.csv`
- `ad_feature.csv`
- `user_profile.csv`

因此采用如下策略：

- 从 `raw_sample` 的历史曝光样本中反推用户历史。
- `label == 1` 的历史作为 `click_seq`。
- `label == 0` 的历史作为 `exposure_seq`。
- 当前样本只使用当前时间之前的历史。

这不是论文中工业行为日志的完整复刻，但它满足 HyFormer 调试需要：

- 有 explicit multi-sequence branch。
- 有每条序列独立 token stream。
- 有 per-sequence query。
- 有跨分支 query boosting。

## 5. 已跑测试

### 5.1 环境

Python:

```text
D:\torch\.venv\Scripts\python.exe
```

关键包：

```text
torch 2.5.1+cu121
pandas 2.3.3
cuda available: True
```

### 5.2 数据探针

原始文件：

```text
raw_sample.csv: 1,114,618,926 bytes
ad_feature.csv: 32,133,243 bytes
user_profile.csv: 25,118,357 bytes
```

smoke 预处理命令：

```bash
python scripts/preprocess_taobao.py --max-raw-rows 100000 --max-rows 2000 --seq-len 50 --output-dir data/taobao_processed_2000_hist100k
```

输出：

```text
non_seq_sparse: (2000, 14)
non_seq_dense:  (2000, 19)
seq_sparse:     (2000, 2, 50, 7)
seq_dense:      (2000, 2, 50, 11)
seq_mask:       (2000, 2, 50)
labels:         (2000,)
```

覆盖率：

```text
positive samples: 94
pos_rate: 4.70%
samples with click_seq: 75
samples with exposure_seq: 576
```

### 5.3 训练结果估计

命令：

```bash
python scripts/run_taobao.py --data-dir data/taobao_processed_2000_hist100k --max-rows 2000 --epochs 3 --batch-size 128 --split-mode random --val-ratio 0.2 --device cuda
```

结果：

```text
field_embed_dim=16
pos_weight_mode=none
best_val_auc=0.5664 @ epoch 3
best epoch val_logloss ~= 0.1605
```

自动正例权重：

```bash
python scripts/run_taobao.py --data-dir data/taobao_processed_2000_hist100k --max-rows 2000 --epochs 3 --batch-size 128 --split-mode random --val-ratio 0.2 --device cuda --pos-weight-mode auto
```

结果：

```text
best_val_auc=0.5119
calibration/accuracy unstable
```

时间切分：

```bash
python scripts/run_taobao.py --data-dir data/taobao_processed_2000_hist100k --max-rows 2000 --epochs 3 --batch-size 128 --split-mode time --val-days 1 --device cuda
```

结果：

```text
train=1758, val=242
val positives=14
best_val_auc=0.4818
```

解释：

- random split 更适合 smoke test。
- time split 更真实，但 2000 行下验证集太小，AUC 噪声大。
- 当前效果说明结构跑通，不说明论文指标已经复现。

## 6. 当前与论文仍有差距的地方

1. 缺少真实行为日志。
   - 这是最大差距。
   - 如果没有 `raw_behavior_log`，Multi-Sequence 的语义只能近似。

2. QueryBoostMixer 是近似实现。
   - 当前保留 token/subspace mixing、独立 subspace MLP、per-token FFN。
   - 但不保证和论文内部实现完全一致。

3. Sequence encoder 是工程简化。
   - 默认 `longer` 用 chunk pooling + cross-attention。
   - 适合小数据调试，但不是完整生产级序列建模。

4. Prediction head 简化。
   - 当前用 query/non-seq pooled concat。
   - 以后可尝试 target-aware attention pooling 或只读 query tokens 的 ablation。

5. ID 处理仍使用 hash bucket。
   - 简单可靠，但存在碰撞。
   - 全量实验建议改为 frequency vocabulary + OOV bucket。

## 7. 下一步建议

P0：

- 用更大 raw history window 预处理，比如 `--max-raw-rows 500000 --max-rows 50000`。
- 观察 click/exposure sequence 覆盖率是否明显提升。

P1：

- 做 encoder ablation：
  - `longer`
  - `full_transformer`
  - `swiglu`

P1：

- 做 readout ablation：
  - query/non-seq separated head
  - query-only head
  - all-token mean head

P2：

- 构建频次词表，减少 hash bucket 碰撞。

P2：

- 清理旧中文注释编码，避免 Windows 控制台显示乱码。

P3：

- 如果能拿到 `raw_behavior_log`，重做数据管线，把 click/exposure 扩展为更真实的多行为序列。

