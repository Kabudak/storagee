# HyFormer Taobao 复现版本总结

更新时间：2026-06-10  
论文依据：HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction, arXiv:2601.12681  
当前目标：在公开 Taobao Display Ad Click 三表数据上调试一个尽量贴近 HyFormer 原文机制的 CTR/recommendation 复现版本。

## 1. 总体定位

本项目不是官方 HyFormer 代码，也不是严格工业生产环境复刻。当前版本的目标是：

- 保持 HyFormer 的核心信息流：Query Generation -> Query Decoding -> Query Boosting -> 多层堆叠。
- 在公开 Taobao 三表数据只有 `raw_sample.csv`、`ad_feature.csv`、`user_profile.csv` 的条件下，构造可运行的 multi-sequence CTR 输入。
- 尽量避免把高基数 ID 当连续数值，使用 embedding 和 field-aware tokenization。
- 控制公开小数据上的参数量，让模型能用于结构调试，而不是追求论文级线上指标。

关键结论：

- 模型结构已经具备 HyFormer 风格的 query/global token、per-sequence decoding、query boosting、multi-sequence branch。
- 数据侧无法完全复现论文的行为日志设定，因为公开数据通常没有 `raw_behavior_log`。当前 click/exposure 历史是从目标曝光表 `raw_sample` 中按时间反推得到的。
- 当前数据足够用于调试 HyFormer 输入组织和模型流转，但不应期待复现论文工业数据集上的绝对指标。

## 2. 论文机制到当前代码的映射

HyFormer 原文强调两个角色分工：

- Sequence Modeling：每条行为序列单独建模，用 query/global token 从序列中提取兴趣表达。
- Feature Interaction：decoded queries 与 non-sequential feature tokens 通过 Query Boosting 做跨序列、跨特征交互。

当前代码映射如下：

| 论文概念 | 当前实现 | 文件 |
|---|---|---|
| Non-sequential feature tokenization | 将用户、目标广告、上下文、历史统计等字段组织成 9 个语义 token | `models/taac_hyformer.py` |
| Sequence branch | 固定两条分支：`click_seq` 和 `exposure_seq` | `scripts/preprocess_taobao.py` |
| Query Generation | 使用所有 non-seq tokens 与所有序列 pooled summaries 拼接，分支专属 MLP 生成 query tokens | `models/taac_hyformer.py` |
| Query Decoding | 每条序列分支独立 cross-attention：query attends to encoded sequence | `main_pytorch.py` |
| Query Boosting | decoded queries + non-seq tokens 拼接后进入 RankMixer-like token mixer 和 per-token FFN | `main_pytorch.py` |
| Multi-layer HyFormer module | 每层输出 updated query 与 updated non-seq tokens，下一层继续使用 | `main_pytorch.py` |
| Prediction head | 分别池化 query tokens 和 non-seq tokens，拼接后 MLP 输出 CTR logit | `models/taac_hyformer.py` |

## 3. 数据集结构

当前使用的公开 Taobao Display Ad Click 数据通常包含三张表：

### `raw_sample.csv`

字段：

- `user`
- `time_stamp`
- `adgroup_id`
- `pid`
- `nonclk`
- `clk`

角色：

- 每行是一条广告展示样本。
- `clk` 是当前 CTR 二分类标签。
- `pid` 被拆成 `pid_type` 和 `pid_id`。
- 当前实现不使用 `nonclk`，读取后删除。

### `ad_feature.csv`

字段：

- `adgroup_id`
- `cate_id`
- `campaign_id`
- `customer`
- `brand`
- `price`

角色：

- 广告侧静态属性表。
- 与 `raw_sample` 按 `adgroup_id` 关联。

### `user_profile.csv`

字段：

- `userid`
- `cms_segid`
- `cms_group_id`
- `final_gender_code`
- `age_level`
- `pvalue_level`
- `shopping_level`
- `occupation`
- `new_user_class_level`

角色：

- 用户画像表。
- 与 `raw_sample` 按 `user = userid` 关联。

当前实现没有假设存在 `raw_behavior_log`。如果将来能拿到官方行为日志，应优先使用行为日志构造真实用户行为序列。

## 4. 预处理流程

入口：

```bash
python scripts/preprocess_taobao.py
```

推荐 smoke test：

```bash
python scripts/preprocess_taobao.py --max-raw-rows 100000 --max-rows 2000 --seq-len 50 --output-dir data/taobao_processed_2000_hist100k
```

主要步骤：

1. 读取三表。
   - `--max-raw-rows` 使用 `pd.read_csv(..., nrows=N)`，适合大文件快速调试。
   - 正式预处理不传该参数，会读取全量 `raw_sample.csv`。
2. 解析 `raw_sample`。
   - `pid` 拆成 `pid_type` 和 `pid_id`。
   - `clk` 重命名为 `label`。
3. 事件级去重。
   - 按 `user + adgroup_id + pid_type + pid_id + time_stamp` 排序。
   - 默认 5 秒窗口内同 key 事件聚合。
   - label 使用 `max`，并保留 `dup_count`、`cluster_span_sec` 作为特征。
4. 关联广告特征和用户画像。
   - `raw_sample` left join `ad_feature`。
   - 再 left join `user_profile`。
5. 构造用户历史。
   - 从当前已关联样本中按用户和时间构造历史。
   - `label == 1` 的历史进入 `click_seq`。
   - `label == 0` 的历史进入 `exposure_seq`。
   - 对当前样本只取 `time_stamp < current_ts` 的历史，避免明显未来泄漏。
6. 构造结构化张量并保存：
   - `non_seq_sparse.pt`
   - `non_seq_dense.pt`
   - `seq_sparse.pt`
   - `seq_dense.pt`
   - `seq_mask.pt`
   - `labels.pt`
   - `timestamps.pt`
   - `metadata.json`

## 5. 输入特征组织

### Non-seq sparse fields

保存为 `non_seq_sparse`，shape 为：

```text
[batch_size, 14]
```

字段：

- `user`
- `adgroup_id`
- `cate_id`
- `campaign_id`
- `customer`
- `brand`
- `pid_type`
- `pid_id`
- `final_gender_code`
- `age_level`
- `pvalue_level`
- `shopping_level`
- `occupation`
- `new_user_class_level`

这些字段通过稳定 hash bucket 进入 embedding。当前 bucket：

- `user`: 524288
- `adgroup_id`: 524288
- `cate_id`: 131072
- `campaign_id`: 262144
- `customer`: 262144
- `brand`: 262144
- `pid_type`: 4096
- `pid_id`: 65536
- 用户画像小字段：64

### Non-seq dense fields

保存为 `non_seq_dense`，shape 为：

```text
[batch_size, 19]
```

字段：

- `price_log`
- `hour_of_day`
- `day_of_week`
- `click_hist_len_log`
- `exposure_hist_len_log`
- `click_last_gap_log`
- `exposure_last_gap_log`
- `click_same_ad_count_log`
- `exposure_same_ad_count_log`
- `click_same_cate_count_log`
- `exposure_same_cate_count_log`
- `click_same_brand_count_log`
- `exposure_same_brand_count_log`
- `click_same_campaign_count_log`
- `exposure_same_campaign_count_log`
- `click_same_customer_count_log`
- `exposure_same_customer_count_log`
- `event_dup_count_log`
- `event_cluster_span_log`

### Non-seq token groups

当前 non-seq features 被组织成 9 个语义 token：

1. `user_profile_token`
2. `user_identity_token`
3. `target_ad_identity_token`
4. `target_ad_attribute_token`
5. `target_price_token`
6. `context_token`
7. `history_summary_token_click`
8. `history_summary_token_exposure`
9. `current_event_token`

每个 token 内部使用 field-aware tokenization：

- sparse 字段先查 embedding。
- 加 field-specific offset，保留字段身份。
- flatten 后经 MLP 投影到 `d_model`。
- dense 字段经 MLP 投影到 `d_model`。
- sparse/dense 表达相加并 LayerNorm。

这一步替代了早期直接 mean pooling 多个 sparse field embedding 的做法，更贴近 HyFormer 对 feature token interaction 的要求。

## 6. 序列特征组织

当前固定支持两条序列：

```text
num_sequences = 2
sequence_names = ["click_seq", "exposure_seq"]
```

保存为：

```text
seq_sparse: [batch_size, 2, seq_len, 7]
seq_dense:  [batch_size, 2, seq_len, 11]
seq_mask:   [batch_size, 2, seq_len]
```

### Sequence sparse fields

每个历史 step 的 sparse 字段：

- `adgroup_id`
- `cate_id`
- `campaign_id`
- `customer`
- `brand`
- `pid_type`
- `pid_id`

### Sequence dense fields

每个历史 step 的 dense 字段：

- `price_log`
- `price_delta_log`
- `price_ratio_log`
- `time_gap_log`
- `same_ad_as_target`
- `same_cate_as_target`
- `same_brand_as_target`
- `same_campaign_as_target`
- `same_customer_as_target`
- `recency_rank_log`
- `relative_position`

### Sequence tokenization

序列 step 使用 `StructuredSequenceStepEncoder`：

- 每个 sparse field 单独 embedding。
- 加 field offset，flatten 后 MLP 投影到 `d_model`。
- dense step features 经 MLP 投影到 `d_model`。
- sparse/dense 相加并 LayerNorm。
- 加 position embedding。
- 加 sequence type embedding，用于区分 click/exposure branch。

## 7. 模型结构

训练入口模型：

```text
TAACHyFormerClassifier
```

默认训练配置：

```text
num_sequences = 2
num_non_seq_tokens = 9
num_queries_per_seq = 8
d_model = 128
field_embed_dim = 16
num_heads = 4
ffn_hidden = 256
hyformer_layers = 4
seq_encoder_type = longer
short_seq_len = 8
```

默认参数量：

```text
field_embed_dim = 16: 45,484,617 parameters
field_embed_dim = 24: 61,795,169 parameters
```

因为公开数据规模远小于论文中的工业数据，当前训练脚本默认使用 `field_embed_dim=16`，保持结构不变的同时降低 embedding 参数量。

### 7.1 Query Generation

输入：

- non-seq tokens: `[B, 9, d_model]`
- pooled sequence summaries:
  - click pooled summary: `[B, d_model]`
  - exposure pooled summary: `[B, d_model]`

构造：

```text
global_context = flatten(non_seq_tokens) + concat(all_pooled_sequence_summaries)
```

每个序列分支有独立 MLP：

```text
global_context -> [B, num_queries_per_seq, d_model]
```

当前实现让每条序列的初始 query 都看到全部 non-seq tokens 和全部 sequence summaries。这比只看本分支 pooled summary 更贴近 HyFormer multi-sequence 设定。

### 7.2 Sequence Representation Encoder

当前支持三种模式：

- `longer`: 默认。将长序列 chunk-pool 成短序列，再用 cross-attention 压缩。属于对论文高效序列建模思想的工程简化。
- `full_transformer`: 标准 self-attention encoder。
- `swiglu`: attention-free FFN encoder，用于快速 ablation。

默认使用 `longer`，因为公开数据下序列较稀疏且样本较小，全量 Transformer 不一定更稳。

### 7.3 Query Decoding

每层 HyFormer layer 中，每个序列分支独立执行：

```text
decoded_query_i = CrossAttention(query_i, encoded_sequence_i)
```

这对应论文中让 global/query token 从对应行为序列中读取兴趣信息。

### 7.4 Query Boosting

每层中将所有 decoded queries 和 non-seq tokens 拼接：

```text
[decoded_click_queries, decoded_exposure_queries, non_seq_tokens]
```

然后送入 `QueryBoostMixer`：

- 如果 `d_model` 不能被 token 数整除，先投影到可分组的 `mixer_dim`。
- 按 token 数切分 channel subspaces。
- 每个 subspace 内跨 token 拼接并经独立 MLP。
- 执行 token-mixing residual。
- 执行 channel FFN residual。
- 输出回 `d_model`，并保留模块外部 residual。

这是对论文 Query Boosting / RankMixer 思路的可运行近似。当前实现保留独立 subspace MLP，但没有完全复刻论文可能使用的所有工业细节。

### 7.5 多层流转

每层输出：

- updated query tokens
- updated non-seq tokens
- encoded sequence tokens
- encoded sequence masks

下一层继续使用上一层 boosted 后的 query 和 non-seq tokens。这样 query 在多层中不断经历：

```text
sequence-aware decoding -> query/non-seq feature interaction
```

### 7.6 Prediction head

最终输出 tokens 排列：

```text
[all_query_tokens, all_non_seq_tokens]
```

当前 head：

- query tokens mean pool -> `query_repr`
- non-seq tokens mean pool -> `non_seq_repr`
- concat `[query_repr, non_seq_repr]`
- MLP 输出一个 CTR logit

这样比直接 mean pooling 所有 tokens 更能保留 HyFormer 中 query 表达和 feature token 表达的角色分工。

## 8. 训练与评估

训练入口：

```bash
python scripts/run_taobao.py
```

核心设置：

- loss: 默认 unweighted `BCEWithLogitsLoss`
- metric:
  - AUC
  - LogLoss
  - Accuracy
- split:
  - 默认 time split，最后 `val_days` 天作为验证集
  - 可用 random split 做 smoke test
- optimizer: AdamW
- optional:
  - `--pos-weight-mode auto` 可在极小不平衡样本上做诊断，但不作为默认 CTR 复现设置

默认不使用 `pos_weight` 的原因：

- CTR 论文评估通常关注原始概率排序和 LogLoss。
- 自动正例权重会改变 loss 目标，可能让小样本 AUC 偶然升高，但校准和阈值行为明显变差。

## 9. Smoke test 结果

数据：

```bash
python scripts/preprocess_taobao.py --max-raw-rows 100000 --max-rows 2000 --seq-len 50 --output-dir data/taobao_processed_2000_hist100k
```

生成张量：

```text
non_seq_sparse: (2000, 14)
non_seq_dense:  (2000, 19)
seq_sparse:     (2000, 2, 50, 7)
seq_dense:      (2000, 2, 50, 11)
seq_mask:       (2000, 2, 50)
labels:         (2000,)
```

样本统计：

```text
positive samples: 94
pos_rate: 4.70%
samples with click_seq: 75
samples with exposure_seq: 576
mean click steps: 0.041
mean exposure steps: 0.8045
```

训练结果估计，3 epoch：

| split | pos_weight_mode | field_embed_dim | best val AUC | val LogLoss 备注 |
|---|---:|---:|---:|---|
| random 80/20 | none | 16 | 0.5664 | best epoch logloss 约 0.1605 |
| random 80/20 | auto | 16 | 0.5119 | 校准明显不稳 |
| time, last 1 day | none | 16 | 0.4818 | 验证集只有 242 行、14 个正例 |

解释：

- 这个结果不能代表论文指标，只能说明当前结构和输入管线可运行。
- click 序列覆盖很低，说明三表反推历史的表达上限有限。
- random split 更适合 smoke test；time split 更接近真实 CTR，但在 2000 行样本上过于噪声。

## 10. 与 HyFormer 原文的一致和简化

较一致的部分：

- 有明确 non-seq tokens。
- 有多序列分支。
- 每条序列有自己的 query generation 和 query decoding。
- Query Boosting 拼接 decoded queries 与 non-seq tokens。
- 多层中 query/non-seq tokens 持续更新。
- Multi-sequence 不是简单 merge，而是分支独立 decoding、统一 boosting。

工程简化部分：

- 公开数据没有 `raw_behavior_log`，序列只能从 `raw_sample` 中反推。
- sequence encoder 的 `longer` 是短序列 cross-attention 近似，不是论文所有生产细节。
- QueryBoostMixer 是 RankMixer-like 实现，保留核心 token/subspace mixing，但不保证与论文代码逐行一致。
- sequence branch 当前固定 2 条：click/exposure。
- prediction head 是 query/non-seq pooled concat MLP，未实现更复杂的 task-specific head。
- 为小数据默认降低 `field_embed_dim` 到 16。

## 11. 主要限制

1. 历史行为稀疏。
   - 公开三表中没有完整行为日志。
   - `click_seq` 覆盖尤其低。

2. 负曝光历史语义较弱。
   - `exposure_seq` 来自 `label == 0` 的曝光样本。
   - 它不是用户主动行为序列，只是未点击展示序列。

3. 小样本 AUC 不稳定。
   - 2000 行 smoke test 中正例很少。
   - time split 验证集更小，指标波动明显。

4. ID bucket 仍有碰撞。
   - 这是参数量和 ID 表达之间的折中。
   - 如果使用全量数据，可考虑构建频次词表替代 hash bucket。

5. 文档和代码里部分旧中文注释在 Windows 控制台显示为乱码。
   - 不影响编译和运行。
   - 后续可单独做编码清理。

## 12. 后续建议

优先级建议：

1. 用更大的 `--max-raw-rows` 或全量数据预处理，再抽样训练。
   - HyFormer 的序列建模依赖足够历史覆盖。

2. 增加 ablation。
   - `longer` vs `full_transformer` vs `swiglu`
   - query/non-seq separated head vs all-token mean head
   - field_embed_dim 16 vs 24

3. 如果资源允许，构建频次词表。
   - 高频 ID 独立 embedding。
   - 低频 ID 进 OOV/hash bucket。

4. 如果找到 `raw_behavior_log`，重构数据管线。
   - 使用真实点击、加购、收藏、购买等多行为序列。
   - 这会比继续调三表反推历史更接近论文设定。

