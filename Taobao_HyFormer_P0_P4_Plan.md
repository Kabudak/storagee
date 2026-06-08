# Taobao HyFormer 优化方案（P0-P4）

## 1. 背景与前提

本文档用于规划当前仓库中 HyFormer 复现项目的高优先级优化方向，范围限定为 `P0` 到 `P4`。

本方案建立在以下前提上：

- 只使用当前公开可得的三张表：
  - `raw_sample.csv`
  - `ad_feature.csv`
  - `user_profile.csv`
- 不假设存在额外的官方 `raw_behavior_log` 表。
- 因此，所有历史序列都必须从 `raw_sample.csv` 中回溯构造。
- 优化目标不是单纯把代码外形做得更像论文，而是在当前可得数据条件下尽可能提升 CTR 离线效果。

本文只覆盖下列优先级：

- `P0`：离散特征 embedding 化
- `P1`：序列 step 特征变厚
- `P2`：基于 5 秒时间窗的事件级去重
- `P3`：按时间切分训练/验证集
- `P4`：non-seq token 语义化分组

不在本轮范围内的内容：

- Query Generator 结构再设计
- 输出 head 再设计
- `QueryBoostMixer` 变体对比
- 更激进的 backbone 级 ablation

## 2. 当前问题判断

当前项目已经具备 HyFormer 的核心主干结构：

- per-sequence sequence encoder
- per-sequence query decoding
- query boosting over query tokens + non-seq tokens

但当前效果差，主要不是 backbone 本身的问题，而是输入表达的问题。

当前最主要的瓶颈有 5 个：

1. 高基数 ID 被当作连续数值直接喂入线性层。
2. 每个历史 step 只有一个 `adgroup_id` 标量，信息太薄。
3. 当前预处理里的 `(user, adgroup_id)` 全局去重过于激进。
4. 训练脚本仍然是随机切分，不符合 CTR 的时间前向评估习惯。
5. non-seq token 目前来自一个扁平向量的统一线性投影，token 语义很弱。

一句话概括：

> 当前版本已经是 HyFormer 风格主干，但还不是 CTR 场景下成熟的特征系统。

## 3. 总体设计原则

后续实现统一遵守以下原则：

1. `main_pytorch.py` 继续只放 backbone 级逻辑。
2. 任务相关输入编码放在 `models/taac_hyformer.py`。
3. 数据预处理、去重、张量构造放在 `scripts/` 与 `utils/`。
4. 尽量使用结构化 sparse/dense 输入，不再把所有特征拍平成一个 float 向量。
5. 去重策略不再做宽泛的广告级去重，而改成事件级时间窗聚合。
6. 先建立稳定的数据契约，再改模型输入编码，再改训练流程。

## 4. P0-P4 详细方案

### P0：离散特征 embedding 化

**优先级：最高**

### 目标

把当前直接作为数值使用的高基数离散特征改成结构化 sparse 输入，再由模型内部用 embedding 编码。

### 为什么优先级最高

CTR 建模最依赖的就是离散特征表达。当前版本里：

- `user`
- `adgroup_id`
- `cate_id`
- `campaign_id`
- `customer`
- `brand`

这些字段本质上都是 ID，但目前在预处理后直接进入线性层。这样会丢掉最重要的“身份信息”。

### 目标字段拆分

建议把 non-seq 特征拆成两类：

#### 1. sparse 字段

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

#### 2. dense 字段

- `price`
- `hour_of_day`
- `day_of_week`
- 后续补充的统计特征

### 实现方式

预处理阶段：

- 为高基数离散字段建立稳定哈希桶或词表编号。
- 输出整数 ID 张量，而不是原始 float 值。
- dense 特征继续输出 float 张量。

模型阶段：

- 为每个 sparse field 建立 embedding。
- dense 特征按语义分组后用小 MLP 投影。
- 再由这些 field/group 组合成 non-seq tokens。

### 数据输出格式

P0 完成后，预处理输出不再只有：

- `non_seq_x.pt`
- `seq_x.pt`
- `labels.pt`

而是改成结构化格式：

- `non_seq_sparse.pt`
- `non_seq_dense.pt`
- `seq_sparse.pt`
- `seq_dense.pt`
- `seq_mask.pt`
- `labels.pt`
- `timestamps.pt`
- `metadata.json`

### 验收标准

- 任何高基数 ID 都不再以原始 float 形式输入模型。
- `metadata.json` 明确记录每个 sparse field 的桶大小或词表大小。
- 模型可以根据 metadata 重建一致的输入语义。

---

### P1：序列 step 特征变厚

**优先级：很高**

### 目标

把当前每个序列 step 只有 `adgroup_id` 一维的做法，升级为包含广告属性、上下文和 target-aware 特征的多字段 step 表达。

### 为什么重要

当前 `seq_x` 实际上近似于：

- `[N, 2, seq_len, 1]`

这意味着 sequence encoder 几乎只能看到“这个历史位置上出现过哪个广告”，看不到广告语义、与当前样本的关系、时间间隔等重要信息。

### 保留的序列组织方式

在当前三表设置下，继续保留两条异构序列作为基线：

- `click_seq`
- `exposure_seq`

这仍然是当前公开数据条件下最自然的划分方式。

### 每个 step 建议包含的字段

#### sparse step 字段

- 历史 `adgroup_id`
- 历史 `cate_id`
- 历史 `campaign_id`
- 历史 `customer`
- 历史 `brand`
- 历史 `pid_type`
- 历史 `pid_id`

#### dense step 字段

- `log1p(price)`
- `log1p(time_gap_sec)`
- `same_ad_as_target`
- `same_cate_as_target`
- `same_brand_as_target`
- `same_campaign_as_target`
- `same_customer_as_target`

### 模型侧处理方式

每个 step 的编码流程统一为：

1. sparse step fields 走 embedding
2. dense step fields 走投影层
3. 将二者融合成一个 `d_model` 大小的 step token

### 验收标准

- 历史 step 不再只有单一 ad ID。
- 至少包含一组 target-aware 匹配特征。
- `metadata.json` 清晰记录 sequence sparse/dense field 顺序。

---

### P2：基于 5 秒时间窗的事件级去重

**优先级：高**

### 目标

替换当前过于宽泛的 `(user, adgroup_id)` 全局去重，改成更贴合数据说明的事件级时间窗聚合。

### 为什么这样做

当前代码用 `(user, adgroup_id)` 去重，会把同一用户在不同时间对同一广告的真实多次曝光一起删掉，这会伤害：

- 曝光频次信息
- 曝光新近性
- 多次接触后的点击模式
- 历史序列真实性

但另一方面，数据说明又明确提示：

- 不能把 `userID + timestamp` 当严格主键
- 因为同一底层事件可能被不同部门记录成“时间非常接近但不完全相同”的多条日志

因此本轮方案直接采用：

> `5 秒时间窗口 + 事件级聚合`

### 具体规则

按以下 key 对样本排序并聚合：

- `user`
- `adgroup_id`
- `pid_type`
- `pid_id`

在同一组 key 下：

- 如果相邻两条记录的 `time_stamp` 差值 `<= 5 秒`
- 则认为它们属于同一个底层曝光事件簇

聚合规则：

- `time_stamp`：取簇内最早时间
- `label`：取 `max(clk)`，只要簇内任一条为点击，则该事件记为点击
- 保留事件级统计：
  - `dup_count`
  - `cluster_span_sec`

### 为什么直接固定 5 秒

这次按你的要求，先不做单独的重复分析流程，直接采用 5 秒窗口作为工程默认值。这样可以立刻替换掉当前过宽的去重逻辑，减少伪重复样本，同时保留跨更长时间的真实重复曝光。

### 预处理改动点

- 去重应在 `raw_sample` 清洗完成后、join 之前执行
- 这样可以减少 join 后的数据量，并让事件定义更清晰

### 验收标准

- 移除当前 `(user, adgroup_id)` 的宽去重逻辑
- 改为 5 秒窗口事件聚合
- metadata 中记录：
  - `dedup_window_sec`
  - 去重前后样本数
  - 平均事件簇大小

---

### P3：按时间切分训练/验证集

**优先级：高**

### 目标

默认采用时间前向切分，而不是随机切分。

### 为什么重要

CTR 离线评估最怕未来信息泄漏。随机切分会导致：

- 训练集和验证集时间分布混杂
- 离线指标偏乐观
- 调参结果不稳定

### 建议默认切法

基于当前数据的时间范围：

- 训练集：前 7 天
- 验证集：最后 1 天

如果未来需要更灵活，可加参数支持：

- `--split-mode time|random`
- `--val-days`

### 训练脚本需同步修改

- `pos_weight` 只能用训练集标签统计
- `run_metadata.json` 需记录：
  - train/val 时间范围
  - train/val 样本数
  - train/val 正样本率

### 验收标准

- 默认不再随机切分
- `pos_weight` 改为 train-only
- 训练元数据中能看到时间切分边界

---

### P4：non-seq token 语义化分组

**优先级：高**

### 目标

把当前“整个 non-seq 向量一次性投影成若干 token”的方式，改成按字段语义分组后分别建 token。

### 为什么重要

当前 non-seq token 虽然数量上已经是多个 token，但语义上并不稳定。HyFormer 的 non-seq token 如果要真正发挥作用，应该尽量对应清晰的业务语义块。

### 建议的 token 分组

建议至少拆成以下 8 类：

1. `user_profile_token`
   - 性别、年龄、消费等级、购物等级、职业、新客等级
2. `user_identity_token`
   - `user`
3. `target_ad_identity_token`
   - `adgroup_id`
4. `target_ad_attribute_token`
   - `cate_id`、`campaign_id`、`customer`、`brand`
5. `target_price_token`
   - `price`
6. `context_token`
   - `pid_type`、`pid_id`、`hour_of_day`、`day_of_week`
7. `history_summary_token_click`
   - 点击历史统计特征
8. `history_summary_token_exposure`
   - 曝光历史统计特征

### 对应的数据侧要求

预处理阶段要额外生成历史摘要特征，例如：

- `click_hist_len`
- `exposure_hist_len`
- `click_last_gap`
- `exposure_last_gap`
- `click_same_cate_count`
- `exposure_same_cate_count`
- `click_same_brand_count`
- `exposure_same_brand_count`

### 模型侧实现方式

每个 token group 有自己的投影逻辑：

- sparse embedding 聚合
- dense 特征投影
- 组内小 MLP

最终把这些语义 token 拼成 non-seq token 序列，再送入 backbone。

### 验收标准

- non-seq token 不再只是 reshape 产物
- 每个 token group 都有明确来源字段
- metadata 记录 token group 名称与字段组成

## 5. 推荐实施顺序

实现顺序应按依赖关系推进，而不是机械地按 P0-P4 编号推进。

### Phase 1：预处理重构

先做：

1. P2：5 秒事件去重
2. P0：non-seq sparse/dense 字段拆分
3. P1：sequence sparse/dense step 字段拆分
4. P4：补充历史摘要特征

产出：

- 结构化张量输出
- 新版 metadata
- `timestamps.pt`

### Phase 2：模型输入重构

再做：

1. P0：non-seq sparse embedding
2. P1：sequence step encoder
3. P4：语义化 non-seq token builder

产出：

- 新版 `models/taac_hyformer.py`
- 尽量不改 `main_pytorch.py` backbone 接口

### Phase 3：训练与评估重构

最后做：

1. P3：时间切分
2. train-only `pos_weight`
3. metadata / CLI / 文档同步

产出：

- 更可信的离线评估流程

## 6. 目标数据契约

P0-P4 完成后，推荐的稳定数据契约如下：

### non-seq

- `non_seq_sparse.pt`
  - shape: `[N, num_non_seq_sparse_fields]`
- `non_seq_dense.pt`
  - shape: `[N, num_non_seq_dense_fields]`

### sequence

- `seq_sparse.pt`
  - shape: `[N, num_sequences, seq_len, num_seq_sparse_fields]`
- `seq_dense.pt`
  - shape: `[N, num_sequences, seq_len, num_seq_dense_fields]`
- `seq_mask.pt`
  - shape: `[N, num_sequences, seq_len]`

### labels / split / metadata

- `labels.pt`
- `timestamps.pt`
- `metadata.json`

推荐 metadata 记录：

- 数据集摘要
- 去重配置与统计
- sparse field schema
- dense field schema
- sequence field schema
- token group schema
- 时间范围摘要

## 7. 未来实现时的验证清单

### 数据侧验证

- sparse 字段全部为整数 ID
- dense 字段全部为有限浮点数
- `seq_mask` 与真实历史长度一致
- 所有历史事件严格满足 `history_ts < current_ts`
- 去重后样本量与正负样本率合理

### 模型侧验证

- `python -m py_compile` 通过
- 小 batch forward 能跑通
- non-seq token 数量与 metadata 一致
- click / exposure 两条分支 shape 正确

### 训练侧验证

- 时间切分边界正确
- `pos_weight` 只来自 train split
- `run_metadata.json` 能看到 split 统计

## 8. 本文档的最终结论

如果只保留一句话，那么应该是：

> 在当前公开三表版本的 Taobao 广告数据下，接下来最值得做的不是继续改 HyFormer 主干，而是先把特征系统、事件去重和时间切分做对。

建议的落地顺序是：

1. `P2`：5 秒事件去重
2. `P0`：离散特征 embedding 化
3. `P1`：序列 step 特征变厚
4. `P3`：时间切分
5. `P4`：non-seq token 语义化

这是当前数据条件下性价比最高、也最稳妥的一条路线。
