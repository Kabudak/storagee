# Taobao HyFormer 本轮改动汇总

## 1. 改动目标

本轮改动围绕以下目标展开：

1. 将输入从扁平 float 向量升级为结构化 `sparse/dense` 特征
2. 将当前宽泛的 `(user, adgroup_id)` 去重改为 `5` 秒时间窗事件级去重
3. 将历史序列从单一 `adgroup_id` 扩展为多字段 step 特征
4. 将训练默认切分方式改为按时间切分
5. 将 non-seq token 改造成有明确语义的 token group

## 2. 新增文档

新增了两份文档：

- [Taobao_HyFormer_P0_P4_Plan.md](</D:/hyformer/storagee/Taobao_HyFormer_P0_P4_Plan.md>)
- [Taobao_HyFormer_Change_Summary.md](</D:/hyformer/storagee/Taobao_HyFormer_Change_Summary.md>)

其中：

- `Taobao_HyFormer_P0_P4_Plan.md` 是方案设计稿
- 当前文件是实际代码落地后的改动摘要

## 3. 预处理改动

文件：

- [scripts/preprocess_taobao.py](</D:/hyformer/storagee/scripts/preprocess_taobao.py>)

### 3.1 去重逻辑

原先做法：

- 按 `(user, adgroup_id)` 全局去重

现改为：

- 按 `user + adgroup_id + pid_type + pid_id`
- 在 `5` 秒时间窗内做事件级聚合

聚合规则：

- `time_stamp` 取最早时间
- `label` 取 `max(clk)`
- 额外保留：
  - `dup_count`
  - `cluster_span_sec`

### 3.2 输出契约

原先主要输出：

- `non_seq_x.pt`
- `seq_x.pt`
- `labels.pt`

现在输出：

- `non_seq_sparse.pt`
- `non_seq_dense.pt`
- `seq_sparse.pt`
- `seq_dense.pt`
- `seq_mask.pt`
- `labels.pt`
- `timestamps.pt`
- `metadata.json`

### 3.3 non-seq 特征

新增结构化 non-seq 字段：

- sparse:
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
- dense:
  - `price_log`
  - `hour_of_day`
  - `day_of_week`
  - click / exposure 历史摘要
  - 当前事件聚合统计

### 3.4 sequence 特征

保留两条历史分支：

- `click_seq`
- `exposure_seq`

每个历史 step 从原先的一维 `adgroup_id` 扩展为：

- sparse:
  - `adgroup_id`
  - `cate_id`
  - `campaign_id`
  - `customer`
  - `brand`
  - `pid_type`
  - `pid_id`
- dense:
  - `price_log`
  - `time_gap_log`
  - `same_ad_as_target`
  - `same_cate_as_target`
  - `same_brand_as_target`
  - `same_campaign_as_target`
  - `same_customer_as_target`

### 3.5 本地时区修正

自查后补了一个关键修正：

- `hour_of_day`
- `day_of_week`

不再按 UTC 计算，而是按 `Asia/Shanghai` 语义计算。  
实现方式是对时间戳加 `8` 小时偏移后再取小时和星期。

## 4. 模型改动

文件：

- [models/taac_hyformer.py](</D:/hyformer/storagee/models/taac_hyformer.py>)

### 4.1 non-seq 输入编码

原先：

- 一个 `Linear(non_seq_dim, num_non_seq_tokens * d_model)`

现在：

- sparse field 走 embedding
- dense field 走投影
- 再按 token group 构建语义化 non-seq tokens

当前 token groups 包括：

- `user_profile_token`
- `user_identity_token`
- `target_ad_identity_token`
- `target_ad_attribute_token`
- `target_price_token`
- `context_token`
- `history_summary_token_click`
- `history_summary_token_exposure`
- `current_event_token`

### 4.2 sequence 输入编码

新增结构化 step encoder：

- sparse step fields -> embedding
- dense step fields -> projection
- 融合成 `d_model` 大小的历史 step token

### 4.3 自查后新增修正

自查中我认为还有两个逻辑值得补，因此已经加上：

1. **sequence position embedding**
   - 让历史顺序对模型显式可见
2. **sequence type embedding**
   - 区分 `click_seq` 和 `exposure_seq`

这是因为原始 backbone 本身没有显式的位置编码，而如果完全不加，历史顺序感会偏弱。

## 5. 训练脚本改动

文件：

- [scripts/run_taobao.py](</D:/hyformer/storagee/scripts/run_taobao.py>)

### 5.1 输入格式适配

训练脚本已改成读取新的结构化张量：

- `non_seq_sparse.pt`
- `non_seq_dense.pt`
- `seq_sparse.pt`
- `seq_dense.pt`
- `seq_mask.pt`
- `labels.pt`
- `timestamps.pt`

### 5.2 时间切分

默认切分方式改为：

- `--split-mode time`

即按时间做 train / val 切分。

同时自查后补了：

- 日切边界按 `Asia/Shanghai` 语义计算
- 不再按 UTC 直接做 `timestamp // 86400`

补充一处这轮 review 新增的小修正：

- 当使用 `--max-rows` 做小规模调试时，训练脚本改为沿完整时间轴等距抽样，而不是只截取最早一段样本，避免时间切分失真。

### 5.3 pos_weight

改为仅根据训练集标签统计 `pos_weight`，避免把验证集信息带进去。

## 6. 命令文档改动

文件：

- [Test.md](</D:/hyformer/storagee/Test.md>)

已同步更新为新流程，包括：

- 新版预处理命令
- 新版训练命令
- 时间切分说明
- 新输出文件列表

## 7. 自查后确认保留的设计选择

这轮自查后，我认为以下设计目前可以保留，不需要继续改：

1. 只保留两条序列：
   - `click_seq`
   - `exposure_seq`
2. 高基数 sparse 特征采用哈希桶而不是显式全量词表
3. 使用共享的 step encoder，再通过 sequence type embedding 区分分支
4. Query Generator 先维持 `non_seq pooled summary + per-seq pooled summary` 路线

## 8. 这轮没有做的事

以下内容这轮没有动：

- `main_pytorch.py` backbone 主体结构
- `QueryBoostMixer` 的具体机制
- 输出 head 的更深层改造
- 长时间训练实测
- 大规模预处理实跑

## 9. 已完成的验证

这轮只做了语法级验证，没有跑训练。

已执行：

```bash
python -m py_compile scripts/preprocess_taobao.py models/taac_hyformer.py scripts/run_taobao.py
```

结果：通过。

## 10. 当前仍需你本地关注的风险点

虽然我已经做了逻辑自查，但下面几件事仍然需要你本地跑时重点盯一下：

1. `5` 秒事件去重后，样本量变化是否符合预期
2. 新哈希桶大小是否会导致你机器上的 embedding 显存占用过大
3. 时间切分后的最后一天正负样本比例是否异常
4. 新版 `seq_len`、`num_non_seq_tokens` 是否和你的实验配置一致
5. 新特征体系下 loss 是否还能稳定下降

## 11. 一句话总结

这轮改动本质上不是“继续雕 backbone”，而是把项目从：

> 扁平数值输入 + 宽去重 + 随机切分

升级成了：

> 结构化 sparse/dense 输入 + 5 秒事件级去重 + 时间前向切分 + 语义化 token 输入

这会让后续的 HyFormer 训练结果更有解释性，也更接近 CTR 场景里真正影响效果的部分。
