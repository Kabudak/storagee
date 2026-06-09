# HyFormer PyTorch Reimplementation

This repository is a PyTorch reading-and-reimplementation project based on the paper:

- Paper: `HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction`
- arXiv: `https://arxiv.org/abs/2601.12681`

The goal of this repository is to keep the same engineering granularity as `OneTrans_Pytorch` while replacing the backbone with a HyFormer-style unified architecture.

## What This Repository Contains

There are two layers in this codebase:

1. The backbone implementation
   This contains the core HyFormer building blocks such as sequence encoders, query decoding, and query boosting.
2. The training wrapper
   This adds dataset loading, feature tensorization, metrics, mixed precision, checkpointing, and CLI training support.

This repository is therefore not only a backbone demo. It is a runnable training project built around a HyFormer-style architecture.

## Repository Structure

### `main_pytorch.py`

This file is the backbone reference implementation. It contains:

- `SwiGLUFeedForward`
- `CrossAttentionBlock`
- `FullTransformerEncoder`
- `LongerStyleEncoder`
- `SequenceRepresentationEncoder`
- `QueryBoostMixer`
- `HyFormerLayer`
- `HyFormerBackbone`

This is the closest file to the paper-level architectural core.

### `models/`

Task-level model wrappers live here.

Current file:

- `models/taac_hyformer.py`

This wraps the backbone into a trainable classifier:

- projects non-sequential features into non-sequence tokens
- projects each grouped sequence into its own sequence-token stream
- generates dedicated query/global tokens for each sequence branch from non-sequential features and sequence mean-pooling summaries
- applies stacked HyFormer layers
- pools the final boosted token representations
- applies a classification head

### `utils/`

Reusable non-model logic lives here.

- `utils/common.py`
  General helpers such as seed setup and split generation.
- `utils/metrics.py`
  Accuracy and AUC computation.
- `utils/taac_data.py`
  Dataset loading, schema handling, feature conversion, multi-sequence grouping, and tensor construction.

### `scripts/`

Runnable entrypoints live here.

Current file:

- `scripts/preprocess_taobao.py`
- `scripts/run_taobao.py`

These scripts are the current Taobao three-table preprocessing and training entrypoints. They handle:

- reading `raw_sample.csv`, `ad_feature.csv`, and `user_profile.csv`
- reconstructing click/exposure histories from prior `raw_sample` rows
- structured sparse/dense tensorization
- HyFormer model construction
- training and validation loops
- metadata and optional checkpoint save

## Backbone Architecture

At a high level, the model flow is:

1. Map non-sequential features into `num_non_seq_tokens` feature tokens.
2. Group sequential features into `num_sequences` behavior branches.
3. Map each behavior branch into its own sequence tokens.
4. Generate `num_queries_per_seq` query/global tokens for each sequence branch.
5. Alternate:
   - `Query Decoding`: each sequence branch's query tokens cross-attend to its own sequence representation
   - `Query Boosting`: decoded query tokens from all sequences and non-sequence tokens are mixed together
6. Pool the final boosted tokens for downstream prediction.

### Input Shapes

- non-sequential input:
  `[batch_size, non_seq_dim]`
- sequential input:
  `[batch_size, num_sequences, seq_len, seq_feature_dim]`

### Query Generation

Initial queries are generated from:

- the non-sequential token set
- mean-pooled summaries from every sequence branch

The current implementation passes global information through one MLP generator per sequence branch. The global context is built from all non-sequential tokens and pooled summaries from every sequence branch. Each generator produces `num_queries_per_seq` query/global tokens, and these tokens are reused and updated across HyFormer layers.

### Sequence Representation Encoding

The backbone supports three sequence encoding modes through `seq_encoder_type`:

- `longer`
  Compress the full sequence through short-query cross-attention.
- `full_transformer`
  Standard self-attention sequence modeling.
- `swiglu`
  Attention-free feed-forward sequence modeling.

### Query Boosting

After sequence-aware decoding, HyFormer mixes:

- decoded query tokens from every sequence branch
- non-sequence tokens

through a lightweight token-mixing block inspired by the paper's MLP-Mixer-style design. This allows:

- cross-query interaction
- cross-sequence interaction
- query-to-feature interaction

within every HyFormer layer.

## Data Tensorization

The public Taobao display-ad data used here usually exposes three CSV files rather than the richer behavior log used in the paper. The preprocessing script therefore:

- joins `raw_sample.csv` with ad and user-profile features
- builds non-sequential sparse/dense fields for the target impression
- reconstructs two explicit sequence branches from earlier `raw_sample` rows:
  - `click_seq`
  - `exposure_seq`
- emits structured sparse/dense tensors plus metadata under `data/taobao_processed`

This is a practical adaptation of the HyFormer multi-sequence interface. It is not identical to the paper's original behavior-log setting.

## Training Pipeline

The current training target is the public Taobao display-ad click data in its common three-table form:

- `raw_sample.csv`
- `ad_feature.csv`
- `user_profile.csv`

Because this public version does not usually include `raw_behavior_log`, click/exposure sequence branches are reconstructed from earlier rows in `raw_sample`. This preserves the HyFormer multi-sequence interface, but it is not semantically identical to the paper's richer behavior-log setting.

### Checkpointing

When `--save-checkpoint` is set, the current training script saves the best model by validation AUC with a filename such as:

- `best_model_epoch03_auc0.6123.pt`

## Recommended Entry Points

### Backbone sanity check

```bash
python main_pytorch.py
```

This runs a small synthetic demo and prints the shape transitions through the backbone.

### Training run

```bash
python scripts/preprocess_taobao.py
python scripts/run_taobao.py --epochs 5 --batch-size 256
```

### Training with a specific sequence encoder

```bash
python scripts/run_taobao.py --epochs 5 --batch-size 256 --seq-encoder-type longer
python scripts/run_taobao.py --epochs 5 --batch-size 256 --seq-encoder-type full_transformer
python scripts/run_taobao.py --epochs 5 --batch-size 256 --seq-encoder-type swiglu
```

## What This README Tries To Clarify

This repository is not an official ByteDance release. It is:

- a faithful PyTorch reimplementation of the paper's main ideas
- adapted into the same project structure as `OneTrans_Pytorch`
- with runnable data, training, and checkpoint engineering added around it

If you are trying to understand where a piece of code lives:

- start with `main_pytorch.py` for the architectural core
- then read `models/taac_hyformer.py` for the task wrapper
- then read `scripts/preprocess_taobao.py` and `scripts/run_taobao.py` for the data and training workflow
