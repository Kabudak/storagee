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

- `scripts/run_taac2026_sample.py`

This script is the main training entrypoint. It handles:

- dataset download or local parquet reading
- feature tensorization
- HyFormer model construction
- AMP setup
- training and validation loops
- checkpoint save and resume

## Backbone Architecture

At a high level, the model flow is:

1. Map non-sequential features into `num_non_seq_tokens` feature tokens.
2. Group sequential features into `num_sequences` behavior branches.
3. Map each behavior branch into its own sequence tokens.
4. Generate `global_tokens_per_seq` query/global tokens for each sequence branch.
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

- the raw non-sequential feature vector
- mean-pooled summaries from every sequence branch

The current implementation passes global information through one MLP generator per sequence branch. Each generator produces `global_tokens_per_seq` query/global tokens, and these tokens are reused and updated across HyFormer layers.

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

The original TAAC sample data does not explicitly expose the same production multi-sequence semantics described in the paper. To keep the project runnable on the public sample while preserving the HyFormer interface, this repository groups detected sequence features into multiple branches automatically.

Current behavior:

- flattened schema:
  uses `domain_*` columns as sequence channels
- raw schema:
  uses parsed sequence feature names from `seq_feature`
- grouping:
  evenly assigns detected sequence channels into `--num-sequences` groups

This keeps the training script faithful to the paper's multi-sequence interface while remaining runnable on the available sample data.

## Training Pipeline

The current training target is the Hugging Face dataset:

- `TAAC2026/data_sample_1000`

The training script supports:

- remote dataset loading
- local parquet loading
- automatic schema handling
- multi-sequence grouping
- mixed precision training on CUDA
- checkpoint save with timestamp-based filenames
- checkpoint resume with optimizer and scaler state restoration

### Mixed Precision

AMP is enabled by default on CUDA.

Supported modes:

- default CUDA AMP
- `--amp-dtype fp16`
- `--amp-dtype bf16`
- `--no-amp`

### Checkpointing

Checkpoints are saved with timestamped filenames such as:

- `best_model_20260414_164500.pt`

Resume behavior:

- `--resume` accepts either a filename under `output-dir`
- or a full checkpoint path

The script restores:

- model weights
- optimizer state
- scaler state
- last completed epoch
- best validation AUC

## Recommended Entry Points

### Backbone sanity check

```bash
python main_pytorch.py
```

This runs a small synthetic demo and prints the shape transitions through the backbone.

### Training run

```bash
python scripts/run_taac2026_sample.py --epochs 5 --batch-size 32
```

### Training with a specific sequence encoder

```bash
python scripts/run_taac2026_sample.py --epochs 5 --batch-size 32 --seq-encoder-type longer
python scripts/run_taac2026_sample.py --epochs 5 --batch-size 32 --seq-encoder-type full_transformer
python scripts/run_taac2026_sample.py --epochs 5 --batch-size 32 --seq-encoder-type swiglu
```

### Resume training

```bash
python scripts/run_taac2026_sample.py --epochs 10 --batch-size 32 --resume best_model_20260414_164500.pt --save-checkpoint
```

## What This README Tries To Clarify

This repository is not an official ByteDance release. It is:

- a faithful PyTorch reimplementation of the paper's main ideas
- adapted into the same project structure as `OneTrans_Pytorch`
- with runnable data, training, and checkpoint engineering added around it

If you are trying to understand where a piece of code lives:

- start with `main_pytorch.py` for the architectural core
- then read `models/taac_hyformer.py` for the task wrapper
- then read `scripts/run_taac2026_sample.py` for the training workflow
