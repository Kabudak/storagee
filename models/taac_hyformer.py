from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main_pytorch import HyFormerBackbone, ensure_non_empty_mask, masked_mean_pool


class TAACHyFormerClassifier(nn.Module):
    def __init__(
        self,
        non_seq_dim: int,
        seq_feature_dim: int,
        num_classes: int,
        seq_len: int,
        num_sequences: int,
        global_tokens_per_seq: int,
        num_non_seq_tokens: int,
        d_model: int,
        num_heads: int,
        ffn_hidden: int,
        hyformer_layers: int,
        seq_encoder_type: str = "longer",
        short_seq_len: int = 8,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.num_sequences = num_sequences
        self.global_tokens_per_seq = global_tokens_per_seq
        self.num_non_seq_tokens = num_non_seq_tokens
        self.d_model = d_model

        self.non_seq_tokenizer = nn.Linear(non_seq_dim, num_non_seq_tokens * d_model)
        self.seq_tokenizers = nn.ModuleList([nn.Linear(seq_feature_dim, d_model) for _ in range(num_sequences)])

        global_info_dim = non_seq_dim + num_sequences * d_model
        self.query_generators = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(global_info_dim, ffn_hidden),
                    nn.SiLU(),
                    nn.Linear(ffn_hidden, global_tokens_per_seq * d_model),
                )
                for _ in range(num_sequences)
            ]
        )

        self.backbone = HyFormerBackbone(
            num_layers=hyformer_layers,
            num_sequences=num_sequences,
            num_queries_per_sequence=self.global_tokens_per_seq,
            num_non_seq_tokens=self.num_non_seq_tokens,
            d_model=d_model,
            num_heads=num_heads,
            ffn_hidden=ffn_hidden,
            encoder_type=seq_encoder_type,
            short_seq_len=short_seq_len,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, num_classes),
        )

    def build_sequence_tokens(self, seq_x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        sequence_tokens: list[torch.Tensor] = []
        sequence_masks: list[torch.Tensor] = []
        pooled_sequences: list[torch.Tensor] = []

        for seq_idx in range(self.num_sequences):
            current_seq_x = seq_x[:, seq_idx, :, :]
            current_mask = ensure_non_empty_mask(current_seq_x.abs().sum(dim=-1) > 0)
            current_tokens = self.seq_tokenizers[seq_idx](current_seq_x)
            sequence_tokens.append(current_tokens)
            sequence_masks.append(current_mask)
            pooled_sequences.append(masked_mean_pool(current_tokens, current_mask))

        return sequence_tokens, sequence_masks, pooled_sequences

    def build_query_tokens(self, non_seq_x: torch.Tensor, pooled_sequences: list[torch.Tensor]) -> list[torch.Tensor]:
        global_info = torch.cat([non_seq_x] + pooled_sequences, dim=-1)
        return [
            generator(global_info).view(non_seq_x.size(0), self.global_tokens_per_seq, self.d_model)
            for generator in self.query_generators
        ]

    def forward(self, non_seq_x: torch.Tensor, seq_x: torch.Tensor) -> torch.Tensor:
        batch_size = non_seq_x.size(0)
        non_seq_tokens = self.non_seq_tokenizer(non_seq_x).view(
            batch_size,
            self.num_non_seq_tokens,
            self.d_model,
        )
        sequence_tokens, sequence_masks, pooled_sequences = self.build_sequence_tokens(seq_x)
        query_tokens = self.build_query_tokens(non_seq_x, pooled_sequences)
        boosted_tokens = self.backbone(query_tokens, non_seq_tokens, sequence_tokens, sequence_masks)
        pooled = boosted_tokens.mean(dim=1)
        return self.head(pooled)
