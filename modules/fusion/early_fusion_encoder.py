# @title modules/fusion/early_fusion_encoder.py
"""
early_fusion_encoder.py — Early-Fusion multimodal encoder for Musipainter.

FuseLIP-style Early Fusion: audio tokens and text tokens are concatenated
into a single sequence BEFORE any cross-modal attention, so every audio
token can attend to every text token at every Transformer layer depth.

The encoder now returns the FULL fused sequence [B, T_a+T_t, output_size]
instead of a single pooled vector. This lets the UNet cross-attention layers
attend to different positions of the fused sequence for different spatial
regions of the generated image — matching how Stable Diffusion was designed
to consume the 77-token CLIP sequence.

Compatibility with any temporal_pool_stride:
  stride=1  → T_a≈376 → total seq ≈ 453  (heavy, not recommended)
  stride=4  → T_a≈94  → total seq ≈ 171  (good balance)
  stride=8  → T_a≈47  → total seq ≈ 124  (fast)
  stride=16 → T_a≈23  → total seq ≈ 100  (minimal)

The UNet cross-attention dim is fixed (768 for SD2 / 1024 for SD2.1) but
sequence length is flexible — SD accepts any length, not just 77.

References:
  FuseLIP paper (Section 3)
  Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models"
  https://docs.pytorch.org/docs/2.12/generated/torch.nn.TransformerEncoderLayer.html
"""

import torch
import torch.nn as nn
from typing import Optional


class EarlyFusionEncoder(nn.Module):
    """
    FuseLIP-style early-fusion multimodal encoder.

    Projects audio and text tokens to a shared d_model space, concatenates
    them, runs a shared TransformerEncoder for cross-modal self-attention,
    then projects every token to output_size.

    Returns [B, T_a+T_t, output_size] — the full fused sequence ready to
    be used as encoder_hidden_states for the UNet cross-attention layers.

    Args:
        audio_dim:    BEATs feature dim per timestep (default 2304 = 3×768).
        text_dim:     CLIP token embedding dim (768 for SD2, 1024 for SD2.1).
        output_size:  SD cross-attention dim; must equal text_dim.
        d_model:      internal Transformer hidden dim.
        nhead:        number of attention heads (must divide d_model).
        num_layers:   Transformer depth.
        dropout:      dropout inside each TransformerEncoderLayer.
        max_audio_len: max audio frames (used for positional embedding table).
        max_text_len:  max text tokens (default 77, CLIP limit).
    """

    def __init__(
        self,
        audio_dim: int = 768 * 3,
        text_dim: int = 768,
        output_size: int = 768,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_audio_len: int = 512,   # generous upper bound for any stride
        max_text_len: int = 77,
    ):
        super().__init__()

        self.d_model = d_model
        self.output_size = output_size

        # Project each modality from its native dim to the shared d_model.
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.text_proj  = nn.Linear(text_dim,  d_model)

        # Learnable scalars added to every token of each modality so the
        # Transformer can tell audio tokens apart from text tokens.
        self.modality_audio_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.modality_text_emb  = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Separate learnable positional tables for audio and text.
        # max_audio_len=512 comfortably covers stride=1 (≈376 frames).
        self.audio_pos_emb = nn.Embedding(max_audio_len, d_model)
        self.text_pos_emb  = nn.Embedding(max_text_len,  d_model)

        # Shared Transformer: cross-modal self-attention at every layer.
        # norm_first=True (Pre-LN) is more stable for mixed-modality input.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # [EARLY-FUSION-SEQ] Project every token in the fused sequence to
        # output_size. This replaces the old mean-pool + single projection.
        # Shape: [B, T_a+T_t, d_model] → [B, T_a+T_t, output_size]
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, output_size)

        self._init_weights()

    def _init_weights(self):
        for module in [self.audio_proj, self.text_proj, self.output_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.audio_pos_emb.weight, std=0.02)
        nn.init.normal_(self.text_pos_emb.weight,  std=0.02)

    def forward(
        self,
        audio_tokens: torch.Tensor,
        text_tokens:  torch.Tensor,
        audio_mask:   Optional[torch.Tensor] = None,
        text_mask:    Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass. Returns the full fused sequence for UNet cross-attention.

        Args:
            audio_tokens: [B, T_a, audio_dim]  float32  — BEATs features.
            text_tokens:  [B, T_t, text_dim]   float32  — CLIP token embeddings.
            audio_mask:   [B, T_a] bool, True = PAD position (optional).
            text_mask:    [B, T_t] bool, True = PAD position (optional).

        Returns:
            fused_seq: [B, T_a+T_t, output_size] float32
                       Full sequence of fused audio+text tokens.
                       Pass directly as encoder_hidden_states to the UNet.
                       T_a depends on temporal_pool_stride used at preprocessing.
        """
        B, T_a, _ = audio_tokens.shape
        _, T_t, _ = text_tokens.shape

        # 1. Project both modalities to shared d_model
        a_emb = self.audio_proj(audio_tokens)   # [B, T_a, d_model]
        t_emb = self.text_proj(text_tokens)     # [B, T_t, d_model]

        # 2. Add modality-type embeddings (audio vs text distinction)
        a_emb = a_emb + self.modality_audio_emb
        t_emb = t_emb + self.modality_text_emb

        # 3. Add positional embeddings
        a_pos = self.audio_pos_emb(torch.arange(T_a, device=audio_tokens.device))
        t_pos = self.text_pos_emb( torch.arange(T_t, device=text_tokens.device))
        a_emb = a_emb + a_pos.unsqueeze(0)     # [B, T_a, d_model]
        t_emb = t_emb + t_pos.unsqueeze(0)     # [B, T_t, d_model]

        # 4. Early Fusion: concatenate along sequence dimension.
        # Audio tokens come first so their positional indices are stable
        # regardless of text length.
        fused = torch.cat([a_emb, t_emb], dim=1)   # [B, T_a+T_t, d_model]

        # 5. Build key-padding mask for the concatenated sequence.
        # Transformer ignores positions where mask=True.
        key_padding_mask = None
        if audio_mask is not None or text_mask is not None:
            if audio_mask is None:
                audio_mask = torch.zeros(B, T_a, dtype=torch.bool, device=fused.device)
            if text_mask is None:
                text_mask = torch.zeros(B, T_t, dtype=torch.bool, device=fused.device)
            key_padding_mask = torch.cat([audio_mask, text_mask], dim=1)

        # 6. Cross-modal self-attention through the shared Transformer.
        out = self.transformer(fused, src_key_padding_mask=key_padding_mask)
        # [B, T_a+T_t, d_model]

        # [EARLY-FUSION-SEQ] Project every token to output_size.
        # No mean-pooling: the full sequence is returned so the UNet
        # cross-attention can attend to different tokens for different
        # spatial regions of the generated image.
        out = self.output_norm(out)          # LayerNorm before projection
        fused_seq = self.output_proj(out)    # [B, T_a+T_t, output_size]

        # Zero-out padding positions so the UNet does not attend to them.
        if key_padding_mask is not None:
            valid = (~key_padding_mask).float().unsqueeze(-1)   # [B, T, 1]
            fused_seq = fused_seq * valid

        return fused_seq   # [B, T_a+T_t, output_size]
