# @title modules/MusicToken/early_fusion_encoder.py
"""
early_fusion_encoder.py — Early-Fusion multimodal encoder for Musipainter.

Implements the FuseLIP-style Early Fusion architecture:
  - Audio tokens (BEATs layers 4+8+12 → projected) and
    text tokens (CLIP vocabulary embeddings → projected)
    are concatenated into a single sequence BEFORE any
    cross-modal attention.
  - A shared nn.TransformerEncoder processes the unified sequence,
    allowing every audio token to attend to every text token (and
    vice-versa) at each depth level.
  - A final linear projection maps the fused representation back
    to the CLIP hidden dimension expected by Stable Diffusion's
    cross-attention (768 for SD v1-4, 1024 for SD 2).

Architecture (FuseLIP §3 / Gemini suggestion):

  audio_tokens [B, T_a, audio_dim]  ──► audio_proj ──► + modality_audio_emb
                                                              │
                                                    cat(dim=1)│
                                                              │
  text_tokens  [B, T_t, text_dim]   ──►  text_proj ──► + modality_text_emb

  fused [B, T_a+T_t, d_model] ──► TransformerEncoder (num_layers) ──► out [B, T_a+T_t, d_model]

  out ──► mean-pool over seq ──► output_proj ──► [B, output_size]

References:
  - FuseLIP paper (document index 19 in codebase context)
  - https://docs.pytorch.org/docs/2.12/generated/torch.nn.TransformerEncoderLayer.html
  - https://docs.pytorch.org/docs/2.12/generated/torch.nn.Linear.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class EarlyFusionEncoder(nn.Module):
    """
    FuseLIP-style early-fusion multimodal encoder.

    Projects audio and text tokens to a shared d_model space, concatenates
    them into a single sequence, then processes the unified sequence with a
    nn.TransformerEncoder so that cross-modal attention operates at every
    layer depth — unlike the late-fusion injection used in the original
    Musipainter (which replaces a single <*> placeholder after CLIP has
    already processed text in isolation).

    Args:
        audio_dim:    feature dimension of BEATs output per timestep.
                      Default 2304 (concat of layers 4, 8, 12 × 768).
        text_dim:     CLIP token-embedding dimension.
                      768 for SD v1-4 (CompVis), 1024 for SD 2 / SD 2.1.
        output_size:  target conditioning dimension for Stable Diffusion
                      cross-attention.  Must equal text_dim.
                      768 for SD v1-4, 1024 for SD 2.
        d_model:      internal hidden dimension of the shared Transformer.
                      512 works well; use 768/1024 to match CLIP if VRAM allows.
        nhead:        number of attention heads.  Must divide d_model evenly.
        num_layers:   depth of the shared TransformerEncoder.
        dropout:      dropout applied inside each TransformerEncoderLayer.
        max_audio_len: maximum number of temporal audio frames (after pooling).
                       Used to register a learnable positional bias buffer.
        max_text_len:  maximum CLIP sequence length (default 77).
    """

    def __init__(
        self,
        audio_dim: int = 768 * 3,          # 2304: BEATs layers 4+8+12
        text_dim: int = 768,               # CLIP hidden size (SD v1-4)
        output_size: int = 768,            # SD cross-attention dim
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_audio_len: int = 376,          # 30 s × 16 kHz / 160 / 8 ≈ 376 frames
        max_text_len: int = 77,            # CLIP max sequence length
    ):
        super().__init__()

        self.d_model = d_model
        self.output_size = output_size

        # ── Projection layers (nn.Linear, as specified) ──────────────────────
        # Map each modality from its native dimension to the shared d_model.
        # These are the learnable "alignment" layers described in FuseLIP §3.
        self.audio_proj = nn.Linear(audio_dim, d_model)   # [B, T_a, d_model]
        self.text_proj  = nn.Linear(text_dim,  d_model)   # [B, T_t, d_model]

        # ── Modality-type embeddings ─────────────────────────────────────────
        # Learnable vectors broadcast-added to every token of each modality so
        # the Transformer can distinguish audio tokens from text tokens.
        # Shape: [1, 1, d_model] → broadcasts over batch and sequence dims.
        self.modality_audio_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.modality_text_emb  = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # ── Positional embeddings ────────────────────────────────────────────
        # Separate learnable position tables for audio and text.
        # max_seq = max_audio_len + max_text_len covers any concatenated length.
        self.audio_pos_emb = nn.Embedding(max_audio_len, d_model)
        self.text_pos_emb  = nn.Embedding(max_text_len,  d_model)

        # ── Shared Transformer Encoder ───────────────────────────────────────
        # nn.TransformerEncoderLayer with batch_first=True so that input shape
        # is (batch, seq_len, d_model) — consistent with the rest of the pipeline.
        # See: https://docs.pytorch.org/docs/2.12/generated/torch.nn.TransformerEncoderLayer.html
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,   # standard 4× expansion
            dropout=dropout,
            activation="gelu",             # matches CLIP / BEATs activation
            batch_first=True,              # (batch, seq, dim) convention
            norm_first=True,               # Pre-LN: more stable for audio+text
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,    # keep simple for DDP / compile compat
        )

        # ── Output projection ─────────────────────────────────────────────────
        # Map from d_model back to output_size (CLIP cross-attention dimension).
        # Used after mean-pooling the fused sequence.
        # See: https://docs.pytorch.org/docs/2.12/generated/torch.nn.Linear.html
        self.output_proj = nn.Linear(d_model, output_size)

        # Layer norm before output projection (stabilises training).
        self.output_norm = nn.LayerNorm(d_model)

        self._init_weights()

    # ──────────────────────────────────────────────────────────────────────────
    def _init_weights(self):
        """Xavier-uniform init for projection layers; small normal for embeddings."""
        for module in [self.audio_proj, self.text_proj, self.output_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.audio_pos_emb.weight, std=0.02)
        nn.init.normal_(self.text_pos_emb.weight,  std=0.02)

    # ──────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        audio_tokens: torch.Tensor,
        text_tokens:  torch.Tensor,
        audio_mask:   Optional[torch.Tensor] = None,
        text_mask:    Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Early-fusion forward pass.

        Args:
            audio_tokens: [B, T_a, audio_dim]  float32
                          Temporal BEATs features (concat layers 4+8+12).
            text_tokens:  [B, T_t, text_dim]   float32
                          CLIP token embeddings for the text prompt sequence.
            audio_mask:   [B, T_a] bool, True = PAD (passed as src_key_padding_mask).
                          Optional — not used for fixed-length audio.
            text_mask:    [B, T_t] bool, True = PAD token (padding positions).
                          Optional — forward still works without it.

        Returns:
            fused_embedding: [B, output_size] float32
                             Fused audio+text representation in the SD
                             cross-attention conditioning space.
        """
        B, T_a, _ = audio_tokens.shape
        B, T_t, _ = text_tokens.shape

        # ── 1. Project both modalities to shared d_model ──────────────────────
        a_emb = self.audio_proj(audio_tokens)   # [B, T_a, d_model]
        t_emb = self.text_proj(text_tokens)     # [B, T_t, d_model]

        # ── 2. Add modality-type embeddings ───────────────────────────────────
        a_emb = a_emb + self.modality_audio_emb   # broadcasts over B and T_a
        t_emb = t_emb + self.modality_text_emb    # broadcasts over B and T_t

        # ── 3. Add positional embeddings ──────────────────────────────────────
        a_pos = self.audio_pos_emb(
            torch.arange(T_a, device=audio_tokens.device)
        )  # [T_a, d_model]
        t_pos = self.text_pos_emb(
            torch.arange(T_t, device=text_tokens.device)
        )  # [T_t, d_model]

        a_emb = a_emb + a_pos.unsqueeze(0)   # [B, T_a, d_model]
        t_emb = t_emb + t_pos.unsqueeze(0)   # [B, T_t, d_model]

        # ── 4. EARLY FUSION: concatenate along the sequence dimension ─────────
        # This is the key FuseLIP operation: both modalities share the same
        # Transformer context, so audio token i can attend to text token j
        # at every layer — not just at the output.
        fused = torch.cat([a_emb, t_emb], dim=1)   # [B, T_a+T_t, d_model]

        # ── 5. Build key-padding mask for the concatenated sequence ───────────
        # Transformer ignores masked positions (True = ignore).
        if audio_mask is not None or text_mask is not None:
            # Default: nothing is masked
            if audio_mask is None:
                audio_mask = torch.zeros(B, T_a, dtype=torch.bool,
                                         device=fused.device)
            if text_mask is None:
                text_mask = torch.zeros(B, T_t, dtype=torch.bool,
                                        device=fused.device)
            key_padding_mask = torch.cat(
                [audio_mask, text_mask], dim=1
            )  # [B, T_a+T_t]
        else:
            key_padding_mask = None

        # ── 6. Shared Transformer Encoder (cross-modal self-attention) ─────────
        # Shape in/out: [B, T_a+T_t, d_model]
        out = self.transformer(
            fused,
            src_key_padding_mask=key_padding_mask,
        )  # [B, T_a+T_t, d_model]

        # ── 7. Pool the fused sequence to a single vector ─────────────────────
        # Mean-pool over unmasked positions only.
        if key_padding_mask is not None:
            # Invert mask: 1.0 for valid tokens, 0.0 for padding
            valid = (~key_padding_mask).float().unsqueeze(-1)   # [B, T, 1]
            pooled = (out * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            pooled = out.mean(dim=1)   # [B, d_model]

        # ── 8. Output projection → SD cross-attention dim ─────────────────────
        pooled = self.output_norm(pooled)          # LayerNorm before projection
        fused_embedding = self.output_proj(pooled) # [B, output_size]

        return fused_embedding
