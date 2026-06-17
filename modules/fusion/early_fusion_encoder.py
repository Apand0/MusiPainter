# @title modules/fusion/early_fusion_encoder.py
"""
early_fusion_encoder.py — Early-Fusion multimodal encoder for Musipainter (Strada B).

[STRADA B — Continuous Early Fusion]
- Audio: continuous features from BEATs (concatenation of layers 4, 8, 12).
- Audio projection: 2-layer MLP with GELU (Musipainter Section 3.3).
- Pooling: Attentive Pooling that collapses to 1 token (Musipainter Section 3.3).
- Fusion: FuseLIP-style early fusion by concatenation of continuous sequences.
- Transformer: shared encoder with bidirectional self-attention (AIAYN).
- Dynamic routing: n_audio_queries controls the temporal resolution sent to the UNet:
    0  → full audio sequence T_a tokens (pure FuseLIP)
    1  → single attentively-pooled token (pure Musipainter, default)
    >1 → Perceiver-style resampler producing N summary tokens

The cosine alignment loss (Musipainter eq. 2/5) always uses the attentively-pooled
token (audio_summary, shape [B, 1, output_size]) regardless of the routing mode,
because it must represent the pure audio modality before cross-modal fusion with text.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  Attentive Pooling (Musipainter Section 3.3)
# ─────────────────────────────────────────────────────────────────────────────

class AttentivePooling(nn.Module):
    """
    Attentive Pooling as described in Musipainter Section 3.3.

    Collapses the temporal dimension T_a into a single dense vector via a
    learned attention mechanism:
        e_audio = sum_t( softmax(w_t) * h_t )
    where w_t = Linear(Tanh(Linear(h_t))).

    Args:
        d_model: feature dimension of the input sequence.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    [B, T, d_model]
            mask: [B, T] bool, True where padding (optional).

        Returns:
            pooled: [B, 1, d_model]
        """
        attn_weights = self.attention(x)          # [B, T, 1]
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(-1), float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=1)   # normalise over T
        pooled = torch.sum(x * attn_weights, dim=1, keepdim=True)  # [B, 1, d_model]
        return pooled


# ─────────────────────────────────────────────────────────────────────────────
#  Audio Resampler — Perceiver-style (n_audio_queries > 1 path)
# ─────────────────────────────────────────────────────────────────────────────

class AudioResampler(nn.Module):
    """
    Perceiver-style cross-attention block that compresses [B, T_a, d_model]
    into [B, num_queries, d_model] using learnable query tokens.

    Architecture per block:
      Pre-LN cross-attention  (Q = learnable queries, K/V = audio projection)
      Residual on queries
      Pre-LN FFN (4x expansion, GELU)
      Residual on queries
      Output LayerNorm

    This matches the AudioResamplerBlock reference in the v10 codebase,
    ensuring the resampler has enough representational capacity.

    Args:
        d_model:     model hidden dimension.
        num_queries: number of output summary tokens N (must be > 1).
        nhead:       number of attention heads in the cross-attention block.
        dropout:     dropout rate applied inside the FFN.
    """

    def __init__(
        self,
        d_model: int,
        num_queries: int,
        nhead: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Learnable query embeddings — initialised small to avoid early saturation
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)

        # Pre-LN cross-attention
        self.norm_q  = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        # Pre-LN FFN (standard Transformer MLP block, AIAYN Sec. 3.3)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        # Output normalisation
        self.norm_out = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    [B, T_a, d_model] — projected audio sequence.
            mask: [B, T_a] bool, True where padding (optional).

        Returns:
            queries: [B, num_queries, d_model]
        """
        B = x.shape[0]
        queries = self.query_tokens.expand(B, -1, -1)   # [B, N, d_model]

        # Pre-LN cross-attention + residual
        q_norm  = self.norm_q(queries)
        kv_norm = self.norm_kv(x)
        attn_out, _ = self.cross_attn(
            query=q_norm,
            key=kv_norm,
            value=kv_norm,
            key_padding_mask=mask,
        )
        queries = queries + attn_out                    # residual

        # Pre-LN FFN + residual
        queries = queries + self.ff(self.norm_ff(queries))

        return self.norm_out(queries)                   # [B, N, d_model]


# ─────────────────────────────────────────────────────────────────────────────
#  EarlyFusionEncoder  (Strada B — Continuous Early Fusion)
# ─────────────────────────────────────────────────────────────────────────────

class EarlyFusionEncoder(nn.Module):
    """
    FuseLIP-style early-fusion encoder with Musipainter Attentive Pooling
    and optional Perceiver-style resampling (Strada B).

    Pipeline:
      1. Project audio [B, T_a, audio_dim] → [B, T_a, d_model]
         via a 2-layer MLP with GELU (Musipainter Section 3.3, eq. 3.3).
      2. Add audio modality embedding + audio positional embeddings.
      3. Attentive Pooling: collapse T_a → 1 token for the cosine loss
         (Musipainter Section 3.3). This step always runs regardless of routing.
      4. Project text [B, T_t, text_dim] → [B, T_t, d_model].
      5. Add text modality embedding + text positional embeddings.
      6. Dynamic routing — select audio tokens sent to the shared Transformer:
           n_audio_queries == 0  → full sequence a_proj  [B, T_a, d_model]
           n_audio_queries == 1  → audio_summary         [B, 1,   d_model]
           n_audio_queries >  1  → AudioResampler output [B, N,   d_model]
      7. Concatenate: [B, actual_T_a + 1 + T_t, d_model]
         (FuseLIP separator token between modalities, FuseLIP Sec. 3.1).
      8. Shared TransformerEncoder (bidirectional self-attention, AIAYN).
      9. LayerNorm + Linear → [B, actual_T_a + 1 + T_t, output_size]
         passed as encoder_hidden_states to the UNet (LDM Sec. 3.3 τ_θ).

    The cosine alignment loss (Musipainter eq. 2/5) always uses the
    attentively-pooled token (audio_summary) projected by loss_proj,
    since it must represent the pure audio modality before cross-modal
    fusion with text in the shared Transformer.

    Args:
        audio_dim:       BEATs concatenated dim (default 768*3=2304).
        text_dim:        CLIP token embedding dim (768 for SD 2.x).
        output_size:     UNet cross-attention dim; must equal unet.cross_attention_dim.
        d_model:         shared Transformer hidden dim (default 512).
        nhead:           number of attention heads in the shared Transformer.
        num_layers:      depth of the shared Transformer.
        dropout:         dropout rate throughout.
        max_audio_len:   positional table size for audio (default 512, covers any stride).
        max_text_len:    positional table size for text (default 77).
        n_audio_queries: routing selector:
                           0  → full T_a sequence (FuseLIP mode)
                           1  → single attentive-pooling token (Musipainter mode, default)
                           >1 → Perceiver resampler with N output tokens
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
        max_audio_len: int = 512,
        max_text_len: int = 77,
        n_audio_queries: int = 1,
    ):
        super().__init__()

        self.d_model         = d_model
        self.output_size     = output_size
        self.n_audio_queries = n_audio_queries

        # ── Step 1: Audio projection — 2-layer MLP with GELU (Musipainter) ──
        # e_bar_audio = W2 * GELU(W1 * phi(a))   [Musipainter Section 3.3]
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # ── Step 2: Audio modality / positional embeddings ───────────────────
        self.modality_audio_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.audio_pos_emb      = nn.Embedding(max_audio_len, d_model)

        # ── Step 3: Attentive Pooling — always runs, produces audio_summary ──
        # Used for the Musipainter cosine loss regardless of n_audio_queries.
        self.attentive_pooling = AttentivePooling(d_model)

        # ── Step 3b: AudioResampler — only instantiated when n_audio_queries > 1
        if self.n_audio_queries > 1:
            self.resampler = AudioResampler(
                d_model=d_model,
                num_queries=n_audio_queries,
                nhead=nhead,
                dropout=dropout,
            )

        # ── Steps 4-5: Text projection + modality / positional embeddings ────
        self.text_proj         = nn.Linear(text_dim, d_model)
        self.modality_text_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.text_pos_emb      = nn.Embedding(max_text_len, d_model)

        # FuseLIP separator token between modalities (FuseLIP Sec. 3.1)
        self.sep_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # ── Step 8: Shared cross-modal Transformer (AIAYN) ───────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,        # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # ── Step 9: Output projection ────────────────────────────────────────
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, output_size)

        # loss_proj: projects the PRE-transformer attentive-pooling token to
        # output_size for the Musipainter cosine alignment loss (eq. 2/5).
        # Kept separate from output_proj so that the alignment targets the pure
        # audio representation before cross-modal fusion with text.
        # Shape produced: [B, 1, output_size] regardless of n_audio_queries.
        self.loss_proj = nn.Linear(d_model, output_size)

        self._init_weights()

    # ─────────────────────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """Xavier-uniform initialisation for all projection layers."""
        for module in [
            self.audio_proj[0],
            self.audio_proj[2],
            self.text_proj,
            self.output_proj,
            self.loss_proj,          # FIX: was missing from the original loop
        ]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.audio_pos_emb.weight, std=0.02)
        nn.init.normal_(self.text_pos_emb.weight,  std=0.02)

    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        audio_tokens: torch.Tensor,
        text_tokens:  torch.Tensor,
        audio_mask:   Optional[torch.Tensor] = None,
        text_mask:    Optional[torch.Tensor] = None,
        return_audio_summary: bool = False,
    ):
        """
        Forward pass.

        Args:
            audio_tokens: [B, T_a, audio_dim]  float32 — BEATs concatenated features.
                          T_a depends on temporal_pool_stride used during preprocessing.
            text_tokens:  [B, T_t, text_dim]   float32 — CLIP token embeddings.
            audio_mask:   [B, T_a] bool, True = PAD (optional, only used when
                          n_audio_queries == 0, i.e. full-sequence mode).
            text_mask:    [B, T_t] bool, True = PAD (optional).
            return_audio_summary: if True, return the PRE-transformer attentively-
                          pooled audio token for the Musipainter cosine loss.

        Returns:
            fused_seq: [B, actual_T_a + 1 + T_t, output_size]
                       actual_T_a = n_audio_queries if n_audio_queries >= 1, else T_a.
                       Passed directly as encoder_hidden_states to the UNet
                       (LDM τ_θ, Rombach et al. Sec. 3.3).

            audio_out (only when return_audio_summary=True):
                       [B, 1, output_size]
                       PRE-transformer attentively-pooled audio token projected via
                       loss_proj. Always shape [B, 1, output_size], regardless of
                       the routing mode selected by n_audio_queries.
                       Used for the Musipainter cosine alignment loss (eq. 2/5):
                         CL = (1 - <e_audio / ||e_audio||, l_hat>)^2
                       where l_hat is the pre-normalised CLIP label vector.
        """
        B, T_a, _ = audio_tokens.shape
        _, T_t, _ = text_tokens.shape

        # ── Steps 1-2: Project audio + modality/positional embeddings ────────
        a_proj = self.audio_proj(audio_tokens)          # [B, T_a, d_model]
        a_proj = a_proj + self.modality_audio_emb       # broadcast [1, 1, d_model]
        a_pos  = self.audio_pos_emb(
            torch.arange(T_a, device=audio_tokens.device)
        )
        a_proj = a_proj + a_pos.unsqueeze(0)            # [B, T_a, d_model]

        # ── Step 3: Attentive Pooling — always executed ───────────────────────
        # audio_summary is PRE-transformer: pure audio for cosine loss.
        # Shape: [B, 1, d_model] regardless of n_audio_queries routing.
        audio_summary = self.attentive_pooling(a_proj, mask=audio_mask)  # [B, 1, d_model]

        # ── Step 6: Dynamic routing — select audio tokens for the UNet ───────
        if self.n_audio_queries == 1:
            # Musipainter mode: single attentively-pooled token.
            # Maximum stability; matches Musipainter Section 3.3 exactly.
            audio_for_fusion = audio_summary                            # [B, 1, d_model]
        elif self.n_audio_queries > 1:
            # Perceiver-style resampler: T_a → N summary tokens.
            audio_for_fusion = self.resampler(a_proj, mask=audio_mask) # [B, N, d_model]
        else:
            # n_audio_queries == 0: full audio sequence (pure FuseLIP mode).
            # Passes all T_a tokens to the shared Transformer.
            audio_for_fusion = a_proj                                   # [B, T_a, d_model]

        actual_T_a = audio_for_fusion.shape[1]

        # ── Steps 4-5: Project text + modality/positional embeddings ─────────
        t_proj = self.text_proj(text_tokens)            # [B, T_t, d_model]
        t_proj = t_proj + self.modality_text_emb
        t_pos  = self.text_pos_emb(
            torch.arange(T_t, device=text_tokens.device)
        )
        t_proj = t_proj + t_pos.unsqueeze(0)            # [B, T_t, d_model]

        # ── Step 7: Concatenate — [B, actual_T_a + 1 + T_t, d_model] ────────
        # Separator token distinguishes modality boundaries (FuseLIP Sec. 3.1).
        sep   = self.sep_token.expand(B, -1, -1)        # [B, 1, d_model]
        fused = torch.cat([audio_for_fusion, sep, t_proj], dim=1)

        # ── Build key-padding mask for the shared Transformer ────────────────
        # Audio and separator positions are never padding (pooled/resampled tokens
        # are always valid). Only text padding needs to be masked.
        # In full-sequence mode (n_audio_queries == 0), audio_mask is respected.
        key_padding_mask = None
        needs_mask = text_mask is not None or (
            audio_mask is not None and self.n_audio_queries == 0
        )
        if needs_mask:
            # Audio side: use audio_mask when in full-sequence mode, else all-False.
            if self.n_audio_queries == 0 and audio_mask is not None:
                audio_side_mask = audio_mask                            # [B, T_a]
            else:
                audio_side_mask = torch.zeros(
                    B, actual_T_a, dtype=torch.bool, device=fused.device
                )
            sep_mask  = torch.zeros(B, 1, dtype=torch.bool, device=fused.device)
            text_side = (
                text_mask
                if text_mask is not None
                else torch.zeros(B, T_t, dtype=torch.bool, device=fused.device)
            )
            key_padding_mask = torch.cat(
                [audio_side_mask, sep_mask, text_side], dim=1
            )

        # ── Step 8: Shared cross-modal Transformer ───────────────────────────
        out = self.transformer(fused, src_key_padding_mask=key_padding_mask)
        # [B, actual_T_a + 1 + T_t, d_model]

        # ── Step 9: Output projection ─────────────────────────────────────────
        out       = self.output_norm(out)
        fused_seq = self.output_proj(out)               # [B, actual_T_a + 1 + T_t, output_size]

        # Zero out text padding positions in the output sequence
        if text_mask is not None:
            full_mask = torch.cat([
                torch.zeros(B, actual_T_a + 1, dtype=torch.bool, device=fused_seq.device),
                text_mask,
            ], dim=1)
            valid     = (~full_mask).float().unsqueeze(-1)  # [B, actual_T_a + 1 + T_t, 1]
            fused_seq = fused_seq * valid

        if return_audio_summary:
            # Project PRE-transformer attentive-pooling token via loss_proj.
            # Shape: [B, 1, output_size] — always, regardless of routing mode.
            pure_audio = self.loss_proj(audio_summary)  # [B, 1, output_size]
            return fused_seq, pure_audio

        return fused_seq    # [B, actual_T_a + 1 + T_t, output_size]
