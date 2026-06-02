# @title modules/fusion/early_fusion_encoder.py
"""
early_fusion_encoder.py — Early-Fusion multimodal encoder for Musipainter.

[AUDIO-RESAMPLER v10]

Problem solved:
  With stride=4, the full fused sequence fed to the UNet was T_a+T_t ≈ 171
  tokens. The UNet cross-attention was trained on 77-token CLIP sequences;
  passing 171 tokens causes audio to numerically dominate over text, producing
  images that are stylistically coherent but structurally fragmented (the
  behaviour reported by Gemini in the project diary).

Solution — Audio Resampler (Perceiver-style cross-attention):
  A fixed set of N_q LEARNABLE QUERY TOKENS (default N_q=32) performs
  multi-head cross-attention over the full audio sequence [B, T_a, d_model].
  The T_a keys/values carry ALL temporal information; the N_q queries
  summarise it into a compact representation [B, N_q, d_model].
  After resampling, the sequence fed to the shared Transformer is:
      [B, N_q + 1 + T_t, d_model]  →  (N_q=32) + (sep=1) + (T_t=77) = 110 tokens
  which is close to the 77 tokens CLIP used, regardless of the stride.

Why this is consistent with the Musipainter paper:
  The paper (Section 3.3 / Fig. 3) describes an Attentive Pooling step that
  collapses T → 1 in the Late Fusion FGAEmbedder. The Audio Resampler is the
  generalised N_q > 1 version of that pooling: instead of a single weighted
  sum it produces N_q summary tokens via multi-head attention, preserving more
  information while keeping the sequence length bounded. Every audio frame
  participates in the resampling via cross-attention — nothing is truncated.

Why this is consistent with FuseLIP Early Fusion:
  FuseLIP concatenates modality tokens BEFORE the shared Transformer. Here we
  still do that: the N_q audio summary tokens + sep + T_t text tokens are all
  fused in the shared Transformer. Early fusion is fully preserved.
  The separator token between modalities follows FuseLIP Sec. 3.1.

Why this is consistent with LDM / Stable Diffusion conditioning (Rombach et al.):
  LDM Sec. 3.3 introduces a domain-specific encoder τ_θ that projects the
  conditioning signal y to a sequence τ_θ(y) ∈ R^{M×d_τ}, which is then
  fed to the UNet's cross-attention layers. EarlyFusionEncoder IS τ_θ: it
  takes (audio, text) and produces encoder_hidden_states of shape
  [B, N_q+1+T_t, output_size] — a fixed-length sequence that the UNet
  cross-attention attends to via Q=W_Q·ϕ_i(z_t), K=W_K·τ_θ(y),
  V=W_V·τ_θ(y) (LDM eq. before eq. 3).

Why the cosine loss uses PRE-transformer audio tokens (Musipainter eq. 2):
  Musipainter eq. (2/5) aligns the pure audio embedding e_audio with the
  pre-normalised CLIP label vector l̂. Using the audio_summary BEFORE the
  shared Transformer ensures that e_audio is uncontaminated by cross-modal
  fusion with text — it represents the audio modality alone, which is exactly
  what the paper's cosine alignment term targets. Using post-transformer tokens
  would align a text-informed mixed representation rather than pure audio.
  audio_summary is therefore correctly taken from the resampler output
  (post AudioResampler, pre shared-Transformer).

The output shape [B, N_q+1+T_t, output_size] is passed directly as
encoder_hidden_states to the UNet — no change to MusicTokenWrapper.forward().
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  Audio Resampler Block
# ─────────────────────────────────────────────────────────────────────────────

class AudioResamplerBlock(nn.Module):
    """
    One Perceiver-style cross-attention block.

    Queries: N_q learnable tokens [B, N_q, d_model]
    Keys/Values: full projected audio sequence [B, T_a, d_model]

    The block applies:
      1. LayerNorm on queries (Pre-LN, consistent with AIAYN / Pre-LN convention)
      2. Multi-head cross-attention  (Q from queries, K/V from audio)
      3. Residual on queries
      4. LayerNorm + FFN + residual  (standard Transformer MLP block, AIAYN Sec. 3.3)

    After stacking `ef_resampler_layers` of these blocks, the queries carry a
    rich, T_a-informed representation compressed into N_q slots.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.norm_q  = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,          # [B, N_q, d_model]
        audio_kv: torch.Tensor,         # [B, T_a, d_model]
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, T_a] bool
    ) -> torch.Tensor:
        # Pre-LN cross-attention
        q_norm  = self.norm_q(queries)
        kv_norm = self.norm_kv(audio_kv)
        attn_out, _ = self.cross_attn(
            query=q_norm,
            key=kv_norm,
            value=kv_norm,
            key_padding_mask=key_padding_mask,
        )
        queries = queries + attn_out                      # residual

        # Pre-LN FFN
        queries = queries + self.ff(self.norm_ff(queries))  # residual
        return queries


# ─────────────────────────────────────────────────────────────────────────────
#  Audio Resampler
# ─────────────────────────────────────────────────────────────────────────────

class AudioResampler(nn.Module):
    """
    Compresses [B, T_a, d_model] → [B, N_q, d_model] via stacked cross-attn.

    The N_q learnable query embeddings are the only parameters that fix the
    output length; T_a can be anything (stride-agnostic).
    """

    def __init__(
        self,
        d_model: int,
        n_queries: int,
        nhead: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_queries = n_queries
        # Learnable query embeddings — initialised small to avoid early saturation
        self.query_tokens = nn.Parameter(
            torch.randn(1, n_queries, d_model) * 0.02
        )
        self.blocks = nn.ModuleList([
            AudioResamplerBlock(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)

    def forward(
        self,
        audio_proj: torch.Tensor,                       # [B, T_a, d_model]
        audio_mask: Optional[torch.Tensor] = None,      # [B, T_a] bool pad mask
    ) -> torch.Tensor:
        B = audio_proj.shape[0]
        # Expand learnable queries to batch size
        queries = self.query_tokens.expand(B, -1, -1)   # [B, N_q, d_model]

        for block in self.blocks:
            queries = block(queries, audio_proj, key_padding_mask=audio_mask)

        return self.norm_out(queries)                    # [B, N_q, d_model]


# ─────────────────────────────────────────────────────────────────────────────
#  EarlyFusionEncoder  (v10 — Audio Resampler)
# ─────────────────────────────────────────────────────────────────────────────

class EarlyFusionEncoder(nn.Module):
    """
    FuseLIP-style early-fusion encoder with Audio Resampler.

    Pipeline:
      1. Project audio [B,T_a,2304] → [B,T_a,d_model]
      2. Add audio modality embedding + audio positional embeddings
         (positional table size = max_audio_len, covers any stride)
      3. AudioResampler: cross-attend N_q learnable queries over T_a frames
         → [B, N_q, d_model]  (PRE-Transformer; used for cosine loss)
      4. Project text [B,T_t,768/1024] → [B,T_t,d_model]
      5. Add text modality embedding + text positional embeddings
      6. Concatenate: [B, N_q + 1 + T_t, d_model]  (N_q audio + sep + T_t text)
         fixed length regardless of stride (FuseLIP-style early fusion)
      7. Shared TransformerEncoder (cross-modal self-attention, AIAYN architecture)
      8. LayerNorm + Linear → [B, N_q + 1 + T_t, output_size]
         passed as encoder_hidden_states to the UNet (LDM Sec. 3.3 τ_θ)

    Args:
        audio_dim:          BEATs concat dim (default 768*3=2304).
        text_dim:           CLIP token embedding dim (768 for SD2, 1024 for SD2.1).
        output_size:        UNet cross-attention dim; must equal text_dim.
        d_model:            shared Transformer hidden dim.
        nhead:              heads in the shared Transformer (AIAYN Sec. 3.2).
        num_layers:         depth of the shared Transformer.
        dropout:            dropout rate throughout.
        n_audio_queries:    N_q — number of audio summary tokens (default 32).
        resampler_heads:    heads in the resampler cross-attention (default 8).
        resampler_layers:   stacked resampler blocks (default 2).
        max_audio_len:      positional table size for audio (≥ max T_a, default 512).
        max_text_len:       positional table size for text (default 77).
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
        n_audio_queries: int = 32,
        resampler_heads: int = 8,
        resampler_layers: int = 2,
        max_audio_len: int = 512,
        max_text_len: int = 77,
    ):
        super().__init__()

        self.d_model      = d_model
        self.output_size  = output_size
        self.n_audio_queries = n_audio_queries

        # ── Step 1-2: Audio projection + modality/positional embeddings ──────
        self.audio_proj         = nn.Linear(audio_dim, d_model)
        self.modality_audio_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.audio_pos_emb      = nn.Embedding(max_audio_len, d_model)

        # ── Step 3: Audio Resampler ───────────────────────────────────────────
        self.audio_resampler = AudioResampler(
            d_model=d_model,
            n_queries=n_audio_queries,
            nhead=resampler_heads,
            num_layers=resampler_layers,
            dropout=dropout,
        )

        # ── Step 4-5: Text projection + modality/positional embeddings ────────
        self.text_proj          = nn.Linear(text_dim, d_model)
        self.modality_text_emb  = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.text_pos_emb       = nn.Embedding(max_text_len, d_model)

        # Learnable modality embedding for the N_q summary tokens (distinct
        # from the positional/modality emb used before resampling)
        self.modality_query_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # [FUSELIP-SEP] Separator token between modalities (FuseLIP Sec. 3.1)
        self.sep_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # ── Step 7: Shared cross-modal Transformer ────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,        # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # ── Step 8: Output projection ─────────────────────────────────────────
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, output_size)
        # loss_proj: projects PRE-transformer audio summary tokens to output_size
        # for the Musipainter cosine loss (eq. 2/5). Kept separate from output_proj
        # so that the cosine alignment is computed on the pure audio representation
        # before cross-modal fusion with text in the shared Transformer.
        self.loss_proj = nn.Linear(d_model, output_size)

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
        return_audio_summary: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            audio_tokens: [B, T_a, audio_dim]  float32 — BEATs features.
                          T_a depends on temporal_pool_stride:
                            stride=1 → ~376, stride=4 → ~94, stride=8 → ~47
            text_tokens:  [B, T_t, text_dim]   float32 — CLIP token embeddings.
            audio_mask:   [B, T_a] bool, True = PAD (optional).
            text_mask:    [B, T_t] bool, True = PAD (optional).
            return_audio_summary: if True, also return the PRE-transformer audio
                          summary tokens for the Musipainter cosine loss.

        Returns:
            fused_seq:   [B, N_q + 1 + T_t, output_size]
                         N_q audio summary tokens + separator + T_t text tokens,
                         fully fused by the shared Transformer.
                         Sequence length is FIXED at N_q+1+T_t regardless of stride.
                         Passed directly as encoder_hidden_states to the UNet
                         (LDM Sec. 3.3 τ_θ: τ_θ(y) ∈ R^{M×d_τ}).

            audio_out   (only if return_audio_summary=True):
                         [B, N_q, output_size]
                         PRE-transformer audio summary tokens projected via
                         loss_proj. These represent pure audio (before cross-modal
                         fusion with text) and are used for the Musipainter cosine
                         alignment loss (eq. 2/5): CL = (1 - <e_audio, l̂>)^2.
                         Using pre-transformer tokens ensures alignment targets
                         the audio modality alone, not the text-fused representation.
        """
        B, T_a, _ = audio_tokens.shape
        _, T_t, _ = text_tokens.shape

        # ── Steps 1-2: Project audio + positional/modality embeddings ─────────
        a_proj = self.audio_proj(audio_tokens)                  # [B, T_a, d_model]
        a_proj = a_proj + self.modality_audio_emb               # broadcast [1,1,d]
        a_pos  = self.audio_pos_emb(
            torch.arange(T_a, device=audio_tokens.device)
        )
        a_proj = a_proj + a_pos.unsqueeze(0)                    # [B, T_a, d_model]

        # ── Step 3: Audio Resampler ───────────────────────────────────────────
        # All T_a frames participate via cross-attention; output is fixed N_q.
        # audio_summary is PRE-transformer: pure audio representation used for
        # the cosine loss (Musipainter eq. 2/5). See return docstring above.
        audio_summary = self.audio_resampler(
            audio_proj=a_proj,
            audio_mask=audio_mask,
        )                                                        # [B, N_q, d_model]

        # Add a dedicated modality embedding to the N_q summary tokens so the
        # shared Transformer can distinguish them from text tokens.
        audio_summary = audio_summary + self.modality_query_emb  # broadcast

        # ── Steps 4-5: Project text + positional/modality embeddings ──────────
        t_proj = self.text_proj(text_tokens)                    # [B, T_t, d_model]
        t_proj = t_proj + self.modality_text_emb
        t_pos  = self.text_pos_emb(
            torch.arange(T_t, device=text_tokens.device)
        )
        t_proj = t_proj + t_pos.unsqueeze(0)                    # [B, T_t, d_model]

        # ── Step 6: Concatenate — fixed length N_q + 1 + T_t ─────────────────
        # Audio summary tokens come first, then separator, then text.
        # Separator token helps the Transformer distinguish modalities
        # (FuseLIP Sec. 3.1: special tokens to separate modalities).
        sep = self.sep_token.expand(B, -1, -1)                  # [B, 1, d_model]
        fused = torch.cat([audio_summary, sep, t_proj], dim=1)  # [B, N_q+1+T_t, d_model]

        # ── Build key-padding mask for the shared Transformer ─────────────────
        # Audio summary tokens are never padding (resampler always produces N_q
        # valid tokens). Text padding is preserved.
        key_padding_mask = None
        if text_mask is not None:
            # [B, N_q+1+T_t]: False for audio summary + separator, text_mask for text
            audio_query_mask = torch.zeros(
                B, self.n_audio_queries + 1,  # +1 for separator
                dtype=torch.bool, device=fused.device
            )
            key_padding_mask = torch.cat([audio_query_mask, text_mask], dim=1)

        # ── Step 7: Shared cross-modal Transformer ────────────────────────────
        out = self.transformer(fused, src_key_padding_mask=key_padding_mask)
        # [B, N_q+1+T_t, d_model]

        # ── Step 8: Output projection ─────────────────────────────────────────
        out = self.output_norm(out)
        fused_seq = self.output_proj(out)                        # [B, N_q+1+T_t, output_size]

        # Zero out text padding positions (audio summary + separator are never padding)
        if text_mask is not None:
            full_mask = torch.cat([
                torch.zeros(B, self.n_audio_queries + 1, dtype=torch.bool, device=fused_seq.device),  # +1 sep
                text_mask
            ], dim=1)
            valid = (~full_mask).float().unsqueeze(-1)           # [B, N_q+1+T_t, 1]
            fused_seq = fused_seq * valid

        if return_audio_summary:
            # Project PRE-transformer audio summary via loss_proj (separate from
            # output_proj) to output_size for use in the cosine alignment loss.
            # Shape: [B, N_q, output_size] — pure audio, pre cross-modal fusion.
            pure_audio = self.loss_proj(audio_summary)           # [B, N_q, output_size]
            return fused_seq, pure_audio

        return fused_seq    # [B, N_q+1+T_t, output_size]
