# @title modules/fusion/cross_attention_encoder.py
"""
cross_attention_encoder.py — Asymmetric Audio-Guided Cross-Attention encoder
for the Musipainter pipeline.

Design rationale
----------------
The text conditions the audio, NOT the other way round.
Q = Audio tokens  →  the sequence whose content we want to enrich.
K = V = Text tokens  →  the conditioning source.

This follows the cross-attention formulation in "Attention Is All You Need"
(Vaswani et al., Sec. 3.2) and mirrors the role of τ_θ in Latent Diffusion
Models (Rombach et al., Sec. 3.3): a domain-specific encoder that maps the
conditioning signal y to a sequence τ_θ(y) ∈ R^{M×d_τ} passed as
encoder_hidden_states to the UNet cross-attention layers.

No Audio Resampler is used here: the full temporal audio sequence [B, T_a, D]
is preserved and conditioned by text at every time step.  The output shape
[B, T_a, output_size] is stride-dependent (T_a varies with temporal_pool_stride)
but the UNet cross-attention is position-agnostic and accepts any sequence
length, so this is not a problem.

Text positional encoding
------------------------
Text tokens enter already carrying CLIP positional embeddings (they come from
`token_embedding` in MusicTokenWrapper._get_text_embeddings, which is the raw
nn.Embedding from the frozen CLIP text encoder).  We therefore do NOT add a
second positional table for text inside this encoder — it would double-count
positions and corrupt the CLIP embedding space.  Only audio receives its own
positional table because audio embeddings are raw BEATs features with no
inherent positional encoding.
"""

import torch
import torch.nn as nn
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  Cross-Attention block
# ─────────────────────────────────────────────────────────────────────────────

class AudioTextCrossAttentionBlock(nn.Module):
    """
    Single Pre-LN cross-attention block where audio queries text.

    Architecture (AIAYN Sec. 3.2 / Pre-LN convention):
        audio  →  LayerNorm  →  Q
        text   →  LayerNorm  →  K, V
        MultiHeadAttention(Q, K, V)  +  residual on audio
        LayerNorm  →  FFN (d_model → 4·d_model → d_model)  +  residual

    Args:
        d_model:  shared hidden dimension.
        nhead:    number of attention heads (must divide d_model evenly).
        dropout:  dropout probability applied inside attention and FFN.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()

        # Separate LayerNorms for Q and KV — standard Pre-LN cross-attention.
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
        audio_q:   torch.Tensor,                        # [B, T_a, d_model]
        text_kv:   torch.Tensor,                        # [B, T_t, d_model]
        text_mask: Optional[torch.Tensor] = None,       # [B, T_t] bool, True = PAD
        audio_key_padding_mask: Optional[torch.Tensor] = None,  # reserved, unused
    ) -> torch.Tensor:
        """
        Args:
            audio_q:   audio token sequence acting as queries.
            text_kv:   text token sequence acting as keys and values.
            text_mask: padding mask for text (True = padding position, ignored
                       in attention). Shape [B, T_t].
            audio_key_padding_mask: reserved for future use; currently ignored
                       because audio frames are never padded in this pipeline.

        Returns:
            audio_q: text-conditioned audio sequence [B, T_a, d_model].
        """
        # Pre-LN cross-attention
        q_norm  = self.norm_q(audio_q)
        kv_norm = self.norm_kv(text_kv)

        attn_out, _ = self.cross_attn(
            query=q_norm,
            key=kv_norm,
            value=kv_norm,
            key_padding_mask=text_mask,   # mask padding on text side
        )
        # Residual: preserves the original audio structure, adds textual context.
        audio_q = audio_q + attn_out

        # Pre-LN FFN
        audio_q = audio_q + self.ff(self.norm_ff(audio_q))
        return audio_q


# ─────────────────────────────────────────────────────────────────────────────
#  Full encoder
# ─────────────────────────────────────────────────────────────────────────────

class FullAudioGuidedCrossAttentionEncoder(nn.Module):
    """
    Text-guided audio encoder for Musipainter (τ_θ in LDM nomenclature).

    Pipeline
    --------
    1. Project audio  [B, T_a, audio_dim] → [B, T_a, d_model]
       + learned audio positional embeddings.
    2. Project text   [B, T_t, text_dim]  → [B, T_t, d_model]
       (no extra positional table — CLIP positions already embedded upstream).
    3. Stack `num_layers` AudioTextCrossAttentionBlocks:
       Q = audio, K = V = text  →  text-conditioned audio [B, T_a, d_model].
    4. LayerNorm + Linear → [B, T_a, output_size]  (passed to UNet).

    Additionally, `loss_proj` projects the *pre-cross-attention* audio
    representation to `output_size` for the Musipainter cosine alignment loss
    (eq. 2/5):  CL = (1 - <e_audio/‖e_audio‖, l̂>)².
    Using the pre-cross-attention representation ensures alignment targets pure
    audio content, not the text-informed fused output.

    Args:
        audio_dim:     BEATs concat dim, default 768*3 = 2304.
        text_dim:      CLIP token embedding dim (768 for SD v1-4, 1024 for SD 2.x).
        output_size:   UNet cross_attention_dim; must equal text_dim for SD 2.x.
        d_model:       shared hidden dim inside the encoder.
        nhead:         attention heads (must divide d_model).
        num_layers:    number of stacked AudioTextCrossAttentionBlocks.
        dropout:       dropout rate throughout.
        max_audio_len: size of the audio positional embedding table.
                       Must be >= the maximum expected T_a.
                       stride=1 → T_a≈376; stride=4 → T_a≈94; stride=8 → T_a≈47.
                       Default 512 covers all common strides with headroom.
    """

    def __init__(
        self,
        audio_dim:     int   = 768 * 3,
        text_dim:      int   = 768,
        output_size:   int   = 768,
        d_model:       int   = 512,
        nhead:         int   = 8,
        num_layers:    int   = 4,
        dropout:       float = 0.1,
        max_audio_len: int   = 512,
    ):
        super().__init__()

        self.d_model     = d_model
        self.output_size = output_size

        # ── Step 1: Audio projection + positional embeddings ──────────────────
        self.audio_proj    = nn.Linear(audio_dim, d_model)
        self.audio_pos_emb = nn.Embedding(max_audio_len, d_model)

        # ── Step 2: Text projection (no positional table — see module docstring)
        self.text_proj = nn.Linear(text_dim, d_model)

        # ── Step 3: Stacked cross-attention blocks ────────────────────────────
        self.cross_attn_layers = nn.ModuleList([
            AudioTextCrossAttentionBlock(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])

        # ── Step 4: Output projection → UNet ─────────────────────────────────
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, output_size)

        # Auxiliary projection for Musipainter cosine loss (pre-cross-attention)
        self.loss_proj = nn.Linear(d_model, output_size)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in [self.audio_proj, self.text_proj, self.output_proj, self.loss_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.audio_pos_emb.weight, std=0.02)

    def forward(
        self,
        audio_tokens: torch.Tensor,
        text_tokens:  torch.Tensor,
        audio_mask:   Optional[torch.Tensor] = None,
        text_mask:    Optional[torch.Tensor] = None,
        return_audio_summary: bool = False,
    ):
        """
        Args:
            audio_tokens: [B, T_a, audio_dim]  float32 — BEATs features.
            text_tokens:  [B, T_t, text_dim]   float32 — CLIP token embeddings
                          (already carry CLIP positional info).
            audio_mask:   [B, T_a] bool, True = PAD.  Currently not forwarded
                          to cross-attention because audio frames are never padded
                          in this pipeline.  Accepted for future compatibility.
            text_mask:    [B, T_t] bool, True = PAD.  Forwarded as
                          key_padding_mask inside each cross-attention block so
                          CLIP padding tokens are ignored.
            return_audio_summary: if True, also return the PRE-cross-attention
                          audio projection for the Musipainter cosine loss.

        Returns:
            fused_seq:    [B, T_a, output_size]  — text-conditioned audio
                          sequence passed as encoder_hidden_states to the UNet.

            pure_audio_out (only when return_audio_summary=True):
                          [B, T_a, output_size]  — PRE-cross-attention audio
                          projected via loss_proj, for cosine alignment loss.
        """
        B, T_a, _ = audio_tokens.shape

        # ── Step 1: Project audio + add positional embeddings ─────────────────
        a_proj = self.audio_proj(audio_tokens)                   # [B, T_a, d_model]
        a_pos  = self.audio_pos_emb(
            torch.arange(T_a, device=audio_tokens.device)
        )
        a_proj = a_proj + a_pos.unsqueeze(0)                     # broadcast [B, T_a, d_model]

        # Snapshot of PURE audio representation (before any text influence).
        # Using .clone() makes the intent explicit and guards against accidental
        # in-place operations modifying this tensor in future edits.
        pure_audio = a_proj.clone()                              # [B, T_a, d_model]

        # ── Step 2: Project text ──────────────────────────────────────────────
        t_proj = self.text_proj(text_tokens)                     # [B, T_t, d_model]

        # ── Step 3: Text-guided cross-attention (Q=audio, K=V=text) ──────────
        audio_cond = a_proj
        for layer in self.cross_attn_layers:
            audio_cond = layer(
                audio_q=audio_cond,
                text_kv=t_proj,
                text_mask=text_mask,
            )
        # audio_cond: [B, T_a, d_model] — each audio frame conditioned by text

        # ── Step 4: Output projection for UNet ───────────────────────────────
        fused_seq = self.output_proj(self.output_norm(audio_cond))  # [B, T_a, output_size]

        if return_audio_summary:
            pure_audio_out = self.loss_proj(pure_audio)          # [B, T_a, output_size]
            return fused_seq, pure_audio_out

        return fused_seq
