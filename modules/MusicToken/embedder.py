# @title modules/MusicToken/embedder.py
"""
embedder.py — Audio projection network for the Musipainter pipeline.

Implements the FGAEmbedder, which projects BEATs audio features into the
CLIP text-embedding space used by Stable Diffusion's text encoder.
Architecture follows Musipainter paper Section 3.3 / Figure 3:
  BEATs concat(layer 4, 8, 12) → Linear → GELU → Linear → Attentive Pooling → [B, D]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FGAEmbedder(nn.Module):
    """
    Projects BEATs audio features into the Stable Diffusion text-embedding space.

    Input:  [B, T, input_size]  — concatenation of BEATs layers 4, 8, 12 (default dim 2304).
    Output: [B, output_size]    — single audio vector in CLIP embedding space.
              output_size = 768  for SD v1-4 (CompVis/stable-diffusion-v1-4)
              output_size = 1024 for SD 2   (stabilityai/stable-diffusion-2)

    Args:
        input_size:  feature dimension per timestep (default 768*3 = 2304).
        output_size: target embedding dimension matching the SD text encoder.
    """

    def __init__(self, input_size: int = 768 * 3, output_size: int = 768):
        super().__init__()

        # Two-layer projection: e_bar_audio = W2(GELU(W1(phi(a))))  [paper eq. 3.3]
        self.fc1  = nn.Linear(input_size, input_size)
        self.gelu = nn.GELU()
        self.fc2  = nn.Linear(input_size, output_size)

        # Attentive pooling: learns a scalar weight per timestep, then collapses T → 1.
        # Architecture: Linear → Tanh → Linear(→1) → Softmax(dim=T) → weighted sum
        self.attn = nn.Sequential(
            nn.Linear(output_size, output_size // 2),
            nn.Tanh(),
            nn.Linear(output_size // 2, 1),
        )

    def forward(self, audio_embs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_embs: [B, T, input_size] float32

        Returns:
            e_audio: [B, output_size] float32
        """
        # --- Projection (W1 → GELU → W2) ---
        x = self.fc1(audio_embs)   # [B, T, input_size]
        x = self.gelu(x)
        x = self.fc2(x)            # [B, T, output_size]

        # --- Attentive Pooling ---
        # Compute a normalised scalar weight for each timestep, then take the weighted sum.
        attn_logits  = self.attn(x)                        # [B, T, 1]
        attn_weights = F.softmax(attn_logits, dim=1)       # [B, T, 1], sums to 1 over T

        e_audio = (attn_weights * x).sum(dim=1)            # [B, output_size]
        return e_audio
