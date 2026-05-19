# @title modules/preprocess/utils.py
"""
utils.py — Shared utility helpers for the MusicToken pipeline.

Provides two layers of tensor serialisation:
  - safetensors (preferred): zero-copy, lazy mmap, no pickle.
  - torch.save / torch.load (legacy): kept for backward-compat with existing .pt checkpoints.
"""

from __future__ import annotations

import os
import torch
from pathlib import Path

# Safetensors helpers (preferred format)
def save_safetensors(tensors: dict, path, metadata: dict | None = None) -> None:
    """
    Save a {str: torch.Tensor} dict as a safetensors file with an explicit fsync.

    Args:
        tensors:  mapping of id → tensor (typically float16).
        path:     output path (.safetensors).
        metadata: optional str→str dict stored in the file header (e.g. stride, sample_rate).
                  Non-string values are coerced with str().
    """
    from safetensors.torch import save_file

    # safetensors requires contiguous tensors; .contiguous() is a no-op when already OK.
    tensors_contiguous = {k: v.contiguous() for k, v in tensors.items()}

    sf_meta: dict[str, str] | None = None
    if metadata:
        sf_meta = {k: str(v) for k, v in metadata.items()}

    save_file(tensors_contiguous, str(path), metadata=sf_meta)

    # Explicit fsync for safety on GCS / Kaggle filesystems.
    with open(path, 'r+b') as f:
        f.flush()
        os.fsync(f.fileno())


def open_safetensors(path, framework: str = "pt", device: str = "cpu"):
    """
    Open a safetensors file in lazy mmap mode and return a SafeOpen handle.

    The handle keeps the file memory-mapped; each call to get_tensor(key) reads
    only the bytes for that tensor (seek + read), without loading the entire file.

    API of the returned object:
        sf.keys()           → list of all ids (strings)
        sf.get_tensor(key)  → torch.Tensor for that key
        sf.metadata()       → str→str header dict (may be None)

    The object is NOT a context manager; keep it as an attribute of the owning class.

    Args:
        path:      path to the .safetensors file.
        framework: "pt" (PyTorch) — the only framework used in this pipeline.
        device:    "cpu" — tensors are moved to GPU by the training loop.
    """
    from safetensors import safe_open
    return safe_open(str(path), framework=framework, device=device)


def load_safetensors(path, device: str = "cpu") -> dict:
    """
    Load a .safetensors file and return a {str: torch.Tensor} dict.

    Symmetric counterpart of save_safetensors. Preferred over torch.load
    for weight files because it is zero-copy, pickle-free and supports
    memory-mapped lazy loading.

    Args:
        path:   path to a .safetensors file produced by save_safetensors.
        device: target device for the loaded tensors (default "cpu").
                Pass "cuda:0" to load directly on GPU.

    Returns:
        dict mapping parameter name → torch.Tensor.
    """
    from safetensors.torch import load_file
    return load_file(str(path), device=device)


# torch.save / torch.load helpers (legacy — backward compat with .pt files)
def save_chunk_safe(chunk: dict, path) -> None:
    """Save a tensor dict to disk with an explicit fsync. Legacy .pt format."""
    with open(path, 'wb') as f:
        torch.save(chunk, f)
        f.flush()
        os.fsync(f.fileno())


def safe_torch_load(path, map_location: str = "cpu") -> dict:
    """Load a .pt checkpoint safely (weights_only=True where possible). Legacy format."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        return torch.load(path, map_location=map_location)
