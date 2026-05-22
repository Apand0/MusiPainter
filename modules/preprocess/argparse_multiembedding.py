# @title argparse_multiembedding.py
"""
Drop-in argument and dataloader patch for multi-dataset audio embeddings.

Provides add_multiembedding_args(parser) and build_embedding_index(args, logger)
to support single or multiple Kaggle Dataset embedding directories with optional
lazy mmap loading.
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def _str2bool(v):
    """Flexible bool parser that accepts 'true/false/1/0/yes/no'."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', '1'):
        return True
    if v.lower() in ('no', 'false', '0'):
        return False
    raise argparse.ArgumentTypeError(f"Boolean expected, got: '{v}'")


def add_multiembedding_args(parser: argparse.ArgumentParser) -> None:
    """Register multi-dataset embedding arguments onto *parser*."""
    try:
        parser.add_argument(
            "--embeddings_dir",
            type=str,
            default="./audio_embeddings/",
            help=(
                "Comma-separated paths to directories containing precomputed "
                "BEATs audio embeddings. Each directory may hold a merged "
                "audio_embeddings.safetensors, per-class chunks "
                "(<Class>_chunk_*.safetensors), or legacy .pt files. "
                "Multiple Kaggle Datasets can be combined by listing their "
                "mount paths: "
                "--embeddings_dir /kaggle/input/emb-part1,/kaggle/input/emb-part2"
            ),
        )
    except argparse.ArgumentError:
        pass

    try:
        parser.add_argument(
            "--embeddings_preload_all",
            type=_str2bool,
            default=True,
            help=(
                "True (default): load all tensors into RAM at init "
                "(fast; zero I/O during training). "
                "False: lazy mmap — each tensor read on demand "
                "(necessary when total embedding size exceeds available RAM). "
                "On Kaggle with stride=8 (~0.82 MB/audio × 25k = ~20 GB), "
                "use False."
            ),
        )
    except argparse.ArgumentError:
        pass

    try:
        parser.add_argument(
            "--embeddings_max_sf_handles",
            type=int,
            default=8,
            help=(
                "LRU cache size for open safetensors file handles "
                "(only relevant when --embeddings_preload_all false). "
                "Each handle keeps one .safetensors file memory-mapped. "
                "Increase to 16–32 if you have many small per-class chunk files."
            ),
        )
    except argparse.ArgumentError:
        pass


def build_embedding_index(args, log=None):
    """Construct and return a LazyEmbeddingIndex from parsed args."""
    from dataloader_colab import LazyEmbeddingIndex

    _log = log or logger

    embeddings_dir = getattr(args, 'embeddings_dir', None)
    if not embeddings_dir:
        _log.warning("build_embedding_index: args.embeddings_dir is empty — returning None.")
        return None

    preload_all    = getattr(args, 'embeddings_preload_all',    True)
    max_sf_handles = getattr(args, 'embeddings_max_sf_handles', 8)

    _log.info(
        f"Building LazyEmbeddingIndex:\n"
        f"  dirs            : {embeddings_dir}\n"
        f"  preload_all     : {preload_all}\n"
        f"  max_sf_handles  : {max_sf_handles}"
    )

    index = LazyEmbeddingIndex(
        embeddings_dirs = embeddings_dir,
        preload_all     = preload_all,
        max_sf_handles  = max_sf_handles,
    )

    _log.info(
        f"LazyEmbeddingIndex ready: {len(index)} audio IDs  "
        f"({'preloaded' if preload_all else 'lazy mmap'})"
    )
    return index


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Self-test")
    add_multiembedding_args(parser)
    args = parser.parse_args([
        "--embeddings_dir", "/tmp/emb_a,/tmp/emb_b",
        "--embeddings_preload_all", "false",
        "--embeddings_max_sf_handles", "16",
    ])

    print(f"embeddings_dir            : {args.embeddings_dir}")
    print(f"embeddings_preload_all    : {args.embeddings_preload_all}")
    print(f"embeddings_max_sf_handles : {args.embeddings_max_sf_handles}")
    print("argparse patch OK.")
