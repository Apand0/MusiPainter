#!/usr/bin/env python
# @title musipainter_test.py
"""
musipainter_test.py — UNIQUE inference/test entry point for Musipainter.

Same logic as musipainter_train.py, but for generation/inference:

    --architecture Musipainter-EF   -> Early Fusion (test_ef.py + MusicToken_no_accel_ef.py)
    --architecture Musipainter-CA   -> Audio-Guided Cross-Attention
                                       (test_ca.py + MusicToken_no_accel_ca.py)

All other arguments (--learned_embeds, --data_dir, --embeddings_dir,
--prompt, --guidance_scale, etc.) are identical to those of the two
original scripts and are forwarded unchanged to the selected branch's script.

Examples:
    python musipainter_test.py --architecture Musipainter-EF \\
        --data_dir ./Museart/ --embeddings_dir ./audio_embeddings/ \\
        --learned_embeds ./output/learned_embeds.safetensors

    python musipainter_test.py --architecture Musipainter-CA \\
        --data_dir ./Museart/ --embeddings_dir ./audio_embeddings/ \\
        --learned_embeds ./output/learned_embeds.safetensors
"""

import argparse
import sys

ARCH_CHOICES = ("Musipainter-EF", "Musipainter-CA")


def _split_architecture_arg(argv):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--architecture", "-A",
        type=str, choices=ARCH_CHOICES, required=True,
        help="Which Musipainter branch to use in inference: Musipainter-EF "
             "(Early Fusion) or Musipainter-CA (Audio-Guided Cross-Attention).",
    )
    known, remaining = pre.parse_known_args(argv)
    return known.architecture, remaining


def main():
    architecture, remaining_argv = _split_architecture_arg(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining_argv

    if architecture == "Musipainter-EF":
        print("=" * 60)
        print("  Musipainter — routing to EARLY FUSION (Musipainter-EF)")
        print("=" * 60)
        import test_ef as delegate
    else:
        print("=" * 60)
        print("  Musipainter — routing to CROSS-ATTENTION (Musipainter-CA)")
        print("=" * 60)
        import test_ca as delegate

    args = delegate.parse_args()
    delegate.inference(args)


if __name__ == "__main__":
    main()
