#!/usr/bin/env python
# @title musipainter_test.py
"""
musipainter_test.py — Entry point UNICO di inferenza/test per Musipainter.

Stessa logica di musipainter_train.py, ma per la generazione/inferenza:

    --architecture Musipainter-EF   → Early Fusion (test_ef.py + MusicToken_no_accel_ef.py)
    --architecture Musipainter-CA   → Audio-Guided Cross-Attention
                                       (test_ca.py + MusicToken_no_accel_ca.py)

Tutti gli altri argomenti (--learned_embeds, --data_dir, --embeddings_dir,
--prompt, --guidance_scale, ecc.) sono identici a quelli dei due script
originali e vengono inoltrati invariati allo script del branch scelto.

Esempi:
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
        help="Quale branch Musipainter usare in inferenza: Musipainter-EF "
             "(Early Fusion) oppure Musipainter-CA (Audio-Guided Cross-Attention).",
    )
    known, remaining = pre.parse_known_args(argv)
    return known.architecture, remaining


def main():
    architecture, remaining_argv = _split_architecture_arg(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining_argv

    if architecture == "Musipainter-EF":
        print("=" * 60)
        print("  Musipainter — routing verso EARLY FUSION (Musipainter-EF)")
        print("=" * 60)
        import test_ef as delegate
    else:
        print("=" * 60)
        print("  Musipainter — routing verso CROSS-ATTENTION (Musipainter-CA)")
        print("=" * 60)
        import test_ca as delegate

    args = delegate.parse_args()
    delegate.inference(args)


if __name__ == "__main__":
    main()
