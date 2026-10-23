#!/usr/bin/env python
# @title musipainter_train.py
"""
musipainter_train.py — UNIQUE training entry point for Musipainter.

Allows choosing which of the two multimodal fusion branches to use,
without modifying the logic of existing files:

    --architecture Musipainter-EF   -> Early Fusion + shared self-attention
                                       (train_ef.py + MusicToken_no_accel_ef.py)
    --architecture Musipainter-CA   -> Audio-Guided Cross-Attention
                                       (train_ca.py + MusicToken_no_accel_ca.py)

All other arguments (--data_dir, --embeddings_dir, --train_batch_size,
--ef_d_model, etc.) are identical to those of the two original scripts and are
forwarded unchanged to the selected branch's script.

Examples:
    python musipainter_train.py --architecture Musipainter-EF \\
        --data_dir ./Museart/ --embeddings_dir ./audio_embeddings/ \\
        --train_batch_size 2 --max_train_steps 20000

    python musipainter_train.py --architecture Musipainter-CA \\
        --data_dir ./Museart/ --embeddings_dir ./audio_embeddings/ \\
        --train_batch_size 8 --max_train_steps 20000

Note: this script does NOT modify train_ef.py / train_ca.py / the two
MusicTokenWrapper. It simply routes execution to the branch
requested, exactly as the user would do by manually launching
`python train_ef.py ...` or `python train_ca.py ...`.
"""

import argparse
import sys

ARCH_CHOICES = ("Musipainter-EF", "Musipainter-CA")


def _split_architecture_arg(argv):
    """
    Extract --architecture / -A from argv (supports both "--architecture X" and
    "--architecture=X"), returning (architecture, remaining_argv).
    Unrecognized arguments are left intact to be
    interpreted by the delegate script's parser.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--architecture", "-A",
        type=str, choices=ARCH_CHOICES, required=True,
        help="Which Musipainter branch to train: Musipainter-EF (Early Fusion) "
             "or Musipainter-CA (Audio-Guided Cross-Attention).",
    )
    known, remaining = pre.parse_known_args(argv)
    return known.architecture, remaining


def main():
    architecture, remaining_argv = _split_architecture_arg(sys.argv[1:])

    # The delegate scripts read argparse directly from sys.argv:
    # reconstruct sys.argv without --architecture before importing them.
    sys.argv = [sys.argv[0]] + remaining_argv

    if architecture == "Musipainter-EF":
        print("=" * 60)
        print("  Musipainter — routing to EARLY FUSION (Musipainter-EF)")
        print("=" * 60)
        import train_ef as delegate
    else:
        print("=" * 60)
        print("  Musipainter — routing to CROSS-ATTENTION (Musipainter-CA)")
        print("=" * 60)
        import train_ca as delegate

    delegate.train_validation()


if __name__ == "__main__":
    main()
