#!/usr/bin/env python
# @title musipainter_train.py
"""
musipainter_train.py — Entry point UNICO di training per Musipainter.

Permette di scegliere quale dei due branch (rami) di fusione multimodale
usare, senza toccare in alcun modo la logica dei file esistenti:

    --architecture Musipainter-EF   → Early Fusion + self-attention condivisa
                                       (train_ef.py + MusicToken_no_accel_ef.py)
    --architecture Musipainter-CA   → Audio-Guided Cross-Attention
                                       (train_ca.py + MusicToken_no_accel_ca.py)

Tutti gli altri argomenti (--data_dir, --embeddings_dir, --train_batch_size,
--ef_d_model, ecc.) sono identici a quelli dei due script originali e vengono
inoltrati invariati allo script del branch scelto.

Esempi:
    python musipainter_train.py --architecture Musipainter-EF \\
        --data_dir ./Museart/ --embeddings_dir ./audio_embeddings/ \\
        --train_batch_size 2 --max_train_steps 20000

    python musipainter_train.py --architecture Musipainter-CA \\
        --data_dir ./Museart/ --embeddings_dir ./audio_embeddings/ \\
        --train_batch_size 8 --max_train_steps 20000

Nota: questo script NON modifica train_ef.py / train_ca.py / i due
MusicTokenWrapper. Si limita a instradare l'esecuzione verso il branch
richiesto, esattamente come farebbe l'utente lanciando manualmente
`python train_ef.py ...` oppure `python train_ca.py ...`.
"""

import argparse
import sys

ARCH_CHOICES = ("Musipainter-EF", "Musipainter-CA")


def _split_architecture_arg(argv):
    """
    Estrae --architecture / -A da argv (supporta sia "--architecture X" che
    "--architecture=X"), restituendo (architecture, argv_rimanente).
    Gli argomenti non riconosciuti vengono lasciati intatti per essere
    interpretati dal parser dello script delegato.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--architecture", "-A",
        type=str, choices=ARCH_CHOICES, required=True,
        help="Quale branch Musipainter addestrare: Musipainter-EF (Early Fusion) "
             "oppure Musipainter-CA (Audio-Guided Cross-Attention).",
    )
    known, remaining = pre.parse_known_args(argv)
    return known.architecture, remaining


def main():
    architecture, remaining_argv = _split_architecture_arg(sys.argv[1:])

    # Gli script delegati leggono argparse direttamente da sys.argv:
    # ricostruiamo sys.argv senza --architecture prima di importarli.
    sys.argv = [sys.argv[0]] + remaining_argv

    if architecture == "Musipainter-EF":
        print("=" * 60)
        print("  Musipainter — routing verso EARLY FUSION (Musipainter-EF)")
        print("=" * 60)
        import train_ef as delegate
    else:
        print("=" * 60)
        print("  Musipainter — routing verso CROSS-ATTENTION (Musipainter-CA)")
        print("=" * 60)
        import train_ca as delegate

    delegate.train_validation()


if __name__ == "__main__":
    main()
