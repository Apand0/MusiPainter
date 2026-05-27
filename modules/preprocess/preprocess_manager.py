# @title preprocess_manager.py
"""
Orchestrates incremental BEATs audio embedding preprocessing.

Handles the ~20 GB /kaggle/working quota by processing in class batches,
monitoring disk space, and printing resume instructions when space runs low.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import shutil
import sys
import time
from pathlib import Path

from modules.BEATs.BEATs import BEATs, BEATsConfig
from modules.preprocess.utils import save_safetensors, open_safetensors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _free_gb(path: str = "./") -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


def _dir_size_gb(path: str) -> float:
    total = 0
    for dirpath, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total / (1024 ** 3)


def _print_banner(title: str) -> None:
    logger.info(f"\n{'='*60}\n  {title}\n{'='*60}")


def _collect_classes(audio_dir: str) -> list[str]:
    """
    Return sorted unique class names found in *audio_dir*.
    Handles flat and set-subfolder layouts.
    """
    classes: set[str] = set()
    audio_path = Path(audio_dir)
    for p in audio_path.rglob("*.wav"):
        classes.add(p.parent.name)
    return sorted(classes)


def _print_upload_instructions(
    output_dir: str,
    stride:     int,
    part_num:   int,
    existing_datasets: list[str],
) -> None:
    """Print Dataset creation instructions for resuming."""
    size_gb = _dir_size_gb(output_dir)
    free_gb = _free_gb(output_dir)
    ds_name = f"musipainter-audio-emb-stride{stride}-part{part_num}"

    logger.info(
        f"\n{'='*60}\n"
        f"  DISK SPACE LOW — PARTIAL RUN SAVED\n"
        f"{'='*60}\n"
        f"\n"
        f"  Output dir size : {size_gb:.1f} GB\n"
        f"  Free disk space : {free_gb:.1f} GB\n"
        f"\n"
        f"  NEXT STEPS:\n"
        f"\n"
        f"  1. Kaggle sidebar → Data → + Add data → New Dataset\n"
        f"  2. Upload the ENTIRE folder:\n"
        f"       {output_dir}\n"
        f"     (include chunks/ and metadata.json)\n"
        f"  3. Name the dataset:  {ds_name}\n"
        f"  4. Publish the dataset.\n"
        f"  5. Add that dataset as INPUT to this notebook:\n"
        f"       /kaggle/input/{ds_name}\n"
        f"  6. Re-run with the extra flag:\n"
        f"\n"
    )

    all_ds = existing_datasets + [f"/kaggle/input/{ds_name}"]
    ds_arg = ",".join(all_ds)
    logger.info(
        f"       python kaggle_preprocess_manager.py \\n"
        f"           --audio_dir {audio_dir_global} \\n"
        f"           --output_base {output_base_global} \\n"
        f"           --temporal_pool_stride {stride} \\n"
        f"           --existing_datasets {ds_arg}\n"
        f"\n"
        f"  The script will skip all IDs already in those datasets.\n"
        f"{'='*60}\n"
    )


audio_dir_global    = ""
output_base_global  = ""


# ─────────────────────────────────────────────────────────────────────────────
#  SINGLE STRIDE RUN
# ─────────────────────────────────────────────────────────────────────────────

def run_stride(
    audio_dir:            str,
    output_dir:           str,
    stride:               int,
    existing_datasets:    list[str],
    beats_checkpoint:     str,
    device:               str,
    batch_size:           int,
    io_workers:           int,
    duration_seconds:     int,
    sample_rate:          int,
    skip_merge:           bool,
    critical_free_gb:     float,
    subfolders:           list[str] | None,
    class_batch_size:     int,
) -> bool:
    """
    Run preprocessing for one temporal_pool_stride value.

    Returns True if completed without hitting disk limit, False if stopped early.
    """
    _print_banner(f"stride={stride}  →  {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    try:
        from modules.preprocess.preprocess_audio_embeddings_colab import (
            preprocess_audio_dataset,
        )
    except ImportError as exc:
        logger.error(
            f"Cannot import preprocess_audio_embeddings_colab: {exc}\n"
            "Run from repo root with modules/ on Python path."
        )
        sys.exit(1)

    if subfolders:
        classes_to_run = subfolders
    else:
        classes_to_run = _collect_classes(audio_dir)

    if not classes_to_run:
        logger.warning(f"No audio classes found in {audio_dir}. Skipping stride {stride}.")
        return True

    logger.info(
        f"Classes to process: {len(classes_to_run)}  "
        f"(class_batch_size={class_batch_size})"
    )
    logger.info(f"Existing datasets for resume: {existing_datasets or 'none'}")

    for batch_start in range(0, len(classes_to_run), class_batch_size):
        class_batch = classes_to_run[batch_start: batch_start + class_batch_size]
        logger.info(
            f"\n[CLASS BATCH] {batch_start // class_batch_size + 1} / "
            f"{-(-len(classes_to_run) // class_batch_size)}  "
            f"→  {class_batch}"
        )

        free = _free_gb(output_dir)
        if free < critical_free_gb:
            logger.warning(f"Only {free:.1f} GB free before class batch — stopping early.")
            return False

        preprocess_audio_dataset(
            audio_dir            = audio_dir,
            output_dir           = output_dir,
            existing_datasets    = ",".join(existing_datasets),
            beats_checkpoint     = beats_checkpoint,
            device               = device,
            sample_rate          = sample_rate,
            duration_seconds     = duration_seconds,
            batch_size           = batch_size,
            temporal_pool_stride = stride,
            io_workers           = io_workers,
            skip_merge           = skip_merge,
            critical_free_gb     = critical_free_gb,
            subfolders           = class_batch,
        )
        gc.collect()

        free_after = _free_gb(output_dir)
        logger.info(f"[DISK] After batch: {free_after:.1f} GB free.")
        if free_after < critical_free_gb:
            logger.warning("Disk critical after class batch — stopping stride run.")
            return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    global audio_dir_global, output_base_global

    parser = argparse.ArgumentParser(
        description="BEATs audio preprocessing manager.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--audio_dir",  type=str, required=True,
        help="Root directory containing .wav files (recursive scan).",
    )
    parser.add_argument(
        "--output_base", type=str, default="./audio_emb",
        help="Base output directory. Each stride gets its own subdirectory.",
    )
    parser.add_argument(
        "--beats_checkpoint", type=str,
        default="models/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
    )
    parser.add_argument(
        "--existing_datasets", type=str, default="",
        help="Comma-separated paths to read-only Kaggle Dataset directories "
             "with prior chunks. Audio IDs found there are skipped.",
    )

    parser.add_argument("--strides",              type=int, nargs="+", default=None,
                        help="Run multiple temporal_pool_stride values. Overrides --temporal_pool_stride.")
    parser.add_argument("--temporal_pool_stride", type=int, default=8)
    parser.add_argument("--duration_seconds",     type=int, default=30)
    parser.add_argument("--batch_size",           type=int, default=8)
    parser.add_argument("--io_workers",           type=int, default=4)
    parser.add_argument("--device",               type=str, default="cuda:0")
    parser.add_argument("--sample_rate",          type=int, default=16000)

    parser.add_argument(
        "--critical_free_gb", type=float, default=2.0,
        help="Stop preprocessing when free disk falls below this (GB).",
    )
    parser.add_argument(
        "--skip_merge", action="store_true",
        help="Do not merge class chunks into a single safetensors file. Saves disk.",
    )
    parser.add_argument(
        "--class_batch_size", type=int, default=5,
        help="Audio classes per disk-check batch. Lower = finer control.",
    )
    parser.add_argument(
        "--subfolders", type=str, nargs="+", default=None,
        help="Only process audio under these subfolder names (class filter).",
    )

    args = parser.parse_args()
    audio_dir_global   = args.audio_dir
    output_base_global = args.output_base

    strides = args.strides if args.strides else [args.temporal_pool_stride]
    existing_datasets = (
        [d.strip() for d in args.existing_datasets.split(",") if d.strip()]
        if args.existing_datasets
        else []
    )

    t_start = time.time()
    _print_banner("Preprocess Manager — Starting")
    logger.info(f"audio_dir          : {args.audio_dir}")
    logger.info(f"output_base        : {args.output_base}")
    logger.info(f"strides            : {strides}")
    logger.info(f"batch_size         : {args.batch_size}")
    logger.info(f"duration_seconds   : {args.duration_seconds}")
    logger.info(f"critical_free_gb   : {args.critical_free_gb}")
    logger.info(f"skip_merge         : {args.skip_merge}")
    logger.info(f"class_batch_size   : {args.class_batch_size}")
    logger.info(f"subfolders filter  : {args.subfolders or 'all'}")
    logger.info(f"existing_datasets  : {existing_datasets or 'none'}")
    logger.info(f"Initial free disk  : {_free_gb(args.output_base if os.path.exists(args.output_base) else '/'):.1f} GB")

    for stride_idx, stride in enumerate(strides, 1):
        output_dir = os.path.join(args.output_base, f"stride{stride}")
        logger.info(f"\n[STRIDE {stride_idx}/{len(strides)}] stride={stride} → {output_dir}")

        completed = run_stride(
            audio_dir         = args.audio_dir,
            output_dir        = output_dir,
            stride            = stride,
            existing_datasets = existing_datasets,
            beats_checkpoint  = args.beats_checkpoint,
            device            = args.device,
            batch_size        = args.batch_size,
            io_workers        = args.io_workers,
            duration_seconds  = args.duration_seconds,
            sample_rate       = args.sample_rate,
            skip_merge        = args.skip_merge,
            critical_free_gb  = args.critical_free_gb,
            subfolders        = args.subfolders,
            class_batch_size  = args.class_batch_size,
        )

        if not completed:
            part_num = len(existing_datasets) + 1
            _print_upload_instructions(
                output_dir        = output_dir,
                stride            = stride,
                part_num          = part_num,
                existing_datasets = existing_datasets,
            )
            logger.info("[MANAGER] Stopping — disk limit hit. See instructions above.")
            sys.exit(0)

        logger.info(f"[STRIDE {stride}] COMPLETE — moving to next stride (if any).")
        gc.collect()

    elapsed = time.time() - t_start
    h, rem  = divmod(int(elapsed), 3600)
    m, s    = divmod(rem, 60)
    _print_banner(f"ALL STRIDES COMPLETE  ({h}h {m:02d}m {s:02d}s)")
    logger.info(
        "Pass output directories to training script:\n"
        + "\n".join(
            f"  --embeddings_dir {os.path.join(args.output_base, f'stride{st}')}"
            for st in strides
        )
    )


if __name__ == "__main__":
    main()
