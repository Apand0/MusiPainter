# @title modules/preprocess/preprocess_audio_embeddings_colab.py
"""
Kaggle-aware BEATs audio embedding pipeline.

Encodes .wav files to BEATs embeddings and writes per-class safetensors chunks.
Supports safe resume from multiple Kaggle Datasets and optional streaming merge.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import shutil
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from tqdm import tqdm

from modules.BEATs.BEATs import BEATs, BEATsConfig
from modules.preprocess.utils import save_safetensors, open_safetensors

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

BEATS_SAMPLE_RATE     = 16000
FLUSH_EVERY_N_BATCHES = 150
CRITICAL_FREE_GB      = 2.0


# ─────────────────────────────────────────────────────────────────────────────
#  DISK SPACE
# ─────────────────────────────────────────────────────────────────────────────

def _free_gb(path: str = "./") -> float:
    """Free disk space in GB at *path*."""
    return shutil.disk_usage(path).free / (1024 ** 3)


def _check_disk_critical(output_dir: str, critical_gb: float = CRITICAL_FREE_GB) -> bool:
    """Return True when free space < *critical_gb* and log recovery instructions."""
    free = _free_gb(output_dir)
    if free < critical_gb:
        logger.error(
            f"\n{'='*60}\n"
            f"[DISK CRITICAL] Only {free:.2f} GB free in {output_dir}.\n"
            f"Stopping preprocessing to protect your data.\n\n"
            f"Next steps:\n"
            f"  1. Kaggle sidebar → 'Data' → '+ Add data' → 'New Dataset'.\n"
            f"  2. Upload the entire output_dir (all chunk_*.safetensors).\n"
            f"  3. Name it e.g. 'musipainter-audio-emb-part1' and publish.\n"
            f"  4. Add that dataset as input to your notebook.\n"
            f"  5. Re-run with:\n"
            f"       --existing_datasets /kaggle/input/musipainter-audio-emb-part1\n"
            f"{'='*60}\n"
        )
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  CLASS EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _class_tag_from_path(audio_path: str, audio_dir: str) -> str:
    """
    Filesystem-safe class tag from a path relative to *audio_dir*.

    Museart layout: <audio_dir>/<set>/<class>/<file>.wav → Train_Hard_Rock.
    Falls back to the immediate parent folder when the structure differs.
    """
    p = Path(audio_path)
    try:
        parts = p.relative_to(audio_dir).parts
        if len(parts) >= 3:
            return f"{parts[0].capitalize()}_{parts[1].replace(' ', '_').replace('/', '-')}"
        elif len(parts) == 2:
            return parts[0].replace(" ", "_").replace("/", "-")
        return "Unknown"
    except ValueError:
        return p.parent.name.replace(" ", "_").replace("/", "-")


# ─────────────────────────────────────────────────────────────────────────────
#  BEATS MODEL
# ─────────────────────────────────────────────────────────────────────────────

def load_beats_model(checkpoint_path: str, device: str = "cuda:0"):
    """Load BEATs checkpoint and wrap with DataParallel when multiple GPUs are available."""
    try:
        from modules.BEATs.BEATs import BEATs, BEATsConfig
    except ImportError:
        logger.error("modules/BEATs/ not in path — add it first.")
        raise

    logger.info(f"Loading BEATs from {checkpoint_path}")
    checkpoint  = torch.load(checkpoint_path, map_location=device)
    cfg         = BEATsConfig(checkpoint['cfg'])
    aud_encoder = BEATs(cfg).to(device)
    aud_encoder.load_state_dict(checkpoint['model'])
    aud_encoder.predictor = None
    aud_encoder.eval()
    aud_encoder.requires_grad_(False)

    if torch.cuda.device_count() > 1:
        logger.info(f"Found {torch.cuda.device_count()} GPUs — enabling DataParallel")

        class _BEATsDP(torch.nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base
            def forward(self, x):
                return self.base.extract_features(x)[1]

        aud_encoder = torch.nn.DataParallel(_BEATsDP(aud_encoder)).to(device)
        logger.info("BEATs wrapped in DataParallel.")
    else:
        logger.info(f"BEATs on single GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    return aud_encoder


# ─────────────────────────────────────────────────────────────────────────────
#  AUDIO I/O
# ─────────────────────────────────────────────────────────────────────────────

def _load_one_audio(
    audio_path: str,
    sample_rate: int,
    duration_seconds: int,
) -> tuple[Optional[torch.Tensor], str]:
    """Load, resample and pad/trim one audio file. Returns (wav[1,T], stem)."""
    try:
        wav, sr = torchaudio.load(audio_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        target_len = sample_rate * duration_seconds
        if wav.shape[1] < target_len:
            wav = wav.repeat(1, math.ceil(target_len / wav.shape[1]))
        wav = wav[:, :target_len]
        return wav, Path(audio_path).stem
    except Exception as exc:
        logger.warning(f"  Error loading {audio_path}: {exc}")
        return None, Path(audio_path).stem


# ─────────────────────────────────────────────────────────────────────────────
#  TEMPORAL POOLING
# ─────────────────────────────────────────────────────────────────────────────

def temporal_pool(features: torch.Tensor, stride: int) -> torch.Tensor:
    """
    Average-pool temporal frames by *stride*: [T, D] → [T//stride, D].

    Non-overlapping windows of *stride* consecutive frames are averaged.
    stride=1 is a no-op; stride=8 reduces 30 s of audio from ~375 to ~46 frames.
    """
    if stride <= 1:
        return features
    T, feat_dim = features.shape
    T_new = T // stride
    return features[:T_new * stride].view(T_new, stride, feat_dim).mean(dim=1)


# ─────────────────────────────────────────────────────────────────────────────
#  BATCH ENCODING
# ─────────────────────────────────────────────────────────────────────────────

def extract_features_batch(
    aud_encoder,
    audio_paths: list[str],
    device: str,
    sample_rate: int = BEATS_SAMPLE_RATE,
    duration_seconds: int = 30,
    temporal_pool_stride: int = 1,
    io_workers: int = 4,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """
    Parallel audio I/O → single GPU BEATs batch → per-sample temporal pooling.

    Returns ({audio_id: tensor[T_pooled, 2304]} float16 on CPU, [error ids]).
    On OOM the batch is halved and retried automatically.
    """
    results: dict[str, torch.Tensor] = {}
    errors:  list[str] = []
    batch_wavs: list[torch.Tensor] = []
    batch_ids:  list[str] = []

    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        futures = {pool.submit(_load_one_audio, p, sample_rate, duration_seconds): p
                   for p in audio_paths}
        for future in as_completed(futures):
            wav, stem = future.result()
            if wav is not None:
                batch_wavs.append(wav)
                batch_ids.append(stem)
            else:
                errors.append(stem)

    if not batch_wavs:
        return results, errors

    pairs      = sorted(zip(batch_ids, batch_wavs), key=lambda x: x[0])
    batch_ids  = [p[0] for p in pairs]
    batch_wavs = [p[1] for p in pairs]

    batch_tensor = torch.cat(batch_wavs, dim=0).to(device)
    del batch_wavs

    encode_ok   = False
    current_bt  = batch_tensor
    current_ids = list(batch_ids)

    while not encode_ok:
        try:
            with torch.no_grad():
                if isinstance(aud_encoder, torch.nn.DataParallel):
                    aud_features = aud_encoder(current_bt)
                else:
                    raw = aud_encoder.extract_features(current_bt)[1]
                    if raw.shape[-1] not in (768, 768 * 3):
                        raise ValueError(f"Unexpected BEATs dim: {raw.shape[-1]}")
                    aud_features = raw
            encode_ok = True
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            half = max(1, current_bt.shape[0] // 2)
            logger.warning(f"  [OOM] batch_size {current_bt.shape[0]} → halved to {half}.")
            if half == current_bt.shape[0]:
                logger.error("  [OOM] Cannot fit even batch_size=1. Skipping batch.")
                del batch_tensor
                return results, errors + current_ids
            current_bt  = current_bt[:half]
            current_ids = current_ids[:half]

    del batch_tensor
    aud_features = aud_features.to(torch.float16).cpu()

    for i, audio_id in enumerate(current_ids):
        feat = aud_features[i]
        if temporal_pool_stride > 1:
            feat = temporal_pool(feat, temporal_pool_stride)
        results[audio_id] = feat.clone()

    del aud_features
    return results, errors


# ─────────────────────────────────────────────────────────────────────────────
#  CHUNK HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _chunks_dir(output_dir: Path) -> Path:
    """Create and return the chunks/ subdirectory."""
    d = output_dir / "chunks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _next_class_chunk_idx(chunks_dir: Path, class_tag: str) -> int:
    """Next available chunk index for *class_tag* (0 when none exist yet)."""
    existing = sorted(chunks_dir.glob(f"{class_tag}_chunk_*.safetensors"))
    return 0 if not existing else int(existing[-1].stem.rsplit("_", 1)[-1]) + 1


def _class_chunk_path(chunks_dir: Path, class_tag: str, idx: int) -> Path:
    return chunks_dir / f"{class_tag}_chunk_{idx:04d}.safetensors"


def _collect_existing_ids(chunks_dir: Path, existing_dataset_dirs: list[Path]) -> set[str]:
    """
    Scan safetensors headers in *chunks_dir* and all *existing_dataset_dirs*.

    Reads only file headers — no tensor data is loaded into RAM.
    Returns the set of audio_ids already encoded in any of those locations.
    """
    done: set[str] = set()

    if chunks_dir.exists():
        for sf_path in sorted(chunks_dir.glob("*.safetensors")):
            try:
                done.update(open_safetensors(sf_path).keys())
            except Exception as exc:
                logger.warning(f"  Unreadable chunk {sf_path.name}: {exc} — skipped.")

    for ds_dir in existing_dataset_dirs:
        if not ds_dir.exists():
            logger.warning(f"  existing_dataset dir not found: {ds_dir} — skipped.")
            continue
        logger.info(f"  Scanning existing dataset: {ds_dir}")
        count_before = len(done)
        for sf_path in ds_dir.rglob("*.safetensors"):
            try:
                done.update(open_safetensors(sf_path).keys())
            except Exception as exc:
                logger.warning(f"    Unreadable {sf_path}: {exc} — skipped.")
        logger.info(f"    → {len(done) - count_before} new IDs found.")

    return done


# ─────────────────────────────────────────────────────────────────────────────
#  PER-CLASS FLUSH
# ─────────────────────────────────────────────────────────────────────────────

def flush_by_class(
    embeddings: dict[str, torch.Tensor],
    audio_id_to_class: dict[str, str],
    chunks_dir: Path,
) -> None:
    """
    Partition *embeddings* by class tag and write one safetensors file per class.

    Output: chunks/<class_tag>_chunk_<NNNN>.safetensors
    e.g.    chunks/Train_Hard_Rock_chunk_0000.safetensors
    """
    class_buffers: dict[str, dict[str, torch.Tensor]] = {}
    for audio_id, tensor in embeddings.items():
        cls = audio_id_to_class.get(audio_id, "Unknown")
        class_buffers.setdefault(cls, {})[audio_id] = tensor

    for cls, cls_tensors in sorted(class_buffers.items()):
        idx  = _next_class_chunk_idx(chunks_dir, cls)
        path = _class_chunk_path(chunks_dir, cls, idx)
        save_safetensors(cls_tensors, path)
        size_mb = path.stat().st_size / (1024 ** 2)
        logger.info(f"  [FLUSH] {path.name}  ({len(cls_tensors)} audio, {size_mb:.1f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
#  MERGE
# ─────────────────────────────────────────────────────────────────────────────

def merge_class_chunks(chunks_dir: Path, output_sf: Path, metadata: dict) -> bool:
    """
    Stream-merge all per-class chunks into one *output_sf*, deleting each source immediately.

    Extra disk required = largest single chunk + 200 MB safety margin.
    Returns False (skip) when free space is insufficient; the dataloader reads chunks natively.
    """
    chunk_files = sorted(chunks_dir.glob("*_chunk_*.safetensors"))
    if not chunk_files:
        logger.info("[MERGE] No class chunks found — nothing to merge.")
        return False

    total_bytes   = sum(p.stat().st_size for p in chunk_files)
    largest_bytes = max(p.stat().st_size for p in chunk_files)
    free_bytes    = shutil.disk_usage(output_sf.parent).free
    min_needed    = largest_bytes + 200 * (1024 ** 2)

    if free_bytes < min_needed:
        logger.warning(
            f"[MERGE SKIPPED] Need ~{largest_bytes/(1024**2):.0f} MB + 200 MB margin "
            f"but only {free_bytes/(1024**2):.0f} MB free. "
            f"Chunks in chunks/ will be read by the dataloader directly."
        )
        return False

    logger.info(
        f"[MERGE] {len(chunk_files)} chunks ({total_bytes/(1024**2):.0f} MB) "
        f"→ {output_sf.name}  (max extra disk: ~{largest_bytes/(1024**2):.0f} MB)"
    )

    merged: dict[str, torch.Tensor] = {}
    for i, chunk_file in enumerate(chunk_files, 1):
        sf = open_safetensors(chunk_file)
        for key in sf.keys():
            merged[key] = sf.get_tensor(key)
        del sf
        chunk_file.unlink()
        logger.info(f"  Read + deleted {chunk_file.name}  ({i}/{len(chunk_files)}, {len(merged)} audio)")
        gc.collect()

    save_safetensors(merged, output_sf, metadata={k: str(v) for k, v in metadata.items()})
    del merged
    gc.collect()

    logger.info(f"[MERGE] Done: {output_sf.name}  ({output_sf.stat().st_size/(1024**2):.1f} MB)")
    try:
        chunks_dir.rmdir()
        logger.info("[MERGE] chunks/ directory removed (empty).")
    except OSError:
        pass

    return True


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_audio_dataset(
    audio_dir:             str,
    output_dir:            str   = "./audio_embeddings/",
    existing_datasets:     str   = "",
    beats_checkpoint:      str   = "models/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
    device:                str   = "cuda:0",
    sample_rate:           int   = BEATS_SAMPLE_RATE,
    duration_seconds:      int   = 30,
    batch_size:            int   = 8,
    temporal_pool_stride:  int   = 8,
    io_workers:            int   = 4,
    skip_merge:            bool  = False,
    critical_free_gb:      float = CRITICAL_FREE_GB,
    subfolders:            Optional[list[str]] = None,
):
    """
    Encode all .wav files under *audio_dir* to BEATs embeddings.

    Writes per-class safetensors chunks to output_dir/chunks/.
    Stops safely when disk falls below *critical_free_gb* and prints resume instructions.
    Audio IDs in *existing_datasets* (comma-separated paths) are skipped.
    Final streaming merge into audio_embeddings.safetensors is attempted unless *skip_merge*.
    """
    t_frames_raw    = (sample_rate * duration_seconds) // 160
    t_frames_pooled = t_frames_raw // temporal_pool_stride if temporal_pool_stride > 1 else t_frames_raw
    feat_dim        = 768 * 3
    mb_per_audio    = t_frames_pooled * feat_dim * 2 / (1024 ** 2)
    max_ram_mb      = FLUSH_EVERY_N_BATCHES * batch_size * mb_per_audio

    _t_start = time.time()
    n_gpus   = torch.cuda.device_count()
    gpu_info = (", ".join(torch.cuda.get_device_name(i) for i in range(n_gpus))
                if n_gpus > 0 else "CPU")

    logger.info("=" * 60)
    logger.info("START: AUDIO PRE-ENCODING (per-class chunks, Kaggle-safe)")
    logger.info(f"timestamp            : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"GPU(s)               : {n_gpus}  ({gpu_info})")
    logger.info(f"beats_checkpoint     : {beats_checkpoint}")
    logger.info(f"audio_dir            : {audio_dir}")
    logger.info(f"output_dir           : {output_dir}")
    logger.info(f"temporal_pool_stride : {temporal_pool_stride}  "
                f"(T: ~{t_frames_raw} → ~{t_frames_pooled})")
    logger.info(f"feature_dim          : {feat_dim} (BEATs layers 4+8+12)")
    logger.info(f"~MB / audio          : {mb_per_audio:.2f}  (float16)")
    logger.info(f"flush every          : {FLUSH_EVERY_N_BATCHES} batches (≤ ~{max_ram_mb:.0f} MB RAM)")
    logger.info(f"skip_merge           : {skip_merge}")
    logger.info(f"critical_free_gb     : {critical_free_gb}")
    logger.info(f"subfolders filter    : {subfolders or 'all'}")
    logger.info(f"existing_datasets    : {existing_datasets or 'none'}")
    logger.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    output_dir_p = Path(output_dir)
    chunks_dir   = _chunks_dir(output_dir_p)
    output_sf    = output_dir_p / "audio_embeddings.safetensors"

    existing_ds_dirs = [Path(ds.strip()) for ds in existing_datasets.split(",") if ds.strip()]
    already_done     = _collect_existing_ids(chunks_dir, existing_ds_dirs)
    if already_done:
        logger.info(f"[Resume] {len(already_done)} audio IDs already encoded — will skip.")

    audio_dir_p = Path(audio_dir)
    all_wav: list[Path] = sorted(
        p for root, _, files in os.walk(audio_dir_p, followlinks=True)
        for f in files if (p := Path(root) / f).suffix.lower() == ".wav"
    )

    if subfolders:
        subfolders_lower = [sf.lower() for sf in subfolders]
        all_wav = [p for p in all_wav if any(sf in str(p).lower() for sf in subfolders_lower)]
        logger.info(f"[FILTER] subfolder filter active → {len(all_wav)} files match.")

    logger.info(f"Found {len(all_wav)} .wav files in {audio_dir_p}")
    to_process = [p for p in all_wav if p.stem not in already_done]
    logger.info(f"To process: {len(to_process)}  (skipped {len(all_wav) - len(to_process)} already done)")

    if not to_process:
        logger.info("[DONE] Nothing to encode. Attempting final merge...")
        if not skip_merge and not output_sf.exists():
            merge_class_chunks(chunks_dir, output_sf,
                               metadata={"total_audios": str(len(already_done)),
                                         "temporal_pool_stride": str(temporal_pool_stride),
                                         "t_frames_pooled": str(t_frames_pooled),
                                         "feature_dim": str(feat_dim)})
        return

    aud_encoder = load_beats_model(beats_checkpoint, device)
    audio_id_to_class: dict[str, str] = {p.stem: _class_tag_from_path(str(p), str(audio_dir_p))
                                          for p in to_process}

    new_embeddings: dict[str, torch.Tensor] = {}
    all_errors:     list[str] = []
    n_new_total = 0
    disk_stop   = False

    batches = list(range(0, len(to_process), batch_size))
    for batch_num, batch_start in enumerate(tqdm(batches, desc="Encoding audio")):
        batch_paths = [str(p) for p in to_process[batch_start: batch_start + batch_size]]
        try:
            batch_results, batch_errors = extract_features_batch(
                aud_encoder, batch_paths, device, sample_rate, duration_seconds,
                temporal_pool_stride=temporal_pool_stride, io_workers=io_workers,
            )
            new_embeddings.update(batch_results)
            all_errors.extend(batch_errors)
            n_new_total += len(batch_results)
        except Exception as exc:
            logger.warning(f"  Error in batch {batch_start}: {exc}")
            all_errors.append(f"batch_{batch_start}: {exc}")

        is_last    = (batch_num == len(batches) - 1)
        do_flush   = ((batch_num + 1) % FLUSH_EVERY_N_BATCHES == 0) or is_last

        if do_flush and new_embeddings:
            flush_by_class(new_embeddings, audio_id_to_class, chunks_dir)
            new_embeddings = {}
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if _check_disk_critical(output_dir, critical_free_gb):
                disk_stop = True
                break

    if new_embeddings:
        flush_by_class(new_embeddings, audio_id_to_class, chunks_dir)
        new_embeddings = {}
        gc.collect()

    logger.info(f"\n[ENCODING DONE] {n_new_total} new audio encoded  ({len(all_errors)} errors)")

    if disk_stop:
        logger.warning(
            "[PARTIAL RUN] Encoding stopped due to low disk space.\n"
            "Upload chunks/ to a Kaggle Dataset, then re-run with --existing_datasets <path>."
        )
        _save_metadata(output_dir_p, chunks_dir, already_done | set(audio_id_to_class.keys()),
                       all_errors, t_frames_raw, t_frames_pooled, feat_dim,
                       sample_rate, duration_seconds, batch_size, temporal_pool_stride,
                       beats_checkpoint, device, merged=False)
        return

    merged = False
    if not skip_merge and not output_sf.exists():
        merged = merge_class_chunks(
            chunks_dir, output_sf,
            metadata={"total_audios": str(n_new_total + len(already_done)),
                      "temporal_pool_stride": str(temporal_pool_stride),
                      "t_frames_raw": str(t_frames_raw),
                      "t_frames_pooled": str(t_frames_pooled),
                      "feature_dim": str(feat_dim),
                      "sample_rate": str(sample_rate),
                      "duration_seconds": str(duration_seconds),
                      "batch_size_used": str(batch_size),
                      "beats_model": Path(beats_checkpoint).stem,
                      "device_used": device,
                      "errors": str(len(all_errors))},
        )
    elif skip_merge:
        logger.info("[SKIP_MERGE] Merge skipped. Class chunks in chunks/ read by the dataloader.")

    _save_metadata(output_dir_p, chunks_dir, already_done, all_errors,
                   t_frames_raw, t_frames_pooled, feat_dim, sample_rate,
                   duration_seconds, batch_size, temporal_pool_stride,
                   beats_checkpoint, device, merged=merged)

    elapsed = time.time() - _t_start
    h, rem  = divmod(int(elapsed), 3600)
    m, s    = divmod(rem, 60)
    logger.info("=" * 60)
    logger.info("COMPLETED: AUDIO PRE-ENCODING")
    logger.info(f"total time           : {f'{h}h {m:02d}m {s:02d}s' if h else f'{m}m {s:02d}s'}")
    logger.info(f"audio encoded        : {n_new_total}")
    logger.info(f"errors               : {len(all_errors)}")
    logger.info(f"throughput           : {n_new_total / max(elapsed, 1):.2f} audio/s")
    logger.info(f"final merge          : {'yes' if merged else 'no (class chunks in chunks/)'}")
    logger.info(f"output_dir           : {output_dir}")
    logger.info("=" * 60)


def _save_metadata(
    output_dir_p: Path, chunks_dir: Path, all_done_ids: set,
    all_errors: list, t_frames_raw: int, t_frames_pooled: int,
    feat_dim: int, sample_rate: int, duration_seconds: int,
    batch_size: int, temporal_pool_stride: int,
    beats_checkpoint: str, device: str, merged: bool,
) -> None:
    """Write metadata.json beside the output safetensors / chunks dir."""
    all_chunk_ids: list[str] = []
    if chunks_dir.exists():
        for sf_path in sorted(chunks_dir.glob("*_chunk_*.safetensors")):
            try:
                all_chunk_ids.extend(open_safetensors(sf_path).keys())
            except Exception:
                pass

    meta = {
        "total_audios": len(all_chunk_ids),
        "audio_ids": all_chunk_ids,
        "output_file": "audio_embeddings.safetensors" if merged else "chunks/",
        "chunks_dir": str(chunks_dir) if not merged else None,
        "embedding_shape": [t_frames_pooled, feat_dim],
        "feature_dim": feat_dim,
        "sample_rate": sample_rate,
        "duration_seconds": duration_seconds,
        "batch_size_used": batch_size,
        "temporal_pool_stride": temporal_pool_stride,
        "t_frames_raw": t_frames_raw,
        "t_frames_pooled": t_frames_pooled,
        "beats_model": Path(beats_checkpoint).stem,
        "device_used": device,
        "errors": all_errors,
    }
    metadata_path = output_dir_p / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(meta, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-encode audio files to BEATs embeddings (Kaggle-safe, per-class chunks)."
    )
    parser.add_argument("--audio_dir",            type=str,   required=True)
    parser.add_argument("--output_dir",           type=str,   default="./output/audio_embeddings/")
    parser.add_argument("--existing_datasets",    type=str,   default="",
                        help="Comma-separated paths to Kaggle Dataset directories with prior chunks.")
    parser.add_argument("--beats_checkpoint",     type=str,
                        default="models/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt")
    parser.add_argument("--device",               type=str,   default="cuda:0")
    parser.add_argument("--sample_rate",          type=int,   default=BEATS_SAMPLE_RATE)
    parser.add_argument("--duration_seconds",     type=int,   default=30)
    parser.add_argument("--batch_size",           type=int,   default=8)
    parser.add_argument("--temporal_pool_stride", type=int,   default=8,
                        help="Average-pool stride (1 = no pooling, 8 = 8× compression).")
    parser.add_argument("--io_workers",           type=int,   default=4)
    parser.add_argument("--skip_merge",           action="store_true",
                        help="Leave per-class chunks in chunks/ without merging. Recommended on Kaggle.")
    parser.add_argument("--critical_free_gb",     type=float, default=CRITICAL_FREE_GB,
                        help="Stop encoding when free disk falls below this (GB).")
    parser.add_argument("--subfolders",           type=str,   nargs="+", default=None,
                        help="Process only audio under these subfolder names.")

    args = parser.parse_args()
    preprocess_audio_dataset(
        audio_dir=args.audio_dir, output_dir=args.output_dir,
        existing_datasets=args.existing_datasets, beats_checkpoint=args.beats_checkpoint,
        device=args.device, sample_rate=args.sample_rate,
        duration_seconds=args.duration_seconds, batch_size=args.batch_size,
        temporal_pool_stride=args.temporal_pool_stride, io_workers=args.io_workers,
        skip_merge=args.skip_merge, critical_free_gb=args.critical_free_gb,
        subfolders=args.subfolders,
    )
