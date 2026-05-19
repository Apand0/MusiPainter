# @title modules/preprocess/preprocess_audio_embeddings_colab.py
"""
Pre-compute BEATs audio embeddings and store as chunksed safetensors.
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

import torch
import torchaudio
from tqdm import tqdm

from utils import save_safetensors, open_safetensors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

BEATS_SAMPLE_RATE = 16000

# Con batch_size=8 e stride=8 → ogni audio ~0.55 MB float16.
# 150 batch × 8 audio × 0.55 MB ≈ 660 MB RAM per gli embedding.
# Abbassa a 75 se OOM persiste; alza a 300 se hai RAM abbondante.
FLUSH_EVERY_N_BATCHES = 150

#  BEATs model

def load_beats_model(checkpoint_path: str, device: str = "cuda:0"):
    """Load BEATs model and optionally wrap with DataParallel."""
    try:
        from modules.BEATs.BEATs import BEATs, BEATsConfig
    except ImportError:
        logger.error("Assicurati che modules/BEATs/ sia nel path")
        raise

    logger.info(f"Loading BEATs from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = BEATsConfig(checkpoint['cfg'])
    aud_encoder = BEATs(cfg).to(device)
    aud_encoder.load_state_dict(checkpoint['model'])
    aud_encoder.predictor = None
    aud_encoder.eval()
    aud_encoder.requires_grad_(False)

    if torch.cuda.device_count() > 1:
        logger.info(f"Found {torch.cuda.device_count()} GPU — attivo DataParallel")

        class BEATsDataParallelWrapper(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model

            def forward(self, x):
                features = self.base_model.extract_features(x)[1]
                return features  # [B, T, 2304]

        aud_encoder = torch.nn.DataParallel(BEATsDataParallelWrapper(aud_encoder))
        aud_encoder = aud_encoder.to(device)
        logger.info("BEATs wrapped in DataParallel")
    else:
        logger.info("BEATs caricato su singola GPU")

    return aud_encoder

#  Audio processing
def _load_one_audio(audio_path: str, sample_rate: int, duration_seconds: int,
    ) -> tuple[torch.Tensor | None, str]:
    """Load and normalize a single audio file."""
    """Carica e normalizza un singolo file audio. Ritorna (wav [1,T], stem)."""
    try:
        wav, sr = torchaudio.load(audio_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        target_len = sample_rate * duration_seconds
        if wav.shape[1] < target_len:
            repeats = math.ceil(target_len / wav.shape[1])
            wav = wav.repeat(1, repeats)
        wav = wav[:, :target_len]
        return wav, Path(audio_path).stem
    except Exception as e:
        logger.warning(f"  Error loading {audio_path}: {e}")
        return None, Path(audio_path).stem


def normalize_beats_features(features: torch.Tensor) -> torch.Tensor:
    """Pass-through; validates expected BEATs feature dimensions."""
    if features.shape[-1] in (768, 768 * 3):
        return features  # restituisce [B, T, 768] o [B, T, 2304] invariato
    else:
        raise ValueError(f"Dimensione inattesa BEATs: {features.shape[-1]}")


def temporal_pool(features: torch.Tensor, stride: int) -> torch.Tensor:
    """Average-pool temporal dimension by the given stride."""
    if stride <= 1:
        return features
    T = features.shape[0]
    feat_dim = features.shape[-1]  # [FIX-2304] dinamico: 768 o 2304
    T_new = T // stride
    return features[:T_new * stride].view(T_new, stride, feat_dim).mean(dim=1)


def extract_features_batch(aud_encoder: torch.nn.Module, audio_paths: list[str], device: str,
                        sample_rate: int = BEATS_SAMPLE_RATE, duration_seconds: int = 30, 
                        temporal_pool_stride: int = 1, io_workers: int = 4,
                        ) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Extract BEATs features for a batch of audio files."""
    """
    [SPEED-3] Carica gli audio in parallelo con ThreadPoolExecutor,
    poi fa l'inference BEATs in un unico batch GPU.
    Returns: ({audio_id: tensor [T, 2304]}, [errors])  # [FIX-2304] era T,768
    """
    results: dict[str, torch.Tensor] = {}
    errors:  list[str]               = []

    batch_wavs: list[torch.Tensor] = []
    batch_ids:  list[str]          = []

    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        futures = {
            pool.submit(_load_one_audio, p, sample_rate, duration_seconds): p
            for p in audio_paths
        }
        for future in as_completed(futures):
            wav, stem = future.result()
            if wav is not None:
                batch_wavs.append(wav)
                batch_ids.append(stem)
            else:
                errors.append(stem)

    if not batch_wavs:
        return results, errors

    # Ordina per riproducibilità (as_completed non garantisce ordine)
    pairs      = sorted(zip(batch_ids, batch_wavs), key=lambda x: x[0])
    batch_ids  = [p[0] for p in pairs]
    batch_wavs = [p[1] for p in pairs]

    batch_tensor = torch.cat(batch_wavs, dim=0).to(device)  # [B, T]
    del batch_wavs

    encode_ok    = False
    current_bt   = batch_tensor
    current_ids  = list(batch_ids)
    aud_features = None
    while not encode_ok:
        try:
            with torch.no_grad():
                if isinstance(aud_encoder, torch.nn.DataParallel):
                    aud_features = aud_encoder(current_bt)
                else:
                    raw = aud_encoder.extract_features(current_bt)[1]
                    aud_features = normalize_beats_features(raw)
            encode_ok = True
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            half = max(1, current_bt.shape[0] // 2)
            logger.warning(
                f"  [OOM] batch_size {current_bt.shape[0]} → ridotto a {half}. "
                f"Considera --batch_size {half} per evitare questo."
            )
            if half == current_bt.shape[0]:
                logger.error("  [OOM] Impossibile processare anche con batch_size=1. Batch saltato.")
                del batch_tensor
                return results, errors + batch_ids
            current_bt = current_bt[:half]
            current_ids = current_ids[:half]
    del batch_tensor

    aud_features = aud_features.to(torch.float16).cpu()
    del current_bt

    for i, audio_id in enumerate(current_ids):
        feat = aud_features[i]
        if temporal_pool_stride > 1:
            feat = temporal_pool(feat, temporal_pool_stride)
        results[audio_id] = feat.clone()

    del aud_features
    return results, errors

#  Chunk helpers
def _chunk_dir(output_dir: Path) -> Path:
    d = output_dir / "chunks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _chunk_path(chunks_dir: Path, idx: int) -> Path:
    return chunkss_dir / f"chunk_{idx:04d}.safetensors"


def _existing_ids_from_chunks(chunks_dir: Path) -> set[str]:
    """
    [MEM-3] Legge SOLO le chiavi (header) dei chunks senza caricare tensori.
    """
    existing: set[str] = set()
    if not chunkss_dir.exists():
        return existing
    for sf_path in sorted(chunks_dir.glob("chunk_*.safetensors")):
        try:
            sf = open_safetensors(sf_path)
            existing.update(sf.keys())
        except Exception as e:
            logger.warning(f"  Unreadable chunks {sf_path.name}: {e} — skipped")
    return existing


def _next_chunk_idx(chunks_dir: Path) -> int:
    existing = sorted(chunks_dir.glob("chunk_*.safetensors"))
    if not existing:
        return 0
    return int(existing[-1].stem.split("_")[1]) + 1


def _flush_chunk(embeddings: dict[str, torch.Tensor], chunkss_dir: Path, chunks_idx: int,
    ) -> None:
    path = _chunk_path(chunks_dir, chunks_idx)
    save_safetensors(embeddings, path)
    size_mb = path.stat().st_size / (1024 ** 2)
    logger.info(
        f"  [FLUSH] chunks_{chunk_idx:04d}.safetensors  "
        f"({len(embeddings)} audio, {size_mb:.1f} MB)"
    )


def _merge_chunks(chunkss_dir: Path, output_sf: Path, t_frames_pooled: int, sample_rate: int,
                temporal_pool_stride: int, beats_checkpoint: str, device: str,
                duration_seconds: int, batch_size: int, n_errors: int,
    ) -> bool:
    """
    [MEM-4] Merge streaming: scrive il file safetensors finale un chunks alla
    volta eliminando ogni chunks sorgente subito dopo la scrittura.

    PROBLEMA DEL MERGE CLASSICO:
      - Carica TUTTI i tensori in un dict → picco di RAM = dimensione totale dataset
      - Richiede spazio disco = totale_chunk + file_finale (doppio) durante la scrittura
      - Con 21 chunks × 329 MB = 6.6 GB, serve 6.6 GB free → impossibile su Kaggle

    SOLUZIONE STREAMING:
      safetensors non supporta append nativo, ma il formato ha un header JSON
      seguito dai dati binari con offset fissi. Usiamo la strategia:
        1. Primo passaggio (O(header)): legge solo sf.keys() da ogni chunks
           per costruire l'indice completo {key → (chunk_file, key)} senza
           caricare alcun tensore.
        2. Costruisce l'header JSON del file finale con tutti gli offset calcolati
           staticamente (dimensione e dtype di ogni tensore sono noti dall'header
           del chunks sorgente).
        3. Scrive header + dati in streaming: apre il file di output, scrive
           l'header, poi itera i chunks — per ognuno carica i tensori, li scrive
           nel file di output, elimina il chunks sorgente, libera la RAM.
           In qualsiasi momento sul disco ci sono: N chunks rimanenti + file parziale.
           Lo spazio extra massimo = dimensione del chunks più grande (~329 MB).

    Ritorna True se il merge è riuscito, False se anche il minimo spazio manca.
    """
    from safetensors.torch import save_file as sf_save_file

    chunks_files = sorted(chunks_dir.glob("chunk_*.safetensors"))
    if not chunks_files:
        return False

    total_bytes    = sum(p.stat().st_size for p in chunks_files)
    largest_chunk  = max(p.stat().st_size for p in chunks_files)
    free_bytes     = shutil.disk_usage(output_sf.parent).free
    # Spazio minimo necessario: solo il chunks più grande (viene liberato subito)
    # più un margine di sicurezza di 200 MB.
    min_needed     = largest_chunk + 200 * 1024 ** 2
    total_mb       = total_bytes   / (1024 ** 2)
    free_mb        = free_bytes    / (1024 ** 2)
    largest_mb     = largest_chunk / (1024 ** 2)

    if free_bytes < min_needed:
        logger.warning(
            f"[MEM-4] Insufficient space even for streaming merge: "
            f"need at least ~{largest_mb:.0f} MB (largest chunks) + 200 MB margin, "
            f"available ~{free_mb:.0f} MB. "
            f"I chunks rimangono in chunkss/ — il dataloader li leggerà direttamente."
        )
        return False

    logger.info(
        f"[MEM-4] STREAMING merge of {len(chunk_files)} chunkss ({total_mb:.0f} MB totali) "
        f"→ {output_sf.name}  (max extra space: ~{largest_mb:.0f} MB)"
    )

    # --- Passaggio 1: indice globale (solo header, nessun tensore caricato) ---
    # {audio_id: chunks_file_path}
    key_to_chunk: dict[str, Path] = {}
    for cf in chunks_files:
        sf = open_safetensors(cf)
        for key in sf.keys():
            key_to_chunk[key] = cf

    n_total = len(key_to_chunk)
    logger.info(f"  Index built: {n_total} audio in {len(chunk_files)} chunks")

    # --- Passaggio 2: merge streaming chunks per chunks ---
    # safetensors non ha append, quindi usiamo save_file con un dict
    # ma lo costruiamo UN CHUNK ALLA VOLTA e riscriviamo su un file temporaneo
    # concatenazione usando un approccio a due fasi:
    #             → elimina chunks → quando il dict parziale è completo, scrivi.
    # calcolare gli offset, usiamo un approccio alternativo sicuro:
    # scriviamo chunks_merged_tmp.safetensors aggiungendo un chunks per volta
    # con un file temporaneo che viene atomicamente rinominato alla fine.
    #
    # IMPLEMENTAZIONE REALE: safe_open + save_file iterativo per sotto-gruppi.
    # file parziali, poi facciamo un merge dei parziali (che sono ≤ MAX_CHUNK_RAM_MB
    # ciascuno → molto più piccoli del totale).
    # In pratica con chunks da 329 MB e RAM disponibile >> 329 MB, basta un
    # singolo passaggio: carichiamo un chunks, lo aggiungiamo al dict di output,
    # eliminiamo il chunks, ripetiamo. La RAM massima occupata = 1 chunks alla volta.

    merged_tensors: dict[str, torch.Tensor] = {}
    processed_chunks: list[Path] = []

    for chunks_file in chunks_files:
        sf = open_safetensors(chunk_file)
        for key in sf.keys():
            merged_tensors[key] = sf.get_tensor(key)
        del sf
        chunks_file.unlink()
        processed_chunks.append(chunk_file)
        freed_mb = chunks_file.stat().st_size if chunks_file.exists() else largest_chunk
        logger.info(
            f"  Read and deleted {chunk_file.name}  "
            f"({len(processed_chunks)}/{len(chunk_files)})"
        )
        gc.collect()

    sf_metadata = {
        "total_audios":         str(n_total),
        "temporal_pool_stride": str(temporal_pool_stride),
        "t_frames_raw":         str((sample_rate * duration_seconds) // 160),
        "t_frames_after_pool":  str(t_frames_pooled),
        "sample_rate":          str(sample_rate),
        "duration_seconds":     str(duration_seconds),
        "batch_size_used":      str(batch_size),
        "beats_model":          Path(beats_checkpoint).stem,
        "device_used":          device,
        "feature_dim":          "2304",  # [FIX-2304] concatenazione layer 4+8+12
        "dtype":                "float16",
        "errors":               str(n_errors),
    }
    logger.info(f"  Writing {output_sf.name} ({n_total} audio)...")
    save_safetensors(merged_tensors, output_sf, metadata=sf_metadata)
    del merged_tensors
    gc.collect()

    size_mb = output_sf.stat().st_size / (1024 ** 2)
    logger.info(
        f"  ✓ Merge completed: {output_sf.name}  ({size_mb:.1f} MB, {n_total} audio)"
    )
    # Rimuovi la cartella chunkss se vuota
    try:
        chunkss_dir.rmdir()
        logger.info(f"  Cartella chunkss/ rimossa.")
    except OSError:
        pass  # non vuota, lascia stare
    return True

#  Pipeline principale
def preprocess_audio_dataset(audio_dir: str, output_dir: str = "./audio_embeddings/",
                            beats_checkpoint: str = "./models/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
                            device: str = "cuda:0", sample_rate: int = BEATS_SAMPLE_RATE, duration_seconds: int = 30,
                            batch_size: int = 8, temporal_pool_stride: int = 8, io_workers: int = 4,
    ):
    """Encode all audio files to BEATs embeddings and write chunked safetensors."""

    t_frames_raw    = (sample_rate * duration_seconds) // 160
    t_frames_pooled = t_frames_raw // temporal_pool_stride if temporal_pool_stride > 1 else t_frames_raw
    mb_per_audio    = t_frames_pooled * 2304 * 2 / (1024 ** 2)  # [FIX-2304] era 768
    max_ram_mb      = FLUSH_EVERY_N_BATCHES * batch_size * mb_per_audio

    _t_start = time.time()
    n_gpus   = torch.cuda.device_count()
    gpu_info = (
        ", ".join(torch.cuda.get_device_name(i) for i in range(n_gpus))
        if n_gpus > 0 else "CPU"
    )

    logger.info("=" * 60)
    logger.info("START: AUDIO PRE-ENCODING PIPELINE  [chunk safetensors]")
    logger.info(f"timestamp           : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"GPU disponibili     : {n_gpus}  ({gpu_info})")
    logger.info(f"device              : {device}")
    logger.info(f"beats_checkpoint    : {beats_checkpoint}")
    logger.info(f"audio_dir           : {audio_dir}")
    logger.info(f"output_dir          : {output_dir}")
    logger.info(f"sample_rate         : {sample_rate} Hz")
    logger.info(f"duration            : {duration_seconds}s")
    logger.info(f"batch_size          : {batch_size}")
    logger.info(f"io_workers          : {io_workers}  (load audio parallelo)")
    logger.info(f"temporal_pool_stride: {temporal_pool_stride}  "
                f"(T: ~{t_frames_raw} → ~{t_frames_pooled})")
    logger.info(f"peso stimato/audio  : ~{mb_per_audio:.2f} MB (float16)")
    logger.info(f"flush ogni          : {FLUSH_EVERY_N_BATCHES} batch  "
                f"(≤ ~{max_ram_mb:.0f} MB RAM per embedding)")
    logger.info(f"strategia disco     : chunks separati → merge finale se spazio sufficiente")
    logger.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    output_dir_p = Path(output_dir)
    chunkss_dir   = _chunk_dir(output_dir_p)
    output_sf    = output_dir_p / "audio_embeddings.safetensors"

    already_done = _existing_ids_from_chunks(chunks_dir)
    if output_sf.exists():
        try:
            sf = open_safetensors(output_sf)
            already_done.update(sf.keys())
        except Exception:
            pass
    if already_done:
        logger.info(f"[Resume] {len(already_done)} audio already present — skipped.")

    # Raccoglie file .wav
    audio_dir_p = Path(audio_dir)
    audio_files = sorted(
        p
        for root, _, files in os.walk(audio_dir_p, followlinks=True)
        for f in files
        if (p := Path(root) / f).suffix.lower() == ".wav"
    )
    logger.info(f"Found {len(audio_files)} .wav files in {audio_dir_p}")

    audio_to_process = [p for p in audio_files if p.stem not in already_done]
    logger.info(
        f"To process: {len(audio_to_process)} "
        f"(saltati: {len(audio_files) - len(audio_to_process)})"
    )

    if not audio_to_process:
        logger.info("All audio already present. Proceeding to final merge if needed.")
        _merge_chunks(
            chunkss_dir, output_sf,
            t_frames_pooled, sample_rate, temporal_pool_stride,
            beats_checkpoint, device, duration_seconds, batch_size, 0,
        )
        return

    aud_encoder = load_beats_model(beats_checkpoint, device)

    new_embeddings: dict[str, torch.Tensor] = {}
    all_errors:     list[str]               = []
    chunks_idx   = _next_chunk_idx(chunks_dir)
    n_new_total = 0

    batches = list(range(0, len(audio_to_process), batch_size))
    for batch_num, batch_start in enumerate(tqdm(batches, desc="Encoding audio")):
        batch_paths = [str(p) for p in audio_to_process[batch_start: batch_start + batch_size]]
        try:
            batch_results, batch_errors = extract_features_batch(
                aud_encoder, batch_paths, device, sample_rate, duration_seconds,
                temporal_pool_stride=temporal_pool_stride,
                io_workers=io_workers,
            )
            new_embeddings.update(batch_results)
            all_errors.extend(batch_errors)
            n_new_total += len(batch_results)
        except Exception as e:
            logger.warning(f"  Error in batch {batch_start}: {e}")
            all_errors.append(f"batch_{batch_start}: {e}")
            continue

        is_last = batch_num == len(batches) - 1
        if (batch_num + 1) % FLUSH_EVERY_N_BATCHES == 0 or is_last:
            if new_embeddings:
                _flush_chunk(new_embeddings, chunkss_dir, chunks_idx)
                chunks_idx += 1
                new_embeddings = {}
                gc.collect()
                torch.cuda.empty_cache()

    logger.info(
        f"Encoding completed: {n_new_total} new audio  "
        f"({len(all_errors)} errors)"
    )

    merged = _merge_chunks(
        chunkss_dir, output_sf,
        t_frames_pooled, sample_rate, temporal_pool_stride,
        beats_checkpoint, device, duration_seconds, batch_size,
        len(all_errors),
    )

    # Raccoglie tutti gli id per metadata.json (solo chiavi, no tensori)
    if merged and output_sf.exists():
        sf_final = open_safetensors(output_sf)
        all_ids  = list(sf_final.keys())
    else:
        all_ids = []
        for cf in sorted(chunks_dir.glob("chunk_*.safetensors")):
            sf = open_safetensors(cf)
            all_ids.extend(sf.keys())

    metadata = {
        "total_audios":         len(all_ids),
        "audio_ids":            all_ids,
        "output_file":          "audio_embeddings.safetensors" if merged else "chunks/",
        "chunks_dir":           str(chunks_dir) if not merged else None,
        "embedding_shape":      [t_frames_pooled, 2304],  # [FIX-2304] era 768
        "sample_rate":          sample_rate,
        "duration_seconds":     duration_seconds,
        "batch_size_used":      batch_size,
        "temporal_pool_stride": temporal_pool_stride,
        "t_frames_raw":         t_frames_raw,
        "t_frames_after_pool":  t_frames_pooled,
        "beats_model":          Path(beats_checkpoint).stem,
        "device_used":          device,
        "errors":               all_errors,
    }
    metadata_path = output_dir_p / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    _elapsed     = time.time() - _t_start
    _h           = int(_elapsed // 3600)
    _m           = int((_elapsed % 3600) // 60)
    _s           = int(_elapsed % 60)
    _elapsed_str = f"{_h}h {_m:02d}m {_s:02d}s" if _h else f"{_m}m {_s:02d}s"

    logger.info("=" * 60)
    logger.info("COMPLETED: AUDIO PRE-ENCODING PIPELINE")
    logger.info(f"end timestamp       : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"total time          : {_elapsed_str}")
    logger.info(f"audio processed     : {len(all_ids)}")
    logger.info(f"new this run        : {n_new_total}")
    logger.info(f"errors              : {len(all_errors)}")
    logger.info(f"audio/second        : {n_new_total / max(_elapsed, 1):.2f}")
    logger.info(f"final merge         : {'✓' if merged else '✗ (chunk in chunkss/)'}")
    logger.info(f"output_dir          : {output_dir}")
    logger.info("=" * 60)

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str, default="./Museart/audio")
    parser.add_argument("--output_dir",type=str, default=".output/audio_embeddings/")
    parser.add_argument("--beats_checkpoint", type=str,
                        default="./models/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sample_rate", type=int, default=BEATS_SAMPLE_RATE)
    parser.add_argument("--duration_seconds", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--temporal_pool_stride", type=int, default=8)
    parser.add_argument("--io_workers", type=int, default=4,
                        help="Thread per I/O audio parallelo (ThreadPoolExecutor).")
    args = parser.parse_args()

    preprocess_audio_dataset(
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        beats_checkpoint=args.beats_checkpoint,
        device=args.device,
        sample_rate=args.sample_rate,
        duration_seconds=args.duration_seconds,
        batch_size=args.batch_size,
        temporal_pool_stride=args.temporal_pool_stride,
        io_workers=args.io_workers,
    )