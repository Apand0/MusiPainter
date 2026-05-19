# @title modules/preprocess/preprocess_image_latents_colab.py
"""
Pre-compute VAE image latents and store as chunksed safetensors.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from diffusers import AutoencoderKL
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import save_safetensors, open_safetensors

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    force=True,
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Con batch_size=32 e risoluzione 512 → ogni latente ~25 KB float16.
# Alza a 300 se hai spazio; abbassa a 50 se OOM.
FLUSH_EVERY_N_BATCHES = 100

#  Dataset / DataLoader
class ImagePreprocessDataset(Dataset):
    """Load and preprocess images for VAE encoding."""
    def __init__(self, image_paths: list, resolution: int, center_crop: bool):
        self.image_paths = image_paths
        self.resolution  = resolution
        self.center_crop = center_crop

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            arr = np.array(img, dtype=np.uint8)

            if self.center_crop:
                h, w = arr.shape[0], arr.shape[1]
                crop = min(h, w)
                arr  = arr[(h - crop) // 2:(h + crop) // 2,
                           (w - crop) // 2:(w + crop) // 2]

            img    = Image.fromarray(arr).resize(
                (self.resolution, self.resolution), Image.BICUBIC
            )
            arr    = np.array(img, dtype=np.float32)
            arr    = arr / 127.5 - 1.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
            return tensor, Path(img_path).stem, True
        except Exception as e:
            logger.warning(f"  Error loading {img_path}: {e}")
            dummy = torch.zeros(3, self.resolution, self.resolution)
            return dummy, Path(img_path).stem, False


def collate_skip_errors(batch):
    tensors, ids, oks = zip(*batch)
    valid = [i for i, ok in enumerate(oks) if ok]
    if not valid:
        return None, None
    stacked   = torch.stack([tensors[i] for i in valid])
    valid_ids = [ids[i] for i in valid]
    return stacked, valid_ids

#  VAE
class VAEDataParallelWrapper(nn.Module):
    """Wrap VAE encode step for DataParallel."""
    def __init__(self, vae: AutoencoderKL):
        super().__init__()
        self.vae = vae

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        return latents * 0.18215


def load_vae(pretrained_model_name_or_path: str, device: str) -> nn.Module:
    """Load and wrap the SD VAE for latent pre-encoding."""
    logger.info(f"Loading VAE from '{pretrained_model_name_or_path}'...")
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float16,
        cache_dir="/tmp/hf_model_cache",
    )
    vae.eval()
    vae.requires_grad_(False)

    n_gpus  = torch.cuda.device_count()
    wrapper = VAEDataParallelWrapper(vae).to(device)

    if n_gpus > 1:
        logger.info(
            f"[OPT-1] {n_gpus} GPU — DataParallel VAE: "
            + ", ".join(torch.cuda.get_device_name(i) for i in range(n_gpus))
        )
        return nn.DataParallel(wrapper)

    logger.info(
        f"[INFO] Single GPU: "
        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )
    return wrapper


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


def _flush_chunk( latents: dict[str, torch.Tensor], chunkss_dir: Path, chunks_idx: int,
    ) -> None:
    path = _chunk_path(chunks_dir, chunks_idx)
    save_safetensors(latents, path)
    size_mb = path.stat().st_size / (1024 ** 2)
    logger.info(
        f"  [FLUSH] chunks_{chunk_idx:04d}.safetensors  "
        f"({len(latents)} immagini, {size_mb:.1f} MB)"
    )


def _merge_chunks( chunkss_dir: Path, output_sf: Path, resolution: int, center_crop: bool,
                pretrained_model_name_or_path: str, device: str, n_errors: int,
    ) -> bool:
    """
    [MEM-4] Merge STREAMING: elimina ogni chunks sorgente subito dopo averlo
    letto, recuperando spazio disco progressivamente.
    Lo spazio extra massimo usato = chunks più grande (invece del totale).
    Ritorna True se il merge è riuscito, False se anche il minimo spazio manca.
    """
    chunks_files = sorted(chunks_dir.glob("chunk_*.safetensors"))
    if not chunks_files:
        return False

    total_bytes   = sum(p.stat().st_size for p in chunks_files)
    largest_chunk = max(p.stat().st_size for p in chunks_files)
    free_bytes    = shutil.disk_usage(output_sf.parent).free
    min_needed    = largest_chunk + 200 * 1024 ** 2
    total_mb      = total_bytes   / (1024 ** 2)
    free_mb       = free_bytes    / (1024 ** 2)
    largest_mb    = largest_chunk / (1024 ** 2)

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

    all_latents: dict[str, torch.Tensor] = {}
    for chunks_file in chunks_files:
        sf = open_safetensors(chunk_file)
        for key in sf.keys():
            all_latents[key] = sf.get_tensor(key)
        del sf
        chunks_file.unlink()
        logger.info(f"  Read and deleted {chunk_file.name}  ({len(all_latents)} latenti finora)")
        gc.collect()

    latent_h = resolution // 8
    latent_w = resolution // 8
    sf_metadata = {
        "total_images": str(len(all_latents)),
        "resolution":   str(resolution),
        "center_crop":  str(center_crop),
        "latent_h":     str(latent_h),
        "latent_w":     str(latent_w),
        "vae_scale":    "0.18215",
        "dtype":        "float16",
        "model":        pretrained_model_name_or_path,
        "device_used":  device,
        "errors":       str(n_errors),
    }
    logger.info(f"  Writing {output_sf.name} ({len(all_latents)} immagini)...")
    save_safetensors(all_latents, output_sf, metadata=sf_metadata)
    del all_latents
    gc.collect()

    size_mb = output_sf.stat().st_size / (1024 ** 2)
    logger.info(f"  ✓ Merge completed: {output_sf.name}  ({size_mb:.1f} MB)")
    try:
        chunkss_dir.rmdir()
        logger.info(f"  Cartella chunkss/ rimossa.")
    except OSError:
        pass
    return True

#  Pipeline principale
def preprocess_image_latents( image_dir: str, output_dir: str, pretrained_model_name_or_path: str,
                            resolution: int = 512, center_crop: bool = True, batch_size: int = 32,
                            device: str = "cuda:0", num_workers: int = 4,
    ):
    """Encode all images to VAE latents and write chunked safetensors."""
    latent_h     = resolution // 8
    latent_w     = resolution // 8
    kb_per_image = 4 * latent_h * latent_w * 2 / 1024
    max_ram_mb   = FLUSH_EVERY_N_BATCHES * batch_size * kb_per_image / 1024

    _t_start = time.time()
    n_gpus    = torch.cuda.device_count()
    gpu_names = (
        ", ".join(torch.cuda.get_device_name(i) for i in range(n_gpus))
        if n_gpus > 0 else "CPU"
    )

    logger.info("=" * 60)
    logger.info("START: IMAGE LATENTS PRE-ENCODING PIPELINE  [chunk safetensors]")
    logger.info(f"timestamp           : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"model               : {pretrained_model_name_or_path}")
    logger.info(f"image_dir           : {image_dir}")
    logger.info(f"output_dir          : {output_dir}")
    logger.info(f"resolution          : {resolution}x{resolution}")
    logger.info(f"center_crop         : {center_crop}")
    logger.info(f"batch_size          : {batch_size}")
    logger.info(f"num_workers (CPU)   : {num_workers}")
    logger.info(f"latent shape attesa : [4, {latent_h}, {latent_w}]")
    logger.info(f"peso stimato/img    : ~{kb_per_image:.1f} KB  (float16)")
    logger.info(f"flush ogni          : {FLUSH_EVERY_N_BATCHES} batch  "
                f"(≤ ~{max_ram_mb:.0f} MB RAM per i latenti)")
    logger.info(f"strategia disco     : chunks separati → merge finale se spazio sufficiente")
    logger.info(f"GPU disponibili     : {n_gpus}  ({gpu_names})")
    logger.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    output_dir_p  = Path(output_dir)
    chunkss_dir    = _chunk_dir(output_dir_p)
    output_sf     = output_dir_p / "image_latents.safetensors"
    metadata_path = output_dir_p / "metadata.json"

    already_done = _existing_ids_from_chunks(chunks_dir)
    if output_sf.exists():
        try:
            sf = open_safetensors(output_sf)
            already_done.update(sf.keys())
        except Exception:
            pass
    if already_done:
        logger.info(f"[Resume] {len(already_done)} immagini già presenti — saltate.")

    # Raccoglie file immagine
    image_dir_p = Path(image_dir)
    image_files = sorted(
        p
        for root, _, files in os.walk(image_dir_p, followlinks=True)
        for f in files
        if (p := Path(root) / f).suffix.lower() in SUPPORTED_EXTENSIONS
    )
    logger.info(f"Found {len(image_files)} images in {image_dir_p}")
    if not image_files:
        logger.error("Nessuna immagine trovata. Controlla --image_dir.")
        return

    images_to_process = [p for p in image_files if p.stem not in already_done]
    logger.info(
        f"To process: {len(images_to_process)} "
        f"(saltate: {len(image_files) - len(images_to_process)})"
    )

    if not images_to_process:
        logger.info("Tutte le immagini già processed. Proceeding to final merge if needed.")
        _merge_chunks(
            chunkss_dir, output_sf,
            resolution, center_crop, pretrained_model_name_or_path, device, 0,
        )
        return

    # PyTorch stesso avverte di questo; lo clampiam automaticamente.
    if n_gpus <= 1 and num_workers > 2:
        logger.warning(
            f"  [WARN] num_workers={num_workers} → clampato a 2 (singola GPU su Colab). "
            f"Usa --num_workers 2 per silenziare questo avviso."
        )
        num_workers = 2

    vae_model = load_vae(pretrained_model_name_or_path, device)

    dataset    = ImagePreprocessDataset(
        [str(p) for p in images_to_process], resolution, center_crop
    )
    use_pin    = (num_workers > 0) and torch.cuda.is_available()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin,
        collate_fn=collate_skip_errors,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    new_latents:   dict[str, torch.Tensor] = {}
    errors:        list[str]               = []
    chunks_idx      = _next_chunk_idx(chunks_dir)
    n_new_total    = 0
    t0             = time.time()

    for batch_num, (pixel_values, batch_ids) in enumerate(
        tqdm(dataloader, desc="VAE encode")
    ):
        if pixel_values is None:
            continue

        pixel_values = pixel_values.to(device, dtype=torch.float16, non_blocking=True)

        # senza crashare il processo intero.
        encode_ok = False
        current_pv = pixel_values
        current_ids = list(batch_ids)
        while not encode_ok:
            try:
                with torch.no_grad():
                    with autocast('cuda'):
                        latents = vae_model(current_pv)
                encode_ok = True
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                half = max(1, current_pv.shape[0] // 2)
                logger.warning(
                    f"  [OOM] batch_size {current_pv.shape[0]} → ridotto a {half}. "
                    f"Considera di usare --batch_size {half} per evitare questo."
                )
                if half == current_pv.shape[0]:
                    logger.error(
                        f"  [OOM] Impossibile processare anche con batch_size=1. "
                        f"Batch saltato."
                    )
                    for bid in (batch_ids or []):
                        errors.append(bid)
                    current_pv = None
                    break
                current_pv = current_pv[:half]
                current_ids = current_ids[:half]

        if current_pv is None:
            del pixel_values
            continue

        latents_cpu = latents.half().cpu()
        del current_pv, latents

        for i, image_id in enumerate(current_ids):
            new_latents[image_id] = latents_cpu[i].clone()
            n_new_total += 1

        del latents_cpu

        # Log velocità
        if n_new_total % (batch_size * 10) == 0:
            elapsed   = time.time() - t0
            img_per_s = n_new_total / max(elapsed, 1e-3)
            logger.info(f"  {n_new_total} immagini processed  ({img_per_s:.1f} img/s)")

        is_last = batch_num == len(dataloader) - 1
        if (batch_num + 1) % FLUSH_EVERY_N_BATCHES == 0 or is_last:
            if new_latents:
                _flush_chunk(new_latents, chunkss_dir, chunks_idx)
                chunks_idx += 1
                new_latents = {}
                gc.collect()
                torch.cuda.empty_cache()

    elapsed   = time.time() - t0
    img_per_s = n_new_total / max(elapsed, 1e-3)
    logger.info(
        f"Encoding completed: {n_new_total} images in {elapsed:.1f}s "
        f"({img_per_s:.1f} img/s | {len(errors)} errors)"
    )

    merged = _merge_chunks(
        chunkss_dir, output_sf,
        resolution, center_crop, pretrained_model_name_or_path, device,
        len(errors),
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

    sample_shape = [4, latent_h, latent_w]
    meta = {
        "total_images":    len(all_ids),
        "image_ids":       all_ids,
        "output_file":     "image_latents.safetensors" if merged else "chunks/",
        "chunks_dir":      str(chunks_dir) if not merged else None,
        "latent_shape":    sample_shape,
        "resolution":      resolution,
        "center_crop":     center_crop,
        "batch_size_used": batch_size,
        "vae_scale":       0.18215,
        "dtype":           "float16",
        "model":           pretrained_model_name_or_path,
        "device_used":     device,
        "errors":          errors,
        "elapsed_s":       round(time.time() - _t_start, 1),
    }
    with open(metadata_path, "w") as f:
        json.dump(meta, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    _total_elapsed = time.time() - _t_start
    _h = int(_total_elapsed // 3600)
    _m = int((_total_elapsed % 3600) // 60)
    _s = int(_total_elapsed % 60)
    _elapsed_str = f"{_h}h {_m:02d}m {_s:02d}s" if _h else f"{_m}m {_s:02d}s"

    logger.info("=" * 60)
    logger.info("COMPLETED: IMAGE LATENTS PRE-ENCODING")
    logger.info(f"end timestamp       : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"total time          : {_elapsed_str}")
    logger.info(f"immagini processed : {len(all_ids)}")
    logger.info(f"nuove questo run    : {n_new_total}")
    logger.info(f"errors              : {len(errors)}")
    logger.info(f"throughput          : {n_new_total / max(_total_elapsed, 1e-3):.1f} img/s")
    logger.info(f"final merge         : {'✓' if merged else '✗ (chunk in chunkss/)'}")
    logger.info(f"output_dir          : {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-calcola latenti VAE per tutte le immagini del dataset."
    )
    parser.add_argument("--image_dir", type=str, default="./Museart/images/")
    parser.add_argument("--output_dir", type=str, default="./output/image_latents/")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="stabilityai/stable-diffusion-2")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--center_crop", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    preprocess_image_latents(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        resolution=args.resolution,
        center_crop=args.center_crop,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
    )