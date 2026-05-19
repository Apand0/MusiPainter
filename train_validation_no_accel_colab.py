# @title train_validation_no_accel_colab.py
"""
DDP Training & Validation for Musipainter.
Supports single-GPU and multi-GPU (torchrun) modes.
"""

import argparse
import gc
import itertools
import logging
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.amp as _amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def autocast(enabled=True):
    return _amp.autocast('cuda', enabled=enabled)
try:
    from torch.amp import GradScaler  # PyTorch >= 2.3
except ImportError:
    from torch.cuda.amp import GradScaler

from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
from torch.utils.tensorboard import SummaryWriter

import datasets
import diffusers
import transformers

from accelerate import Accelerator
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from tqdm.auto import tqdm
from transformers import CLIPTokenizer

from modules.MusicToken.MusicToken_no_accel import MusicTokenWrapper
from dataloader_colab import Museart

check_min_version("0.12.0")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


#  DDP UTILITIES

def _setup_ddp():
    """
    [DDP-1] Inizializza il processo group NCCL se le variabili torchrun
    sono presenti (RANK, WORLD_SIZE, LOCAL_RANK). Altrimenti non fa nulla
    e il training gira in modalità single-GPU.

    Returns:
        is_ddp    : bool — True se DDP è attivo
        rank      : int  — rank globale del processo (0 se single-GPU)
        local_rank: int  — rank locale = indice GPU su questa macchina
        world_size: int  — numero totale di processi (1 se single-GPU)
    """
    rank       = int(os.environ.get("RANK",       -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE",  1))

    is_ddp = (rank != -1 and world_size > 1)

    if is_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    return is_ddp, max(rank, 0), max(local_rank, 0), world_size


def _teardown_ddp(is_ddp: bool):
    """[DDP-10] Distrugge il processo group alla fine del training."""
    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()


def _is_main_process(rank: int) -> bool:
    """True solo per rank 0 — usato per logging, salvataggio, ecc."""
    return rank == 0


def _barrier(is_ddp: bool):
    """Sincronizza tutti i processi DDP. No-op in single-GPU."""
    if is_ddp and dist.is_initialized():
        dist.barrier()


#  PREFETCH LOADER

class PrefetchLoader:
    """
    [OPT-PREFETCH] Pre-carica il prossimo batch su GPU mentre quello corrente
    viene elaborato, usando un CUDA stream separato.
    Invariato rispetto alla v7.
    """

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self._use_cuda = device.type == "cuda" and torch.cuda.is_available()

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        if not self._use_cuda:
            for batch in self.loader:
                yield batch
            return

        stream = torch.cuda.Stream()
        first = True
        batch = None

        for next_batch in self.loader:
            with torch.cuda.stream(stream):
                next_batch_gpu = {
                    k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in next_batch.items()
                }

            if not first:
                torch.cuda.current_stream().wait_stream(stream)
                yield batch
            else:
                first = False

            batch = next_batch_gpu

        if batch is not None:
            torch.cuda.current_stream().wait_stream(stream)
            yield batch


#  CHECKPOINT UTILITIES

def _unwrap_compiled(module):
    """
    [FIX-COMPILE-SAVE] Restituisce il modulo originale da un OptimizedModule
    (torch.compile) o da un DDP wrapper.
    """
    # Unwrap DDP
    m = module.module if isinstance(module, DDP) else module
    # Unwrap torch.compile
    return getattr(m, '_orig_mod', m)


def save_progress(module, save_path):
    """
    Save a module state_dict to disk.

    If *save_path* ends with '.safetensors', the file is written with the
    safetensors library (zero-copy, pickle-free). Otherwise falls back to
    torch.save for legacy .bin/.pt paths.
    """
    logger.info(f"Saving weights to {save_path}")
    state = _unwrap_compiled(module).state_dict()
    if str(save_path).endswith('.safetensors'):
        # Preferred format: no pickle, safe for distribution.
        from utils import save_safetensors
        save_safetensors(state, save_path)
    else:
        # Legacy path — backward compat with existing .bin checkpoints.
        torch.save(state, save_path)


def save_checkpoint(embedder, optimizer, scaler, lr_scheduler, global_step, 
                    best_vloss, lora_layers, save_path, best_model_path=None):
    """Save a full training checkpoint for resuming."""
    """[RESUME] Salva un checkpoint completo per poter riprendere il training."""
    ckpt = {
        "global_step":         global_step,
        "best_vloss":          best_vloss,
        "best_model_path":     best_model_path,
        "embedder_state_dict": _unwrap_compiled(embedder).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict":   scaler.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "timestamp":           timestamp,
    }
    if lora_layers is not None:
        ckpt["lora_state_dict"] = lora_layers.state_dict()
    torch.save(ckpt, save_path)
    logger.info(f"[RESUME] Checkpoint saved: {save_path} (step={global_step})")


def load_checkpoint(resume_path, embedder, optimizer, scaler, lr_scheduler,
                    lora_layers, device):
    """Load a checkpoint and restore all training states."""
    """[RESUME] Carica un checkpoint e ripristina tutti gli stati."""
    if not os.path.exists(resume_path):
        logger.warning(f"[RESUME] Checkpoint not found: {resume_path}. "
                       "Training starts from scratch.")
        return 0, float('inf'), None

    logger.info(f"[RESUME] Loading checkpoint: {resume_path}")
    ckpt = torch.load(resume_path, map_location=device)

    _unwrap_compiled(embedder).load_state_dict(ckpt["embedder_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])

    if lora_layers is not None and "lora_state_dict" in ckpt:
        lora_layers.load_state_dict(ckpt["lora_state_dict"])

    global_step     = ckpt["global_step"]
    best_vloss      = ckpt.get("best_vloss", float('inf'))
    best_model_path = ckpt.get("best_model_path", None)

    if best_model_path is not None and not os.path.exists(best_model_path):
        logger.warning(f"[RESUME] best_model_path dal checkpoint non trovato: {best_model_path}.")
        best_model_path = None

    logger.info(
        f"[RESUME] Resuming from step={global_step}, best_vloss={best_vloss:.4f}. "
        f"Checkpoint created: {ckpt.get('timestamp', 'N/A')}"
    )
    return global_step, best_vloss, best_model_path


def load_embedder_weights(weight_path, embedder, device, resume_step: int = 0):
    """
    [RESUME-WEIGHTS] Load only the embedder weights from a file.

    Accepts both .safetensors (preferred) and legacy .bin/.pt formats.
    If *weight_path* points to a .bin that does not exist on disk, the
    function automatically tries the corresponding .safetensors path before
    raising FileNotFoundError, ensuring forward compatibility after the
    format migration.

    Args:
        weight_path: path to a .safetensors or .bin weight file.
        embedder:    the FGAEmbedder module (possibly wrapped in DDP/compile).
        device:      map_location device string (e.g. "cuda:0").
        resume_step: step counter to resume from (returned unchanged).
    """
    # --- Automatic .bin → .safetensors fallback ---
    # After the format migration, callers that still pass a .bin path will
    # transparently pick up the new .safetensors file if it exists.
    resolved = weight_path
    if not os.path.exists(resolved) and resolved.endswith(".bin"):
        sf_candidate = resolved[:-4] + ".safetensors"
        if os.path.exists(sf_candidate):
            logger.info(
                f"[RESUME-WEIGHTS] .bin not found, using .safetensors: {sf_candidate}"
            )
            resolved = sf_candidate

    if not os.path.exists(resolved):
        raise FileNotFoundError(
            f"[RESUME-WEIGHTS] Weight file not found: {resolved}"
        )

    logger.info(f"[RESUME-WEIGHTS] Loading embedder weights from: {resolved}")

    if resolved.endswith(".safetensors"):
        from utils import load_safetensors
        state_dict = load_safetensors(resolved, device=str(device))
    else:
        state_dict = torch.load(resolved, map_location=device, weights_only=True)

    missing, unexpected = _unwrap_compiled(embedder).load_state_dict(state_dict, strict=True)
    if missing:
        logger.warning(f"[RESUME-WEIGHTS] Missing keys: {missing}")
    if unexpected:
        logger.warning(f"[RESUME-WEIGHTS] Unexpected keys: {unexpected}")

    logger.info(f"[RESUME-WEIGHTS] Embedder loaded from: {resolved} (resume_step={resume_step})")
    return resume_step, float('inf')


#  LABEL EMBEDDING CACHE

def build_label_embedding_cache(tokenizer, txt_embeddings, all_labels, device):
    """
    [OPT-COSINE-STACK] Precalcola target cosine loss come matrice GPU (N × D).
    Invariato rispetto alla v7.
    """
    label_list = sorted(set(all_labels))
    vecs = []
    valid_labels = []
    for label in label_list:
        ids = tokenizer([label]).data['input_ids'][0][1:-1]
        if not ids:
            continue
        ids_t = torch.tensor(ids, device=device)
        with torch.no_grad():
            vec = txt_embeddings[ids_t].mean(dim=0).detach().float()
        vecs.append(vec)
        valid_labels.append(label)

    label_matrix = torch.stack(vecs).to(device)
    label_to_idx = {lbl: i for i, lbl in enumerate(valid_labels)}
    logger.info(
        f"Label cache: {len(valid_labels)} unique labels, "
        f"matrix {list(label_matrix.shape)} on {device}"
    )
    return label_list, label_to_idx, label_matrix


#  VAE PRECOMPUTE

def precompute_vae_latents(vae, dataloader, device, use_amp):
    """
    [PERF-5] Calcola i latenti VAE per tutte le immagini del dataset.
    [DDP-5] Chiamata SOLO su rank 0 — gli altri rank leggono i latenti dal disco
    via LazyLatentIndex, oppure aspettano con _barrier() se serve il risultato.
    """
    logger.info("Precomputing VAE latents...")
    cache = {}
    vae.eval()
    vae = vae.to(device)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="VAE encode"):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            image_ids    = batch["image_id"]
            with autocast(enabled=use_amp):
                latents = vae.encode(
                    pixel_values.to(dtype=torch.float16)
                ).latent_dist.sample() * 0.18215
            for img_id, lat in zip(image_ids, latents):
                cache[img_id] = lat.cpu()
            del pixel_values, latents
    n = len(cache)
    if n > 0:
        sample = next(iter(cache.values()))
        size_mb = sample.numel() * sample.element_size() * n / (1024 ** 2)
        logger.info(
            f"VAE latent cache: {n} immagini, "
            f"shape={list(sample.shape)}, ~{size_mb:.0f} MB RAM (float16)"
        )
    return cache


#  DISK / TB UTILITIES

def _dir_size_mb(path: str) -> float:
    """Return total size of a directory in MB."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total / (1024 ** 2)


def _check_tb_size(writer, tb_dir: str, max_mb: int):
    """[MEM-3] Svuota la cartella TensorBoard se supera max_mb."""
    if max_mb <= 0:
        return writer
    size_mb = _dir_size_mb(tb_dir)
    if size_mb > max_mb:
        logger.warning(
            f"[MEM-3] TensorBoard runs/ uses {size_mb:.0f} MB > {max_mb} MB. "
            "Svuoto la cartella e riapro il writer."
        )
        writer.close()
        import shutil
        shutil.rmtree(tb_dir, ignore_errors=True)
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(os.path.join(tb_dir, f"restart_{timestamp}"))
    return writer


def _rotate_checkpoints(ckpt_paths: list, keep_n: int):
    """[RESUME] Mantiene solo gli ultimi keep_n checkpoint su disco."""
    if keep_n <= 0:
        return
    seen: dict = dict.fromkeys(ckpt_paths)
    ckpt_paths.clear()
    ckpt_paths.extend(seen.keys())
    while len(ckpt_paths) > keep_n:
        old = ckpt_paths.pop(0)
        if os.path.exists(old):
            os.remove(old)
            logger.info(f"[RESUME] Old checkpoint removed: {old}")


def _check_disk_space(output_dir: str, warn_gb: float = 1.0, critical_gb: float = 0.3):
    """[DISK-MONITOR] Controlla lo spazio libero e avvisa se necessario."""
    try:
        import shutil
        total, used, free = shutil.disk_usage(output_dir)
        free_gb  = free  / (1024 ** 3)
        used_gb  = used  / (1024 ** 3)
        total_gb = total / (1024 ** 3)
        working_dir_mb = _dir_size_mb(output_dir)

        if free_gb < critical_gb:
            logger.error(
                f"[DISK-CRITICAL] Disk space critically low on {output_dir}: "
                f"{free_gb:.2f} GB free / {total_gb:.1f} GB total. "
                f"Output dir: {working_dir_mb:.0f} MB. "
                "CRASH RISK — reduce --keep_last_n_checkpoints or clean the directory."
            )
        elif free_gb < warn_gb:
            logger.warning(
                f"[DISK-WARN] Disk space running low: {free_gb:.2f} GB free "
                f"({used_gb:.1f}/{total_gb:.1f} GB used). "
                f"Output dir: {working_dir_mb:.0f} MB."
            )
        else:
            logger.info(
                f"[DISK-OK] {free_gb:.2f} GB free "
                f"({used_gb:.1f}/{total_gb:.1f} GB). "
                f"Output dir: {working_dir_mb:.0f} MB."
            )
        return free_gb
    except Exception as e:
        logger.warning(f"[DISK-MONITOR] Unable to read disk space: {e}")
        return None


#  ARG PARSING

def parse_args():
    """Parse command-line arguments."""

    def _str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', '1'):
            return True
        if v.lower() in ('no', 'false', '0'):
            return False
        raise argparse.ArgumentTypeError(f"Valore booleano atteso, ricevuto: '{v}'")

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_steps", type=int, default=2500)
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default='stabilityai/stable-diffusion-2')
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="./Museart/")
    parser.add_argument("--placeholder_token", type=str, default="<*>")
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--multiple_tokens", type=_str2bool, default=False)
    parser.add_argument("--output_dir", type=str, default="./output/")
    parser.add_argument("--seed", type=int, default=8765)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=20000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-05)
    parser.add_argument("--scale_lr", type=_str2bool, default=False)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow_tf32", action="store_true", default=True)
    parser.add_argument("--report_to", type=str, default="none",
                        help="'tensorboard' o 'none'.")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--data_set", type=str, default='train',
                        choices=['train', 'validation', 'test'])
    parser.add_argument("--lambda_a", type=float, default=0.01)
    parser.add_argument("--lambda_b", type=float, default=0)
    parser.add_argument("--lambda_c", type=float, default=0.01)
    parser.add_argument("--run_name", type=str, default='MusicToken')
    parser.add_argument("--cosine_loss", type=_str2bool, default=True)
    parser.add_argument("--input_length", type=int, default=30)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--validation_batch_size", type=int, default=4)
    parser.add_argument("--lora", type=_str2bool, default=False)
    parser.add_argument("--revision", type=str, default=None, required=False)
    parser.add_argument("--embeddings_dir", type=str, default="./audio_embeddings/")
    parser.add_argument("--use_precomputed_embeddings", type=_str2bool, default=True)
    parser.add_argument("--validate_every_n_epochs", type=int, default=5)
    parser.add_argument("--validate_every_n_steps", type=int, default=0)
    parser.add_argument("--precompute_vae_latents", type=_str2bool, default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--max_tb_size_mb", type=int, default=200)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--workers_after_precompute", type=int, default=0)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--resume_from_embedder", type=str, default=None)
    parser.add_argument("--resume_step", type=int, default=0)
    parser.add_argument("--keep_last_n_checkpoints", type=int, default=2)
    parser.add_argument("--hf_cache_dir", type=str, default="/tmp/hf_model_cache")
    parser.add_argument("--latents_dir", type=str, default="./image_latents/")

    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.data_dir is None:
        raise ValueError("Specificare --data_dir.")

    args.image_latents_dir = args.latents_dir
    return args


#  TRAINING MAIN

def train_validation():
    """Main training and validation loop."""
    args = parse_args()

    is_ddp, rank, local_rank, world_size = _setup_ddp()
    is_main = _is_main_process(rank)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    #  BANNER (solo rank 0)
    if is_main:
        _n_gpus = torch.cuda.device_count()
        print(f"\n{'='*60}")
        print(f"  Environment       : {'KAGGLE' if os.path.exists('/kaggle') else 'COLAB/OTHER'}")
        print(f"  GPUs available    : {_n_gpus}")
        if is_ddp:
            print(f"  Mode              : DistributedDataParallel ({world_size}x GPU)")
            for _gi in range(_n_gpus):
                print(f"    cuda:{_gi} → {torch.cuda.get_device_name(_gi)}")
            print(f"  How to launch     : torchrun --nproc_per_node={_n_gpus} train_validation_no_accel_colab.py [args]")
        elif torch.cuda.is_available():
            print(f"  Mode              : Single GPU ({torch.cuda.get_device_name(0)})")
            print(f"  How to launch     : python train_validation_no_accel_colab.py [args]")
        else:
            print("  Mode              : CPU (very slow)")
        print(f"{'='*60}\n")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main else logging.WARNING,
        force=True,
    )
    if is_main:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()

    _hf_cache = args.hf_cache_dir

    # Accelerator usato solo per compatibilità import diffusers (nessuna funzione attiva)
    Accelerator()

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'weights'), exist_ok=True)

    #  TOKENIZER + SCHEDULER
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name,
                                                  cache_dir=_hf_cache)
    else:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer",
            cache_dir=_hf_cache,
        )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler",
        cache_dir=_hf_cache,
    )

    num_added_tokens = tokenizer.add_tokens(args.placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(f"Token {args.placeholder_token} già nel tokenizer.")

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps
            * args.train_batch_size * world_size  # [DDP] world_size invece di n_gpus locale
        )

    # TensorBoard solo su rank 0
    use_tb = (args.report_to != "none") and is_main
    tb_dir = os.path.join(args.output_dir, "runs")
    if use_tb:
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(os.path.join(tb_dir, f"music_to_image_{timestamp}"))
    else:
        writer = None
        if is_main:
            logger.info("[MEM-3] TensorBoard disabled (--report_to none)")

    #  DATASET
    args.data_set = 'train'
    train_dataset = Museart(args=args, tokenizer=tokenizer, logger=logger)

    if is_main:
        logger.info("=" * 60)
        logger.info("PRECOMPUTED DATA SUMMARY")
        _emb = getattr(train_dataset, 'audio_embeddings', None)
        if _emb is not None:
            from dataloader_colab import LazyEmbeddingIndex
            if isinstance(_emb, LazyEmbeddingIndex):
                _emb_files = sorted(_emb.embeddings_dir.glob("audio_embeddings_*.pt"))
                _emb_size  = sum(f.stat().st_size for f in _emb_files) / (1024**2)
                logger.info(f"  [AUDIO]  ✓ {len(_emb)} embeddings  |  "
                            f"{len(_emb_files)} chunks  |  {_emb_size:.1f} MB")
            else:
                logger.info(f"  [AUDIO]  ✓ {len(_emb)} embeddings (preloaded in RAM)")
        else:
            logger.info("  [AUDIO]  ✗ No precomputed embeddings — BEATs runtime")
        _lat = getattr(train_dataset, 'image_latents', None)
        if _lat is not None:
            from dataloader_colab import LazyLatentIndex
            if isinstance(_lat, LazyLatentIndex):
                _lat_files = sorted(_lat.latents_dir.glob("image_latents_*.pt"))
                _lat_size  = sum(f.stat().st_size for f in _lat_files) / (1024**2)
                logger.info(f"  [IMAGE]  ✓ {len(_lat)} latents     |  "
                            f"{len(_lat_files)} chunks  |  {_lat_size:.1f} MB")
        else:
            logger.info("  [IMAGE]  ✗ No precomputed latents — img_proc() + VAE runtime")
        logger.info("=" * 60)

    use_pin = (args.dataloader_num_workers > 0) and torch.cuda.is_available()

    if is_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )
        train_shuffle = False   # il sampler gestisce lo shuffle
    else:
        train_sampler = None
        train_shuffle = True

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=args.dataloader_num_workers,
        pin_memory=use_pin,
        persistent_workers=(args.dataloader_num_workers > 0),
        generator=torch.Generator() if not is_ddp else None,
    )

    overrode_max = False
    steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_epochs * steps_per_epoch
        overrode_max = True

    #  MODELLO
    model = MusicTokenWrapper(args)
    base_model = model  # riferimento diretto prima di DDP

    _placeholder_token_id = tokenizer.convert_tokens_to_ids(args.placeholder_token)
    base_model.text_encoder.resize_token_embeddings(len(tokenizer))
    base_model.set_placeholder_token_id(_placeholder_token_id)
    if is_main:
        logger.info(f"[FIX-PLACEHOLDER] placeholder token '{args.placeholder_token}' mapped to ID {_placeholder_token_id}")

    #  PRECALCOLO VAE
    train_latent_cache = {}
    valid_latent_cache = {}

    args.data_set = 'validation'
    validation_dataset = Museart(args=args, tokenizer=tokenizer, logger=logger)
    args.data_set = 'train'

    if args.precompute_vae_latents:
        if args.latents_dir:
            if is_main:
                logger.info(
                    "[PREIMG] latents_dir provided → "
                    "latents loaded via LazyLatentIndex in dataloader. "
                    "No runtime precomputation needed."
                )
            train_latent_cache = None
            valid_latent_cache = None
        else:
            # Solo rank 0 esegue il precalcolo VAE
            if is_main:
                logger.info("[DDP-5] Precomputing VAE latents on rank 0...")
                precompute_train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=args.train_batch_size * 2,
                    shuffle=False,
                    num_workers=args.dataloader_num_workers,
                    pin_memory=use_pin,
                )
                precompute_valid_loader = torch.utils.data.DataLoader(
                    validation_dataset,
                    batch_size=args.validation_batch_size * 2,
                    shuffle=False,
                    num_workers=args.dataloader_num_workers,
                    pin_memory=use_pin,
                )
                logger.info("[PERF-5] Precomputing train VAE latents...")
                train_latent_cache = precompute_vae_latents(
                    base_model.vae, precompute_train_loader, device,
                    use_amp=(args.mixed_precision == "fp16")
                )
                logger.info("[PERF-5] Precomputing validation VAE latents...")
                valid_latent_cache = precompute_vae_latents(
                    base_model.vae, precompute_valid_loader, device,
                    use_amp=(args.mixed_precision == "fp16")
                )
                del precompute_train_loader, precompute_valid_loader
                logger.info("[MEM-5] Moving VAE to CPU (before DDP wrap)")
                base_model.vae.to("cpu")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(
                    f"[PERF-5] VAE cache ready: "
                    f"{len(train_latent_cache)} train + {len(valid_latent_cache)} validation latents."
                )

            _barrier(is_ddp)
    else:
        if is_main:
            logger.info("[PERF-5] precompute_vae_latents=False: vae.encode() nel loop")

    #  GRADIENT CHECKPOINTING (prima di DDP)
    if args.gradient_checkpointing:
        try:
            base_model.unet.enable_gradient_checkpointing()
            if is_main:
                logger.info("[OPT-STE-GC] UNet gradient checkpointing enabled")
        except Exception as e:
            if is_main:
                logger.warning(f"[PERF-6] Gradient checkpointing unavailable: {e}")
    else:
        if is_main:
            logger.info("[OPT-STE-GC] Gradient checkpointing disabled (correct with STE)")

    model = model.to(device)

    # torch.compile su FGAEmbedder — sicuro sia in single-GPU che DDP
    try:
        base_model.embedder = torch.compile(
            base_model.embedder, mode="default", fullgraph=False
        )
        if is_main:
            logger.info("[OPT-COMPILE] FGAEmbedder compilato (compatibile con DDP)")
    except Exception as e:
        if is_main:
            logger.info(f"[OPT-COMPILE] Skipped: {e}")

    if is_ddp:
        if is_main:
            logger.info(
                f"[DDP-4] Wrapping model with DDP: rank={rank}, device={device}, world_size={world_size}"
            )
        _find_unused = args.lora
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=_find_unused,
            broadcast_buffers=False,         # buffer frozen non necessitano sync
        )
        if is_main:
            logger.info(
                f"[DDP-4] find_unused_parameters={_find_unused} "
                f"({'LoRA attivo' if _find_unused else 'STE garantisce grad su tutti i param'})"
            )
    else:
        if is_main:
            logger.info(f"[SINGLE-GPU] "
                        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Riferimento al modulo base (senza DDP wrapper)
    base_model = model.module if isinstance(model, DDP) else model

    if is_main:
        logger.info("[OPT-COMPILE-UNET] SKIPPED: frozen/DDP preferred for stability")

    #  OPTIMIZER + SCALER
    trainable_params = list(_unwrap_compiled(base_model.embedder).parameters())
    if args.lora:
        trainable_params += list(base_model.lora_layers.parameters())

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    if overrode_max:
        args.max_train_steps = args.num_epochs * steps_per_epoch
    args.num_epochs = math.ceil(args.max_train_steps / steps_per_epoch)

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*before.*optimizer.step.*")
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

    use_amp = (args.mixed_precision == "fp16") and torch.cuda.is_available()
    try:
        scaler = GradScaler('cuda', enabled=use_amp)
    except TypeError:
        scaler = GradScaler(enabled=use_amp)

    global_step = 0
    best_vloss  = float('inf')
    best_model_path = None

    if args.resume_from_checkpoint:
        if is_main:
            global_step, best_vloss, _resumed_best_path = load_checkpoint(
                args.resume_from_checkpoint,
                base_model.embedder,
                optimizer, scaler, lr_scheduler,
                base_model.lora_layers if args.lora else None,
                device,
            )
            best_model_path = _resumed_best_path
        if is_ddp:
            _gs_tensor = torch.tensor([global_step], dtype=torch.long, device=device)
            _bv_tensor = torch.tensor([best_vloss], dtype=torch.float64, device=device)
            dist.broadcast(_gs_tensor, src=0)
            dist.broadcast(_bv_tensor, src=0)
            global_step = _gs_tensor.item()
            best_vloss  = _bv_tensor.item()
            # Broadcast state_dict embedder
            for param in _unwrap_compiled(base_model.embedder).parameters():
                dist.broadcast(param.data, src=0)

    if args.resume_from_embedder:
        if args.resume_from_checkpoint:
            if is_main:
                logger.warning("[RESUME-BIN] --resume_from_embedder ignored: "
                               "--resume_from_checkpoint takes precedence.")
        else:
            if is_main:
                global_step, best_vloss = load_embedder_weights(
                    bin_path=args.resume_from_embedder,
                    embedder=base_model.embedder,
                    device=device,
                    resume_step=args.resume_step,
                )
            if is_ddp:
                _gs_tensor = torch.tensor([global_step], dtype=torch.long, device=device)
                dist.broadcast(_gs_tensor, src=0)
                global_step = _gs_tensor.item()
                for param in _unwrap_compiled(base_model.embedder).parameters():
                    dist.broadcast(param.data, src=0)

    #  VALIDATION DATALOADER
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.validation_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        pin_memory=use_pin,
        persistent_workers=(args.dataloader_num_workers > 0),
        generator=torch.Generator(),
    )

    n_workers_loop = args.workers_after_precompute if args.precompute_vae_latents else args.dataloader_num_workers
    if args.precompute_vae_latents and n_workers_loop != args.dataloader_num_workers:
        if is_main:
            logger.info(
                f"[MEM-6] Recreating train_dataloader: "
                f"num_workers {args.dataloader_num_workers} → {n_workers_loop}"
            )
        del train_dataloader
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=n_workers_loop,
            pin_memory=(n_workers_loop > 0) and torch.cuda.is_available(),
            persistent_workers=False,
            generator=torch.Generator() if not is_ddp else None,
        )
        steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    n_gpus = world_size  # in DDP = numero totale di processi = numero di GPU
    total_batch = args.train_batch_size * args.gradient_accumulation_steps * n_gpus

    _t_start = time.time()
    _gpu_info = (
        ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
        if torch.cuda.is_available() else "CPU"
    )

    if is_main:
        logger.info("=" * 60)
        logger.info("START: TRAIN & VALIDATION PIPELINE")
        logger.info(f"timestamp           : {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"GPU disponibili     : {torch.cuda.device_count()}  ({_gpu_info})")
        logger.info(f"Modalità            : {'DDP (' + str(world_size) + 'x GPU)' if is_ddp else 'Single GPU'}")
        logger.info(f"base_model          : {args.pretrained_model_name_or_path}")
        logger.info(f"train / validation samples: {len(train_dataset)} / {len(validation_dataset)}")
        logger.info(f"train_batch_size    : {args.train_batch_size}  (eff. {total_batch})")
        logger.info(f"grad_accum_steps    : {args.gradient_accumulation_steps}")
        logger.info(f"num_epochs          : {args.num_epochs}")
        logger.info(f"max_train_steps     : {args.max_train_steps}")
        logger.info(f"save_steps          : {args.save_steps}")
        logger.info(f"initial global_step : {global_step}  "
                    f"{'(resume)' if global_step > 0 else '(nuovo training)'}")
        logger.info(f"learning_rate       : {args.learning_rate:.2e}  ({args.lr_scheduler})")
        logger.info(f"lr_warmup_steps     : {args.lr_warmup_steps}")
        logger.info(f"AMP fp16            : {'ON' if use_amp else 'OFF'}")
        logger.info(f"validate_every_n    : {args.validate_every_n_epochs} epoch")
        logger.info(f"num_workers         : {args.dataloader_num_workers}  pin={use_pin}")
        logger.info(f"world_size (DDP)    : {world_size}")
        logger.info(f"embeddings_dir      : {args.embeddings_dir}")
        logger.info(f"latents_dir         : {args.latents_dir or 'None (img_proc runtime)'}")
        logger.info(f"precompute_vae      : {args.precompute_vae_latents}")
        logger.info(f"grad_checkpointing  : {args.gradient_checkpointing}")
        logger.info(f"output_dir          : {args.output_dir}")
        logger.info(f"TensorBoard         : "
                    f"{'ON (max ' + str(args.max_tb_size_mb) + ' MB)' if use_tb else 'OFF'}")
        logger.info(f"resume_from         : {args.resume_from_checkpoint or 'None'}")
        logger.info(f"keep_last_n_ckpt    : {args.keep_last_n_checkpoints}")
        logger.info("=" * 60)

    _prediction_type = noise_scheduler.config.prediction_type

    txt_embeddings = base_model.text_encoder.get_input_embeddings().weight
    _label_to_idx = None
    _label_matrix = None
    if args.cosine_loss:
        all_labels = train_dataset.label + validation_dataset.label
        _, _label_to_idx, _label_matrix = build_label_embedding_cache(
            tokenizer, txt_embeddings, all_labels, device
        )

    #  TRAINING LOOP
    epoch_number = global_step // steps_per_epoch if steps_per_epoch > 0 else 0
    resume_ckpt_paths: list = []
    _last_validated_step: int = -1

    def _get_prefetch():
        return PrefetchLoader(train_dataloader, device)

    if is_main:
        # Diagnostica percorso latenti
        _lat_check = getattr(train_dataset, 'image_latents', None)
        if _lat_check is not None:
            logger.info(f"[DIAG] ✓ PATH 1: LazyLatentIndex active ({len(_lat_check)} latenti). "
                        "VAE NON usato nel loop.")
        elif train_latent_cache:
            logger.info(f"[DIAG] ✓ PERCORSO 2: cache in-memory attiva "
                        f"({len(train_latent_cache)} latenti).")
        else:
            logger.warning("[DIAG] ✗ PATH 3: VAE encode at runtime — SLOW!")

    _steps_done_in_first_epoch = (
        global_step - (global_step // steps_per_epoch) * steps_per_epoch
        if steps_per_epoch > 0 else 0
    )
    batches_to_skip_first_epoch = _steps_done_in_first_epoch * args.gradient_accumulation_steps

    _noise_buf = None
    _ts_buf    = None
    _step_start_time = None

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not is_main)
    progress_bar.set_description("Steps")

    for epoch in range(epoch_number, args.num_epochs):

        if is_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        running_train_loss = torch.tensor(0.0, device=device)
        loss_count = 0
        if is_main:
            print(f'\nEPOCH {epoch + 1}/{args.num_epochs}  (global_step={global_step})')
        model.train()
        if args.lora:
            base_model.unet.eval()
            base_model.lora_layers.train()
        optimizer.zero_grad()

        _skip = batches_to_skip_first_epoch if epoch == epoch_number else 0
        _enumerated = enumerate(
            itertools.islice(_get_prefetch(), _skip, None), start=_skip
        )

        for i, batch in _enumerated:

            audio_features = batch["audio_features"]
            input_ids      = batch["input_ids"]

            with autocast(enabled=use_amp):
                # Latenti VAE: tre percorsi in ordine di priorità
                is_pre = batch.get("is_precomputed_latent", None)
                if is_pre is not None and bool(is_pre.all()):
                    latents = batch["pixel_values"].to(dtype=torch.float16)
                elif train_latent_cache:
                    latents = torch.stack(
                        [train_latent_cache[iid] for iid in batch["image_id"]]
                    ).to(device, non_blocking=True)
                else:
                    pixel_values = batch["pixel_values"]
                    with torch.no_grad():
                        latents = base_model.vae.encode(
                            pixel_values.to(dtype=torch.float16)
                        ).latent_dist.sample() * 0.18215

                bsz = latents.shape[0]
                if _noise_buf is None or _noise_buf.shape != latents.shape:
                    _noise_buf = torch.empty(latents.shape, dtype=torch.float32, device=device)
                    _ts_buf    = torch.empty((bsz,), dtype=torch.long, device=device)
                _noise_buf.normal_()
                _ts_buf.random_(0, noise_scheduler.config.num_train_timesteps)
                noise     = _noise_buf
                timesteps = _ts_buf
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                if _prediction_type == "epsilon":
                    target = noise
                elif _prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps).float()
                else:
                    raise ValueError(f"Unknown prediction type: {_prediction_type}")

                model_pred, audio_token = model(
                    audio_features,
                    input_ids,
                    noisy_latents,
                    timesteps,
                )

                loss = F.mse_loss(model_pred, target, reduction="mean")

                norm_dim = 2 if audio_token.dim() > 2 else 1
                _reg = args.lambda_a * torch.mean(torch.abs(audio_token))
                if args.lambda_b > 0:
                    _reg = _reg + args.lambda_b * (
                        torch.norm(audio_token, p=2, dim=norm_dim) ** 2
                    ).mean()
                loss = loss + _reg

                if args.cosine_loss and _label_to_idx is not None:
                    labels = batch['label']
                    row_idxs = [_label_to_idx[lbl] for lbl in labels if lbl in _label_to_idx]
                    aud_idxs = [j for j, lbl in enumerate(labels) if lbl in _label_to_idx]
                    if row_idxs:
                        idx_t     = torch.tensor(row_idxs, dtype=torch.long, device=device)
                        aud_idx_t = torch.tensor(aud_idxs, dtype=torch.long, device=device)
                        ct        = _label_matrix.index_select(0, idx_t)
                        at        = audio_token.index_select(0, aud_idx_t)
                        embedds   = at[:, -1, :] if args.multiple_tokens else at
                        embedds_n = F.normalize(embedds.float(), dim=1)
                        ct_n      = F.normalize(ct.float(), dim=1)
                        cs        = (embedds_n * ct_n).sum(dim=1).mean()
                        loss      = loss + args.lambda_c * (1 - cs) ** 2

                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if ((i + 1) % args.gradient_accumulation_steps == 0) or \
               (i + 1 == len(train_dataloader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if is_main:
                    progress_bar.update(1)
                global_step += 1

                if is_main:
                    if global_step == 10:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        _step_start_time = time.time()
                    elif global_step == 40:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        _elapsed = time.time() - _step_start_time
                        _sps = _elapsed / 30
                        _eta_h = int(args.max_train_steps * _sps // 3600)
                        _eta_m = int((args.max_train_steps * _sps % 3600) // 60)
                        logger.info(
                            f"[PERF] Real speed step 10-40 (post-STE): {_sps:.2f}s/step  "
                            f"→ ETA {args.max_train_steps} steps: ~{_eta_h}h {_eta_m:02d}m"
                        )

                _loss_val = loss.detach() * args.gradient_accumulation_steps
                running_train_loss += _loss_val
                loss_count += 1

                if is_main and global_step % 20 == 0:
                    _lv_display = running_train_loss.item() / max(loss_count, 1)
                    progress_bar.set_postfix(
                        loss=f"{_lv_display:.4f}",
                        lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                    )

            if is_main and global_step % args.save_steps == 0 and global_step > 0:
                _ckpt_path_this_step = os.path.join(
                    args.output_dir,
                    f"weights/checkpoint_step{global_step}.pt"
                )
                _already_saved_this_step = (_ckpt_path_this_step in resume_ckpt_paths)
                if not _already_saved_this_step:
                    save_progress(
                        base_model.embedder,
                        os.path.join(args.output_dir,
                                     f"weights/{args.run_name}_embeds-step{global_step}.safetensors")
                    )
                    if args.lora:
                        save_progress(
                            base_model.lora_layers,
                            os.path.join(args.output_dir,
                                         f"weights/{args.run_name}_lora-step{global_step}.safetensors")
                        )
                    save_checkpoint(
                        embedder=base_model.embedder,
                        optimizer=optimizer,
                        scaler=scaler,
                        lr_scheduler=lr_scheduler,
                        global_step=global_step,
                        best_vloss=best_vloss,
                        lora_layers=base_model.lora_layers if args.lora else None,
                        save_path=_ckpt_path_this_step,
                        best_model_path=best_model_path,
                    )
                    resume_ckpt_paths.append(_ckpt_path_this_step)
                    _rotate_checkpoints(resume_ckpt_paths, args.keep_last_n_checkpoints)
                    _check_disk_space(args.output_dir)

            _barrier(is_ddp)

            # VALIDATION MID-EPOCH
            if (args.validate_every_n_steps > 0
                    and global_step % args.validate_every_n_steps == 0
                    and global_step > 0
                    and global_step < args.max_train_steps
                    and _last_validated_step != global_step):

                if is_main:
                    _rtl_mid = running_train_loss.item()
                    _avg_train_mid = _rtl_mid / max(loss_count, 1)
                    running_valid_loss_mid = 0.0
                    model.eval()
                    with torch.no_grad():
                        for _i_v, _vb in enumerate(validation_dataloader):
                            _af   = _vb["audio_features"].to(device, non_blocking=True)
                            _iids = _vb["input_ids"].to(device, non_blocking=True)
                            with autocast(enabled=use_amp):
                                _is_pre = _vb.get("is_precomputed_latent", None)
                                if _is_pre is not None and bool(_is_pre.all()):
                                    _lats = _vb["pixel_values"].to(device, dtype=torch.float16, non_blocking=True)
                                elif valid_latent_cache:
                                    _lats = torch.stack(
                                        [valid_latent_cache[iid] for iid in _vb["image_id"]]
                                    ).to(device, non_blocking=True)
                                else:
                                    _pv = _vb["pixel_values"].to(device, non_blocking=True)
                                    _lats = base_model.vae.encode(
                                        _pv.to(dtype=torch.float16)
                                    ).latent_dist.sample() * 0.18215
                                _nv = torch.randn(_lats.shape, dtype=torch.float32, device=device)
                                _bv = _lats.shape[0]
                                _tv = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                                    (_bv,), device=device).long()
                                _nl = noise_scheduler.add_noise(_lats, _nv, _tv)
                                _mp, _at = model(_af, _iids, _nl, _tv)
                                _tgt = _nv if _prediction_type == "epsilon" \
                                    else noise_scheduler.get_velocity(_lats, _nv, _tv).float()
                                _lv = F.mse_loss(_mp, _tgt, reduction="mean")
                                _nd = 2 if _at.dim() > 2 else 1
                                _reg_v = args.lambda_a * torch.mean(torch.abs(_at))
                                if args.lambda_b > 0:
                                    _reg_v = _reg_v + args.lambda_b * (
                                        torch.norm(_at, p=2, dim=_nd) ** 2
                                    ).mean()
                                _lv = _lv + _reg_v
                                if args.cosine_loss and _label_to_idx is not None:
                                    _lbs = _vb['label']
                                    _ridxs = [_label_to_idx[l] for l in _lbs if l in _label_to_idx]
                                    _aidxs = [j for j, l in enumerate(_lbs) if l in _label_to_idx]
                                    if _ridxs:
                                        _idxtv = torch.tensor(_ridxs, device=device)
                                        _ct = _label_matrix.index_select(0, _idxtv)
                                        _ae = _at[_aidxs]
                                        _em = _ae[:, -1, :] if args.multiple_tokens else _ae
                                        _em_n = F.normalize(_em.float(), dim=1)
                                        _ct_n = F.normalize(_ct.float(), dim=1)
                                        _cs = (_em_n * _ct_n).sum(dim=1).mean()
                                        _lv = _lv + args.lambda_c * (1 - _cs) ** 2
                            running_valid_loss_mid += _lv.item()
                    _avg_valid_mid = running_valid_loss_mid / max(_i_v + 1, 1)
                    print(f'[MID-EPOCH step={global_step}]  train={_avg_train_mid:.4f}  valid={_avg_valid_mid:.4f}')
                    if _avg_valid_mid < best_vloss:
                        best_vloss = _avg_valid_mid
                        _new_best = os.path.join(args.output_dir, f'best_model_embedder_{timestamp}.safetensors')
                        _new_best_tmp = _new_best + ".tmp"
                        # Save best embedder in safetensors format (pickle-free).
                        from utils import save_safetensors as _sf_save
                        _sf_save(_unwrap_compiled(base_model.embedder).state_dict(), _new_best_tmp)
                        if best_model_path is not None and os.path.exists(best_model_path) and best_model_path != _new_best:
                            os.remove(best_model_path)
                            logger.info(f"[MEM-2] Removed old best: {best_model_path}")
                        os.replace(_new_best_tmp, _new_best)
                        if args.lora:
                            _new_best_lora = _new_best.replace(
                                'best_model_embedder_', 'best_model_lora_'
                            )
                            _new_best_lora_tmp = _new_best_lora + ".tmp"
                            # LoRA best weights also saved in safetensors format.
                            _sf_save(base_model.lora_layers.state_dict(), _new_best_lora_tmp)
                            os.replace(_new_best_lora_tmp, _new_best_lora)
                            logger.info(f"New best LoRA (mid-epoch): {_new_best_lora}")
                        best_model_path = _new_best
                        logger.info(f"New best (mid-epoch step={global_step}): vloss={best_vloss:.4f}")
                    model.train()
                    base_model.unet.eval()
                    base_model.text_encoder.eval()
                    base_model.vae.eval()
                    if args.lora:
                        base_model.lora_layers.train()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                _barrier(is_ddp)
                _last_validated_step = global_step

            if global_step >= args.max_train_steps:
                break

        _rtl_scalar = running_train_loss.item() if hasattr(running_train_loss, 'item') else running_train_loss
        avg_loss_train = _rtl_scalar / max(loss_count, 1)
        batches_to_skip_first_epoch = 0

        #  VALIDATION DI FINE EPOCH
        do_val = (
            (args.validate_every_n_steps > 0 and global_step % args.validate_every_n_steps == 0 and global_step > 0)
            or (args.validate_every_n_steps == 0 and ((epoch + 1) % args.validate_every_n_epochs == 0))
            or (global_step >= args.max_train_steps)
        )
        if _last_validated_step == global_step:
            do_val = False

        if do_val:
            if is_main:
                if use_tb and writer is not None and args.max_tb_size_mb > 0:
                    writer = _check_tb_size(writer, tb_dir, args.max_tb_size_mb)

                running_valid_loss = 0.0
                model.eval()
                with torch.no_grad():
                    for i, vb in enumerate(validation_dataloader):
                        af   = vb["audio_features"].to(device, non_blocking=True)
                        iids = vb["input_ids"].to(device, non_blocking=True)
                        with autocast(enabled=use_amp):
                            is_pre = vb.get("is_precomputed_latent", None)
                            if is_pre is not None and bool(is_pre.all()):
                                lats = vb["pixel_values"].to(device, dtype=torch.float16, non_blocking=True)
                            elif valid_latent_cache:
                                lats = torch.stack(
                                    [valid_latent_cache[iid] for iid in vb["image_id"]]
                                ).to(device, non_blocking=True)
                            else:
                                pv = vb["pixel_values"].to(device, non_blocking=True)
                                lats = base_model.vae.encode(
                                    pv.to(dtype=torch.float16)
                                ).latent_dist.sample() * 0.18215
                            nv = torch.randn(lats.shape, dtype=torch.float32, device=device)
                            bv = lats.shape[0]
                            tv = torch.randint(
                                0, noise_scheduler.config.num_train_timesteps,
                                (bv,), device=device
                            ).long()
                            nl = noise_scheduler.add_noise(lats, nv, tv)
                            mp, at = model(af, iids, nl, tv)
                            tgt = nv if _prediction_type == "epsilon" \
                                else noise_scheduler.get_velocity(lats, nv, tv).float()
                            lv = F.mse_loss(mp, tgt, reduction="mean")
                            nd = 2 if at.dim() > 2 else 1
                            _reg_v = args.lambda_a * torch.mean(torch.abs(at))
                            if args.lambda_b > 0:
                                _reg_v = _reg_v + args.lambda_b * (
                                    torch.norm(at, p=2, dim=nd) ** 2
                                ).mean()
                            lv = lv + _reg_v
                            if args.cosine_loss and _label_to_idx is not None:
                                lbs = vb['label']
                                row_idxs_v = [_label_to_idx[l] for l in lbs if l in _label_to_idx]
                                aud_idxs_v = [j for j, l in enumerate(lbs) if l in _label_to_idx]
                                if row_idxs_v:
                                    idx_tv = torch.tensor(row_idxs_v, device=device)
                                    ct     = _label_matrix.index_select(0, idx_tv)
                                    ae     = at[aud_idxs_v]
                                    em     = ae[:, -1, :] if args.multiple_tokens else ae
                                    em_n   = F.normalize(em.float(), dim=1)
                                    ct_n   = F.normalize(ct.float(), dim=1)
                                    cs     = (em_n * ct_n).sum(dim=1).mean()
                                    lv     = lv + args.lambda_c * (1 - cs) ** 2
                        running_valid_loss += lv.item()

                avg_valid = running_valid_loss / max(i + 1, 1)
                print(f'  → LOSS  train={avg_loss_train:.4f}  valid={avg_valid:.4f}')

                if use_tb and writer is not None:
                    writer.add_scalars(
                        'Training vs. Validation Loss',
                        {'Training': avg_loss_train, 'Validation': avg_valid},
                        epoch + 1,
                    )
                    writer.flush()

                if avg_valid < best_vloss:
                    best_vloss = avg_valid
                    new_best_path = os.path.join(
                        args.output_dir, f'best_model_embedder_{timestamp}.safetensors'
                    )
                    new_best_path_tmp = new_best_path + ".tmp"
                    # Save best embedder weights in safetensors format (pickle-free).
                    from utils import save_safetensors as _sf_save_ep
                    _sf_save_ep(_unwrap_compiled(base_model.embedder).state_dict(), new_best_path_tmp)
                    if best_model_path is not None and os.path.exists(best_model_path) and best_model_path != new_best_path:
                        os.remove(best_model_path)
                        logger.info(f"[MEM-2] Removed old best: {best_model_path}")
                    os.replace(new_best_path_tmp, new_best_path)
                    if args.lora:
                        _best_lora_path = new_best_path.replace(
                            'best_model_embedder_', 'best_model_lora_'
                        )
                        _best_lora_tmp = _best_lora_path + ".tmp"
                        # LoRA best weights also saved in safetensors format.
                        _sf_save_ep(base_model.lora_layers.state_dict(), _best_lora_tmp)
                        os.replace(_best_lora_tmp, _best_lora_path)
                        logger.info(f"  ✓ New best LoRA: {_best_lora_path}")
                    best_model_path = new_best_path
                    logger.info(
                        f"  ✓ New best embedder: {new_best_path} "
                        f"(vloss={best_vloss:.4f})"
                    )

                model.train()
                base_model.unet.eval()
                base_model.text_encoder.eval()
                base_model.vae.eval()
                if args.lora:
                    base_model.lora_layers.train()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            _barrier(is_ddp)

        else:
            if is_main:
                print(f'  → LOSS  train={avg_loss_train:.4f}  [validation skip]')

        if global_step >= args.max_train_steps:
            break

    #  SALVATAGGIO FINALE (solo rank 0)
    if is_main:
        save_progress(base_model.embedder,
                      os.path.join(args.output_dir, "learned_embeds.safetensors"))
        if args.lora:
            save_progress(base_model.lora_layers,
                          os.path.join(args.output_dir, "learned_embeds_lora_layers.safetensors"))

        _total_elapsed = time.time() - _t_start
        _h = int(_total_elapsed // 3600)
        _m = int((_total_elapsed % 3600) // 60)
        _s = int(_total_elapsed % 60)
        _elapsed_str = f"{_h}h {_m:02d}m {_s:02d}s" if _h else f"{_m}m {_s:02d}s"

        _output_files = []
        _weights_dir = os.path.join(args.output_dir, "weights")
        for _scan_dir in [args.output_dir, _weights_dir]:
            if not os.path.isdir(_scan_dir):
                continue
            for _fname in sorted(os.listdir(_scan_dir)):
                _fpath = os.path.join(_scan_dir, _fname)
                if os.path.isfile(_fpath) and not _fname.startswith("."):
                    _size_mb = os.path.getsize(_fpath) / (1024 ** 2)
                    _rel = os.path.relpath(_fpath, args.output_dir)
                    _output_files.append((_rel, _size_mb))
        _total_output_mb = sum(s for _, s in _output_files)

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info(f"end timestamp       : {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"total time          : {_elapsed_str}  ({_total_elapsed:.0f}s)")
        logger.info(f"total steps         : {global_step}/{args.max_train_steps}")
        logger.info(f"best validation loss: {best_vloss:.4f}")
        logger.info(f"best embedder       : {best_model_path or 'N/A (nessuna validation)'}")
        logger.info(f"avg time/step       : {_total_elapsed / max(global_step, 1):.2f}s")
        logger.info("  --- Output files ---")
        for _rel, _size_mb in _output_files:
            logger.info(f"    {_rel:<45} {_size_mb:>7.2f} MB")
        logger.info("  --- Total output ---")
        logger.info(f"{len(_output_files)} file  —  total space: {_total_output_mb:.2f} MB")
        logger.info(f"output_dir          : {args.output_dir}")
        logger.info("=" * 60)

        if use_tb and writer is not None:
            writer.close()

    _barrier(is_ddp)
    _teardown_ddp(is_ddp)

    return best_vloss


if __name__ == "__main__":
    train_validation()