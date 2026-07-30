# @title train_ef.py
"""
train_ef.py — DDP Training & Validation per Musipainter (Branch FuseLIP / Early Fusion).

NOTE: rinominato da train_validation_no_accel_colab.py per convivere con la
variante Cross-Attention (train_ca.py) nello stesso repository, selezionabile
tramite musipainter_train.py --architecture Musipainter-EF.
Unica modifica rispetto all'originale: l'import di MusicTokenWrapper punta
al modulo rinominato modules.MusicToken.MusicToken_no_accel_ef.
Nessun'altra logica è stata alterata.
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
    from torch.amp import GradScaler
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

from modules.MusicToken.MusicToken_no_accel_ef import MusicTokenWrapper
from dataloader_colab import Museart

check_min_version("0.12.0")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
#  DDP UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _setup_ddp():
    rank       = int(os.environ.get("RANK",       -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE",  1))
    is_ddp = (rank != -1 and world_size > 1)
    if is_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return is_ddp, max(rank, 0), max(local_rank, 0), world_size


def _teardown_ddp(is_ddp: bool):
    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()


def _is_main_process(rank: int) -> bool:
    return rank == 0


def _barrier(is_ddp: bool):
    if is_ddp and dist.is_initialized():
        dist.barrier()


# ─────────────────────────────────────────────────────────────────────────────
#  PREFETCH LOADER
# ─────────────────────────────────────────────────────────────────────────────

class PrefetchLoader:
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


# ─────────────────────────────────────────────────────────────────────────────
#  CHECKPOINT UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap_compiled(module):
    m = module.module if isinstance(module, DDP) else module
    return getattr(m, '_orig_mod', m)


def save_progress(module, save_path):
    logger.info(f"Saving weights to {save_path}")
    state = _unwrap_compiled(module).state_dict()
    if str(save_path).endswith('.safetensors'):
        from modules.preprocess.utils import save_safetensors
        save_safetensors(state, save_path)
    else:
        torch.save(state, save_path)


def save_checkpoint(embedder, optimizer, scaler, lr_scheduler, global_step,
                    best_vloss, lora_layers, save_path, best_model_path=None):
    ckpt = {
        "global_step":             global_step,
        "best_vloss":              best_vloss,
        "best_model_path":         best_model_path,
        "embedder_state_dict":     _unwrap_compiled(embedder).state_dict(),
        "optimizer_state_dict":    optimizer.state_dict(),
        "scaler_state_dict":       scaler.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "timestamp":               timestamp,
        "arch":                    "early_fusion_v9",
    }
    if lora_layers is not None:
        ckpt["lora_state_dict"] = lora_layers.state_dict()
    torch.save(ckpt, save_path)
    logger.info(f"[RESUME] Checkpoint saved: {save_path} (step={global_step})")


def load_checkpoint(resume_path, embedder, optimizer, scaler, lr_scheduler,
                    lora_layers, device):
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
        logger.warning(f"[RESUME] best_model_path non trovato su disco: {best_model_path}.")
        best_model_path = None
    _arch = ckpt.get("arch", "early_fusion_v9")
    if _arch != "early_fusion_v9":
        logger.warning(
            f"[RESUME] Checkpoint arch='{_arch}' — addestrato con architettura diversa. "
            "I pesi potrebbero non essere compatibili con EarlyFusionEncoder."
        )
    logger.info(f"[RESUME] Ripresa da step={global_step}, best_vloss={best_vloss:.4f}.")
    return global_step, best_vloss, best_model_path


def load_embedder_weights(weight_path, embedder, device, resume_step: int = 0):
    resolved = weight_path
    if not os.path.exists(resolved) and resolved.endswith(".bin"):
        sf_candidate = resolved[:-4] + ".safetensors"
        if os.path.exists(sf_candidate):
            logger.info(f"[RESUME-WEIGHTS] .bin non trovato, usando .safetensors: {sf_candidate}")
            resolved = sf_candidate
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"[RESUME-WEIGHTS] File pesi non trovato: {resolved}")
    logger.info(f"[RESUME-WEIGHTS] Loading embedder weights from: {resolved}")
    if resolved.endswith(".safetensors"):
        from modules.preprocess.utils import load_safetensors
        state_dict = load_safetensors(resolved, device=str(device))
    else:
        state_dict = torch.load(resolved, map_location=device, weights_only=True)
    missing, unexpected = _unwrap_compiled(embedder).load_state_dict(state_dict, strict=True)
    if missing:
        logger.warning(f"[RESUME-WEIGHTS] Missing keys: {missing}")
    if unexpected:
        logger.warning(f"[RESUME-WEIGHTS] Unexpected keys: {unexpected}")
    logger.info(f"[RESUME-WEIGHTS] EarlyFusionEncoder caricato (resume_step={resume_step})")
    return resume_step, float('inf')


# ─────────────────────────────────────────────────────────────────────────────
#  LABEL EMBEDDING CACHE
# ─────────────────────────────────────────────────────────────────────────────

def build_label_embedding_cache(tokenizer, token_embedding, all_labels, device):
    """
    Pre-calcola i vettori target per la cosine loss per ogni label unico.

    Media degli embedding CLIP per le parole del label (esclusi BOS/EOS),
    poi normalizzata L2 per produrre l̂ come da Musipainter eq. (2/5).
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
            vec = token_embedding(ids_t).float().mean(dim=0).detach()
            vec = F.normalize(vec.unsqueeze(0), dim=1).squeeze(0)
        vecs.append(vec)
        valid_labels.append(label)

    label_matrix = torch.stack(vecs).to(device)
    label_to_idx = {lbl: i for i, lbl in enumerate(valid_labels)}
    logger.info(
        f"Label cache: {len(valid_labels)} label unici, "
        f"matrix {list(label_matrix.shape)} su {device}"
    )
    return label_list, label_to_idx, label_matrix


# ─────────────────────────────────────────────────────────────────────────────
#  VAE PRECOMPUTE
# ─────────────────────────────────────────────────────────────────────────────

def precompute_vae_latents(vae, dataloader, device, use_amp):
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
            f"VAE latent cache: {n} immagini, shape={list(sample.shape)}, "
            f"~{size_mb:.0f} MB (float16)"
        )
    return cache


# ─────────────────────────────────────────────────────────────────────────────
#  DISK / TB UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _dir_size_mb(path: str) -> float:
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
    if max_mb <= 0:
        return writer
    size_mb = _dir_size_mb(tb_dir)
    if size_mb > max_mb:
        logger.warning(
            f"TensorBoard runs/ usa {size_mb:.0f} MB > {max_mb} MB. "
            "Pulizia e riapertura writer."
        )
        writer.close()
        import shutil
        shutil.rmtree(tb_dir, ignore_errors=True)
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(os.path.join(tb_dir, f"restart_{timestamp}"))
    return writer


def _rotate_checkpoints(ckpt_paths: list, keep_n: int):
    if keep_n <= 0:
        return
    seen = dict.fromkeys(ckpt_paths)
    ckpt_paths.clear()
    ckpt_paths.extend(seen.keys())
    while len(ckpt_paths) > keep_n:
        old = ckpt_paths.pop(0)
        if os.path.exists(old):
            os.remove(old)
            logger.info(f"[RESUME] Vecchio checkpoint rimosso: {old}")


def _check_disk_space(output_dir: str, warn_gb: float = 1.0, critical_gb: float = 0.3):
    try:
        import shutil
        total, used, free = shutil.disk_usage(output_dir)
        free_gb  = free  / (1024 ** 3)
        used_gb  = used  / (1024 ** 3)
        total_gb = total / (1024 ** 3)
        working_dir_mb = _dir_size_mb(output_dir)
        if free_gb < critical_gb:
            logger.error(
                f"[DISK-CRITICAL] {free_gb:.2f} GB liberi / {total_gb:.1f} GB totali. "
                f"Output dir: {working_dir_mb:.0f} MB. RISCHIO CRASH."
            )
        elif free_gb < warn_gb:
            logger.warning(
                f"[DISK-WARN] {free_gb:.2f} GB liberi ({used_gb:.1f}/{total_gb:.1f} GB)."
            )
        else:
            logger.info(
                f"[DISK-OK] {free_gb:.2f} GB liberi. Output dir: {working_dir_mb:.0f} MB."
            )
        return free_gb
    except Exception as e:
        logger.warning(f"[DISK-MONITOR] Impossibile leggere spazio disco: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  ARG PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    def _str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', '1'):
            return True
        if v.lower() in ('no', 'false', '0'):
            return False
        raise argparse.ArgumentTypeError(f"Boolean expected, got: '{v}'")

    parser = argparse.ArgumentParser()

    from modules.preprocess.argparse_multiembedding import add_multiembedding_args
    add_multiembedding_args(parser)

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
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--data_set", type=str, default='train',
                        choices=['train', 'validation', 'test'])
    parser.add_argument("--lambda_a", type=float, default=0.01)
    parser.add_argument("--lambda_b", type=float, default=0)
    parser.add_argument("--lambda_c", type=float, default=0.01)
    parser.add_argument("--run_name", type=str, default='MusicToken')
    parser.add_argument("--cosine_loss", type=_str2bool, default=True)
    parser.add_argument("--input_length", type=int, default=30)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--validation_batch_size", type=int, default=4)
    parser.add_argument("--lora", type=_str2bool, default=False)
    parser.add_argument("--revision", type=str, default=None, required=False)
    parser.add_argument("--use_precomputed_embeddings", type=_str2bool, default=True)
    parser.add_argument("--validate_every_n_epochs", type=int, default=5)
    parser.add_argument("--validate_every_n_steps", type=int, default=0)
    parser.add_argument("--use_precompute_vae_latents", type=_str2bool, default=True)
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

    # ── Early Fusion hyper-parameters ────────────────────────────────────────
    parser.add_argument("--ef_d_model", type=int, default=512)
    parser.add_argument("--ef_nhead", type=int, default=8)
    parser.add_argument("--ef_num_layers", type=int, default=4)
    parser.add_argument("--ef_dropout", type=float, default=0.1)
    parser.add_argument("--ef_n_audio_queries", type=int, default=1,
                        help="0=Full T_a (FuseLIP), 1=AttentivePooling (default), >1=Resampler")

    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.data_dir is None:
        raise ValueError("Specificare --data_dir.")

    args.image_latents_dir = args.latents_dir
    return args


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING MAIN
# ─────────────────────────────────────────────────────────────────────────────

def train_validation():
    args = parse_args()

    is_ddp, rank, local_rank, world_size = _setup_ddp()
    is_main = _is_main_process(rank)

    device = (
        torch.device(f"cuda:{local_rank}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    if is_main:
        _n_gpus = torch.cuda.device_count()
        print(f"\n{'='*60}")
        print(f"  Musipainter — FuseLIP / Early Fusion")
        print(f"  GPUs available    : {_n_gpus}")
        mode_str = f"DDP ({world_size}x GPU)" if is_ddp else "Single GPU"
        print(f"  Mode              : {mode_str}")
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
    Accelerator()  # compatibility import only

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'weights'), exist_ok=True)

    # ── Tokenizer + noise scheduler ──────────────────────────────────────────
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name, cache_dir=_hf_cache)
    else:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer",
            cache_dir=_hf_cache,
        )

    logger.info(
        "Tokenizer caricato — FuseLIP / Early Fusion mode: "
        "no <*> injection nel CLIP text encoder."
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler",
        cache_dir=_hf_cache,
    )

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * world_size
        )

    use_tb = (args.report_to != "none") and is_main
    tb_dir = os.path.join(args.output_dir, "runs")
    if use_tb:
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(os.path.join(tb_dir, f"music_to_image_{timestamp}"))
    else:
        writer = None

    # ── Datasets ─────────────────────────────────────────────────────────────
    if is_main:
        logger.info(
            f"Audio embeddings dir(s): {args.embeddings_dir}  "
            f"[preload_all={args.embeddings_preload_all}, "
            f"max_sf_handles={args.embeddings_max_sf_handles}]"
        )

    args.data_set = 'train'
    train_dataset = Museart(args=args, tokenizer=tokenizer, logger=logger)

    use_pin = (args.dataloader_num_workers > 0) and torch.cuda.is_available()

    if is_ddp:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank,
            shuffle=True, seed=args.seed,
        )
        train_shuffle = False
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
    steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_epochs * steps_per_epoch
        overrode_max = True

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MusicTokenWrapper(args)
    base_model = model

    # ── VAE precompute ────────────────────────────────────────────────────────
    train_latent_cache = {}
    valid_latent_cache = {}

    args.data_set = 'validation'
    validation_dataset = Museart(args=args, tokenizer=tokenizer, logger=logger)
    args.data_set = 'train'

    if args.use_precompute_vae_latents:
        if args.latents_dir:
            train_latent_cache = None
            valid_latent_cache = None
            if is_main:
                logger.info("[PREIMG] latents_dir fornita → LazyLatentIndex attivo.")
        else:
            if is_main:
                precompute_train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=args.train_batch_size * 2,
                    shuffle=False, num_workers=args.dataloader_num_workers,
                    pin_memory=use_pin,
                )
                precompute_valid_loader = torch.utils.data.DataLoader(
                    validation_dataset, batch_size=args.validation_batch_size * 2,
                    shuffle=False, num_workers=args.dataloader_num_workers,
                    pin_memory=use_pin,
                )
                train_latent_cache = precompute_vae_latents(
                    base_model.vae, precompute_train_loader, device,
                    use_amp=(args.mixed_precision == "fp16")
                )
                valid_latent_cache = precompute_vae_latents(
                    base_model.vae, precompute_valid_loader, device,
                    use_amp=(args.mixed_precision == "fp16")
                )
                del precompute_train_loader, precompute_valid_loader
                base_model.vae.to("cpu")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            _barrier(is_ddp)

    if args.gradient_checkpointing:
        try:
            base_model.unet.enable_gradient_checkpointing()
            if is_main:
                logger.info("UNet gradient checkpointing abilitato da flag CLI")
        except Exception as e:
            if is_main:
                logger.warning(f"Gradient checkpointing non disponibile: {e}")

    model = model.to(device)

    # torch.compile su EarlyFusionEncoder
    try:
        base_model.early_fusion = torch.compile(
            base_model.early_fusion, mode="default", fullgraph=False
        )
        base_model.embedder = base_model.early_fusion
        if is_main:
            logger.info("EarlyFusionEncoder compiled con torch.compile.")
    except Exception as e:
        if is_main:
            logger.info(f"torch.compile skipped: {e}")

    if is_ddp:
        _find_unused = args.lora
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=_find_unused,
            broadcast_buffers=False,
        )
        if is_main:
            logger.info(f"DDP wrapped: rank={rank}, world_size={world_size}")

    base_model = model.module if isinstance(model, DDP) else model

    # ── Optimiser ─────────────────────────────────────────────────────────────
    trainable_params = list(
        _unwrap_compiled(base_model.early_fusion).parameters()
    )
    if args.lora and base_model.lora_layers is not None:
        trainable_params += list(base_model.lora_layers.parameters())

    n_trainable = sum(p.numel() for p in trainable_params)
    if is_main:
        logger.info(
            f"Trainable parameters: {n_trainable:,} "
            f"(EarlyFusionEncoder{' + LoRA' if args.lora else ''})"
        )

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
        warnings.filterwarnings(
            "ignore", message=".*lr_scheduler.step.*before.*optimizer.step.*"
        )
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
            global_step, best_vloss, _resumed_best = load_checkpoint(
                args.resume_from_checkpoint,
                base_model.early_fusion,
                optimizer, scaler, lr_scheduler,
                base_model.lora_layers if args.lora else None,
                device,
            )
            best_model_path = _resumed_best
        if is_ddp:
            _gs = torch.tensor([global_step], dtype=torch.long, device=device)
            _bv = torch.tensor([best_vloss], dtype=torch.float64, device=device)
            dist.broadcast(_gs, src=0)
            dist.broadcast(_bv, src=0)
            global_step = _gs.item()
            best_vloss  = _bv.item()
            for param in _unwrap_compiled(base_model.early_fusion).parameters():
                dist.broadcast(param.data, src=0)

    if args.resume_from_embedder and not args.resume_from_checkpoint:
        if is_main:
            global_step, best_vloss = load_embedder_weights(
                weight_path=args.resume_from_embedder,
                embedder=base_model.early_fusion,
                device=device,
                resume_step=args.resume_step,
            )
        if is_ddp:
            _gs = torch.tensor([global_step], dtype=torch.long, device=device)
            dist.broadcast(_gs, src=0)
            global_step = _gs.item()
            for param in _unwrap_compiled(base_model.early_fusion).parameters():
                dist.broadcast(param.data, src=0)

    # ── Validation dataloader ─────────────────────────────────────────────────
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.validation_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        pin_memory=use_pin,
        persistent_workers=(args.dataloader_num_workers > 0),
        generator=torch.Generator(),
    )

    n_workers_loop = (
        args.workers_after_precompute if args.use_precompute_vae_latents
        else args.dataloader_num_workers
    )
    if args.use_precompute_vae_latents and n_workers_loop != args.dataloader_num_workers:
        if is_main:
            logger.info(
                f"Ricreazione train_dataloader: "
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
        steps_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps
        )

    n_gpus      = world_size
    total_batch = args.train_batch_size * args.gradient_accumulation_steps * n_gpus

    # ── Label embedding cache ─────────────────────────────────────────────────
    _label_to_idx = None
    _label_matrix = None
    if args.cosine_loss:
        all_labels = train_dataset.label + validation_dataset.label
        _, _label_to_idx, _label_matrix = build_label_embedding_cache(
            tokenizer,
            base_model.token_embedding,
            all_labels,
            device,
        )

    _prediction_type = noise_scheduler.config.prediction_type

    if is_main:
        logger.info("=" * 60)
        logger.info("START: TRAIN & VALIDATION — FuseLIP / Early Fusion")
        logger.info(f"timestamp           : {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"base_model          : {args.pretrained_model_name_or_path}")
        logger.info(f"train samples       : {len(train_dataset)}")
        logger.info(f"validation samples  : {len(validation_dataset)}")
        logger.info(f"train_batch_size    : {args.train_batch_size} (eff. {total_batch})")
        logger.info(f"gradient_accum_steps: {args.gradient_accumulation_steps}")
        logger.info(f"max_train_steps     : {args.max_train_steps}")
        logger.info(f"learning_rate       : {args.learning_rate:.2e}")
        logger.info(f"AMP fp16            : {'ON' if use_amp else 'OFF'}")
        logger.info(f"ef_d_model          : {args.ef_d_model}")
        logger.info(f"ef_nhead            : {args.ef_nhead}")
        logger.info(f"ef_num_layers       : {args.ef_num_layers}")
        logger.info(f"ef_dropout          : {args.ef_dropout}")
        logger.info(f"ef_n_audio_queries  : {getattr(args, 'ef_n_audio_queries', 1)}")
        logger.info(f"cosine_loss         : {args.cosine_loss}")
        logger.info(f"trainable params    : {n_trainable:,}")
        logger.info(f"embeddings_dir      : {args.embeddings_dir}")
        logger.info(f"embeddings_preload  : {args.embeddings_preload_all}")
        logger.info(f"latents_dir         : {args.latents_dir}")
        logger.info("=" * 60)

    # ── Helper: single validation pass ────────────────────────────────────────
    def _run_validation(v_loader, v_latent_cache):
        model.eval()
        running_vloss = 0.0
        n_batches = 0
        with torch.no_grad():
            for vb in v_loader:
                af   = vb["audio_features"].to(device, non_blocking=True)
                iids = vb["input_ids"].to(device, non_blocking=True)
                with autocast(enabled=use_amp):
                    is_pre = vb.get("is_precomputed_latent", None)
                    if is_pre is not None and bool(is_pre.all()):
                        lats = vb["pixel_values"].to(device, dtype=torch.float16, non_blocking=True)
                    elif v_latent_cache:
                        lats = torch.stack(
                            [v_latent_cache[iid] for iid in vb["image_id"]]
                        ).to(device, non_blocking=True)
                    else:
                        pv = vb["pixel_values"].to(device, non_blocking=True)
                        lats = (
                            base_model.vae.encode(pv.to(dtype=torch.float16))
                            .latent_dist.sample() * 0.18215
                        )
                    nv = torch.randn(lats.shape, dtype=torch.float32, device=device)
                    bv = lats.shape[0]
                    tv = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (bv,), device=device,
                    ).long()
                    nl  = noise_scheduler.add_noise(lats, nv, tv)
                    mp, fused_seq, audio_summary = model(af, iids, nl, tv)
                    fused_pooled  = fused_seq.mean(dim=1)
                    audio_pooled  = audio_summary.mean(dim=1)

                    tgt = (
                        nv if _prediction_type == "epsilon"
                        else noise_scheduler.get_velocity(lats, nv, tv).float()
                    )
                    lv = F.mse_loss(mp, tgt, reduction="mean")
                    _reg = args.lambda_a * torch.mean(torch.abs(audio_pooled))
                    if args.lambda_b > 0:
                        _reg = _reg + args.lambda_b * (torch.norm(audio_pooled, p=2, dim=1) ** 2).mean()
                    lv = lv + _reg

                    if args.cosine_loss and _label_to_idx is not None:
                        lbs   = vb['label']
                        ridxs = [_label_to_idx[l] for l in lbs if l in _label_to_idx]
                        aidxs = [j for j, l in enumerate(lbs) if l in _label_to_idx]
                        if ridxs:
                            idx_t = torch.tensor(ridxs, device=device)
                            ct    = _label_matrix.index_select(0, idx_t)
                            ae    = audio_pooled[aidxs]
                            em_n  = F.normalize(ae.float(), dim=1)
                            cs    = (em_n * ct.float()).sum(dim=1).mean()
                            lv    = lv + args.lambda_c * (1 - cs) ** 2
                running_vloss += lv.item()
                n_batches += 1
        return running_vloss / max(n_batches, 1)

    # ── Training loop ─────────────────────────────────────────────────────────
    epoch_number = global_step // steps_per_epoch if steps_per_epoch > 0 else 0
    resume_ckpt_paths: list = []
    _last_validated_step: int = -1
    _noise_buf = None
    _ts_buf    = None
    _step_start_time = None
    _steps_done_in_first_epoch = (
        global_step - (global_step // steps_per_epoch) * steps_per_epoch
        if steps_per_epoch > 0 else 0
    )
    batches_to_skip_first_epoch = (
        _steps_done_in_first_epoch * args.gradient_accumulation_steps
    )

    progress_bar = tqdm(
        range(global_step, args.max_train_steps), disable=not is_main
    )
    progress_bar.set_description("Steps")

    def _get_prefetch():
        return PrefetchLoader(train_dataloader, device)

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
            if base_model.lora_layers is not None:
                base_model.lora_layers.train()
        optimizer.zero_grad()

        _skip = batches_to_skip_first_epoch if epoch == epoch_number else 0
        _enumerated = enumerate(
            itertools.islice(_get_prefetch(), _skip, None), start=_skip
        )

        for i, batch in _enumerated:

            audio_features = batch["audio_features"]
            input_ids      = batch["input_ids"]

            # [CFG-DROPOUT] Dropout asincrono per CFG (Ho & Salimans 2022)
            _r = torch.rand(1).item()
            if _r < 0.07:
                audio_features = torch.zeros_like(audio_features)
            elif _r < 0.10:
                audio_features = torch.zeros_like(audio_features)
                input_ids = torch.full_like(input_ids, tokenizer.pad_token_id or 0)

            with autocast(enabled=use_amp):
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
                        latents = (
                            base_model.vae.encode(
                                pixel_values.to(dtype=torch.float16)
                            ).latent_dist.sample() * 0.18215
                        )

                bsz = latents.shape[0]
                if _noise_buf is None or _noise_buf.shape != latents.shape:
                    _noise_buf = torch.empty(latents.shape, dtype=torch.float32, device=device)
                    _ts_buf    = torch.empty((bsz,), dtype=torch.long, device=device)
                _noise_buf.normal_()
                _ts_buf.random_(0, noise_scheduler.config.num_train_timesteps)
                noise         = _noise_buf
                timesteps     = _ts_buf
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                if _prediction_type == "epsilon":
                    target = noise
                elif _prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps).float()
                else:
                    raise ValueError(f"Unknown prediction type: {_prediction_type}")

                model_pred, fused_seq, audio_summary = model(
                    audio_features, input_ids, noisy_latents, timesteps,
                )

                fused_pooled = fused_seq.mean(dim=1)
                audio_pooled = audio_summary.mean(dim=1)

                loss = F.mse_loss(model_pred, target, reduction="mean")

                _reg = args.lambda_a * torch.mean(torch.abs(audio_pooled))
                if args.lambda_b > 0:
                    _reg = _reg + args.lambda_b * (torch.norm(audio_pooled, p=2, dim=1) ** 2).mean()
                loss = loss + _reg

                if args.cosine_loss and _label_to_idx is not None:
                    labels   = batch['label']
                    row_idxs = [_label_to_idx[lbl] for lbl in labels if lbl in _label_to_idx]
                    aud_idxs = [j for j, lbl in enumerate(labels) if lbl in _label_to_idx]
                    if row_idxs:
                        idx_t     = torch.tensor(row_idxs, dtype=torch.long, device=device)
                        aud_idx_t = torch.tensor(aud_idxs, dtype=torch.long, device=device)
                        ct        = _label_matrix.index_select(0, idx_t)
                        at        = audio_pooled.index_select(0, aud_idx_t)
                        em_n      = F.normalize(at.float(), dim=1)
                        cs        = (em_n * ct.float()).sum(dim=1).mean()
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
                        _el = time.time() - _step_start_time
                        _sps = _el / 30
                        _eta_h = int(args.max_train_steps * _sps // 3600)
                        _eta_m = int((args.max_train_steps * _sps % 3600) // 60)
                        logger.info(
                            f"[PERF] Velocità reale (steps 10-40): {_sps:.2f}s/step  "
                            f"ETA: ~{_eta_h}h {_eta_m:02d}m"
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

            # ── Periodic save ────────────────────────────────────────────────
            if is_main and global_step % args.save_steps == 0 and global_step > 0:
                _ckpt_path = os.path.join(
                    args.output_dir,
                    f"weights/checkpoint_step{global_step}.pt"
                )
                if _ckpt_path not in resume_ckpt_paths:
                    save_progress(
                        base_model.early_fusion,
                        os.path.join(
                            args.output_dir,
                            f"weights/{args.run_name}_early_fusion-step{global_step}.safetensors"
                        )
                    )
                    if args.lora and base_model.lora_layers is not None:
                        save_progress(
                            base_model.lora_layers,
                            os.path.join(
                                args.output_dir,
                                f"weights/{args.run_name}_lora-step{global_step}.safetensors"
                            )
                        )
                    save_checkpoint(
                        embedder=base_model.early_fusion,
                        optimizer=optimizer,
                        scaler=scaler,
                        lr_scheduler=lr_scheduler,
                        global_step=global_step,
                        best_vloss=best_vloss,
                        lora_layers=base_model.lora_layers if args.lora else None,
                        save_path=_ckpt_path,
                        best_model_path=best_model_path,
                    )
                    resume_ckpt_paths.append(_ckpt_path)
                    _rotate_checkpoints(resume_ckpt_paths, args.keep_last_n_checkpoints)
                    _check_disk_space(args.output_dir)

            _barrier(is_ddp)

            # ── Mid-epoch validation ─────────────────────────────────────────
            if (
                args.validate_every_n_steps > 0
                and global_step % args.validate_every_n_steps == 0
                and global_step > 0
                and global_step < args.max_train_steps
                and _last_validated_step != global_step
            ):
                if is_main:
                    _avg_t = running_train_loss.item() / max(loss_count, 1)
                    _avg_v = _run_validation(validation_dataloader, valid_latent_cache)
                    print(
                        f'[MID-EPOCH step={global_step}]  '
                        f'train={_avg_t:.4f}  valid={_avg_v:.4f}'
                    )
                    if _avg_v < best_vloss:
                        best_vloss = _avg_v
                        _nb = os.path.join(
                            args.output_dir,
                            f'best_model_early_fusion_{timestamp}.safetensors'
                        )
                        _nb_tmp = _nb + ".tmp"
                        from modules.preprocess.utils import save_safetensors as _sf_save
                        _sf_save(
                            _unwrap_compiled(base_model.early_fusion).state_dict(),
                            _nb_tmp
                        )
                        if best_model_path and os.path.exists(best_model_path) \
                                and best_model_path != _nb:
                            os.remove(best_model_path)
                        os.replace(_nb_tmp, _nb)
                        if args.lora and base_model.lora_layers is not None:
                            _nl = _nb.replace('best_model_early_fusion_', 'best_model_lora_')
                            _nl_tmp = _nl + ".tmp"
                            _sf_save(base_model.lora_layers.state_dict(), _nl_tmp)
                            os.replace(_nl_tmp, _nl)
                        best_model_path = _nb
                        logger.info(
                            f"Nuovo best (mid-epoch step={global_step}): "
                            f"vloss={best_vloss:.4f}"
                        )
                    model.train()
                    base_model.unet.eval()
                    base_model.vae.eval()
                    if args.lora and base_model.lora_layers is not None:
                        base_model.lora_layers.train()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                _barrier(is_ddp)
                _last_validated_step = global_step

            if global_step >= args.max_train_steps:
                break

        avg_loss_train = running_train_loss.item() / max(loss_count, 1)
        batches_to_skip_first_epoch = 0

        # ── End-of-epoch validation ───────────────────────────────────────────
        do_val = (
            (args.validate_every_n_steps > 0
             and global_step % args.validate_every_n_steps == 0
             and global_step > 0)
            or (args.validate_every_n_steps == 0
                and (epoch + 1) % args.validate_every_n_epochs == 0)
            or (global_step >= args.max_train_steps)
        )
        if _last_validated_step == global_step:
            do_val = False

        if do_val:
            if is_main:
                if use_tb and writer is not None and args.max_tb_size_mb > 0:
                    writer = _check_tb_size(writer, tb_dir, args.max_tb_size_mb)
                avg_valid = _run_validation(validation_dataloader, valid_latent_cache)
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
                    _nb = os.path.join(
                        args.output_dir,
                        f'best_model_early_fusion_{timestamp}.safetensors'
                    )
                    _nb_tmp = _nb + ".tmp"
                    from modules.preprocess.utils import save_safetensors as _sf_save_ep
                    _sf_save_ep(
                        _unwrap_compiled(base_model.early_fusion).state_dict(),
                        _nb_tmp
                    )
                    if best_model_path and os.path.exists(best_model_path) \
                            and best_model_path != _nb:
                        os.remove(best_model_path)
                    os.replace(_nb_tmp, _nb)
                    if args.lora and base_model.lora_layers is not None:
                        _nl = _nb.replace('best_model_early_fusion_', 'best_model_lora_')
                        _nl_tmp = _nl + ".tmp"
                        _sf_save_ep(base_model.lora_layers.state_dict(), _nl_tmp)
                        os.replace(_nl_tmp, _nl)
                    best_model_path = _nb
                    logger.info(
                        f"  Nuovo best early_fusion: {_nb} (vloss={best_vloss:.4f})"
                    )
                model.train()
                base_model.unet.eval()
                base_model.vae.eval()
                if args.lora and base_model.lora_layers is not None:
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

    # ── Final save ────────────────────────────────────────────────────────────
    if is_main:
        save_progress(
            base_model.early_fusion,
            os.path.join(args.output_dir, "learned_embeds.safetensors")
        )
        if args.lora and base_model.lora_layers is not None:
            save_progress(
                base_model.lora_layers,
                os.path.join(args.output_dir, "learned_embeds_lora_layers.safetensors")
            )
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETATO CON SUCCESSO")
        logger.info(f"total steps         : {global_step}/{args.max_train_steps}")
        logger.info(f"best validation loss: {best_vloss:.4f}")
        logger.info(f"best model          : {best_model_path or 'N/A'}")
        logger.info(f"output_dir          : {args.output_dir}")
        logger.info("=" * 60)
        if use_tb and writer is not None:
            writer.close()

    _barrier(is_ddp)
    _teardown_ddp(is_ddp)
    return best_vloss


if __name__ == "__main__":
    train_validation()
