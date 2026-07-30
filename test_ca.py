# @title test_ca.py
"""
test_ca.py — Inference script for Musipainter (Audio-Guided Cross-Attention branch).

NOTE: rinominato da test_no_accel_colab.py per convivere con la variante
Early-Fusion (test_ef.py) nello stesso repository, selezionabile tramite
musipainter_test.py --architecture Musipainter-CA.
Unica modifica rispetto all'originale: l'import di MusicTokenWrapper punta
al modulo rinominato modules.MusicToken.MusicToken_no_accel_ca.
Nessun'altra logica è stata alterata.
"""

import argparse
import logging
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import datasets
import diffusers
import transformers
from accelerate import Accelerator
from diffusers.utils import check_min_version
from transformers import CLIPTokenizer
from pathlib import Path

from dataloader_colab import Museart
from modules.MusicToken.MusicToken_no_accel_ca import MusicTokenWrapper

check_min_version("0.12.0")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
#  ARG PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    def _str2bool(v):
        if isinstance(v, bool): return v
        if v.lower() in ('yes', 'true', '1'): return True
        if v.lower() in ('no', 'false', '0'): return False
        raise argparse.ArgumentTypeError(f"Boolean expected, got: '{v}'")

    parser = argparse.ArgumentParser()

    from modules.preprocess.argparse_multiembedding import add_multiembedding_args
    add_multiembedding_args(parser)

    parser.add_argument("--learned_embeds",      type=str, default='./output/learned_embeds.safetensors')
    parser.add_argument("--learned_vae",         type=str, default='./output/vae_learned_embeds.bin')
    parser.add_argument("--learned_aud_encoder", type=str, default='./output/aud_encoder_learned_embeds.bin')
    parser.add_argument("--learned_unet",        type=str, default='./output/unet_learned_embeds.bin')
    parser.add_argument("--learned_embeds_lora", type=str, default='./output/learned_embeds_lora_layers.safetensors')
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='stabilityai/stable-diffusion-2')
    parser.add_argument("--revision",            type=str, default=None)
    parser.add_argument("--tokenizer_name",      type=str, default=None)
    parser.add_argument("--data_dir",            type=str, default="./Museart/")
    parser.add_argument("--latents_dir",         type=str, default="./image_latents/")
    parser.add_argument("--use_precomputed_embeddings", type=_str2bool, default=True)
    parser.add_argument("--placeholder_token",   type=str, default="<*>")
    parser.add_argument("--output_dir",          type=str, default="./output/test/")
    parser.add_argument("--seed",                type=int, default=1234)
    parser.add_argument("--resolution",          type=int, default=512)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--mixed_precision",     type=str, default="fp16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow_tf32",          action="store_true")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--data_set",            type=str, default='test',
                        choices=['train', 'validation', 'test'])
    parser.add_argument("--generation_steps",    type=int, default=20)
    parser.add_argument("--run_name",            type=str, default='MusicToken')
    parser.add_argument("--prompt",              type=str, default='An art image of <*>')
    parser.add_argument("--input_length",        type=int, default=30)
    parser.add_argument("--lora",                type=_str2bool, default=False)
    parser.add_argument("--aud_encoder",         type=_str2bool, default=False)
    parser.add_argument("--unet",                type=_str2bool, default=False)
    parser.add_argument("--vae",                 type=_str2bool, default=False)
    parser.add_argument("--guidance_scale",      type=float, default=7.5)
    parser.add_argument("--center_crop",         action="store_true", default=False)
    parser.add_argument("--hf_cache_dir",        type=str, default="/tmp/hf_model_cache")
    parser.add_argument("--ef_d_model",          type=int, default=512)
    parser.add_argument("--ef_nhead",            type=int, default=8)
    parser.add_argument("--ef_num_layers",       type=int, default=4)
    parser.add_argument("--ef_dropout",          type=float, default=0.1)
    parser.add_argument("--report_to",           type=str, default="tensorboard")
    parser.add_argument("--logging_dir",         type=str, default="logs")
    parser.add_argument("--set_size",            type=str, default='full')

    parser.add_argument(
        "--uncond_mode", type=str, default="zeros",
        choices=["zeros", "text_only"],
        help=(
            "'zeros' (default/recommended): uncond = silent audio + empty text → p(x). "
            "Matches the ~3%% full-zero CFG dropout used during training. "
            "'text_only': uncond = silent audio + text prompt → p(x|text). "
            "Out-of-distribution for the current training schema."
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1:
        args.local_rank = env_local_rank
    if args.data_dir is None:
        raise ValueError("Specify --data_dir.")
    args.image_latents_dir = args.latents_dir
    return args


# ─────────────────────────────────────────────────────────────────────────────
#  CHECKPOINT RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_checkpoint_path(explicit_path, output_dir, stem, label):
    """
    Prefer best_model (lowest validation loss) over learned_embeds
    (final checkpoint, potentially overfit).

    Search order:
      1. explicit_path (if file exists on disk)
      2. best_model_<label>_*.safetensors  ← FIRST (lowest vloss)
      3. <output_dir>/<stem>.safetensors   ← fallback (may be overfit)
      4. weights/ step checkpoints
    """
    import glob

    for c in [explicit_path,
              explicit_path.replace(".bin", ".safetensors") if explicit_path.endswith(".bin") else None]:
        if c and os.path.exists(c):
            logger.info(f"[CKPT] Using explicit path: {c}")
            return c

    best_patterns = [
        os.path.join(output_dir, f"best_model_{label}_*.safetensors"),
        os.path.join(output_dir, f"best_model_audio_guided_cross_attn_*.safetensors"),
        os.path.join(output_dir, f"best_model_{label}_*.bin"),
    ]
    for pat in best_patterns:
        hits = sorted(glob.glob(pat), key=os.path.getmtime, reverse=True)
        if hits:
            logger.info(
                f"[CKPT] Found best_model checkpoint (lowest vloss): {hits[0]}"
            )
            return hits[0]

    for c in [os.path.join(output_dir, f"{stem}.safetensors"),
              os.path.join(output_dir, f"{stem}.bin")]:
        if os.path.exists(c):
            logger.warning(
                f"[CKPT] No best_model found — using final checkpoint: {c}. "
                "WARNING: this may be overfit. Pass --learned_embeds <best_model_path> "
                "to use an earlier checkpoint explicitly."
            )
            return c

    step_patterns = [
        os.path.join(output_dir, "weights", f"*_{label}-step*.safetensors"),
        os.path.join(output_dir, "weights", f"*_audio_guided_cross_attn-step*.safetensors"),
    ]
    for pat in step_patterns:
        hits = sorted(glob.glob(pat), key=os.path.getmtime, reverse=True)
        if hits:
            logger.info(f"[CKPT] Using step checkpoint: {hits[0]}")
            return hits[0]

    raise FileNotFoundError(
        f"[CKPT] No checkpoint found for '{label}'. "
        f"Explicit path '{explicit_path}' does not exist and no patterns matched in '{output_dir}'."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _probe_audio_frame_count(dataset) -> int:
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch  = next(iter(loader))
    t_a    = batch["audio_features"].shape[1]
    logger.info(
        f"[CFG] Probed T_a={t_a} from first sample "
        f"(shape={list(batch['audio_features'].shape)}). "
        f"Resampler path: {'adaptive_avg_pool1d (deterministic)' if t_a <= 150 else 'Perceiver learned queries'}."
    )
    if t_a > 200:
        logger.warning(
            f"[CFG] T_a={t_a} is large — embeddings may be stride=1 (no pooling). "
            "Training used stride=4 (T_a≈94). Ensure test embeddings_dir matches training."
        )
    return t_a


def _build_uncond_embedding(args, base_model, tokenizer, t_a, weight_dtype, device):
    """
    Build unconditional embedding for CFG → [1, 77, output_size].
    """
    audio_dim    = 768 * 3
    silent_audio = torch.zeros(1, t_a, audio_dim, dtype=torch.float32, device=device)

    with torch.no_grad():
        if args.uncond_mode == "zeros":
            uncond_ids = tokenizer(
                [""], padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            ).input_ids.to(device)
            uncond_text = base_model._get_text_embeddings(uncond_ids)
        else:  # text_only
            text_prompt = args.prompt.replace(
                getattr(args, 'placeholder_token', '<*>'), ""
            ).strip()
            uncond_ids = tokenizer(
                [text_prompt], padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            ).input_ids.to(device)
            uncond_text = base_model._get_text_embeddings(uncond_ids)

        fused_seq = base_model.early_fusion(
            audio_tokens=silent_audio,
            text_tokens=uncond_text.float(),
            return_audio_summary=False,
        )
        uncond_emb = base_model._resample_to_77(fused_seq)   # [1, 77, D]

    logger.info(
        f"[CFG] uncond_mode={args.uncond_mode} | "
        f"silent_audio=[1,{t_a},{audio_dim}] | "
        f"uncond_embeddings={list(uncond_emb.shape)}"
    )
    return uncond_emb.to(weight_dtype)


# ─────────────────────────────────────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def inference(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "imgs", args.run_name), exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO, force=True,
    )

    _t_start  = time.time()
    _n_gpus   = torch.cuda.device_count()
    _gpu_info = (", ".join(torch.cuda.get_device_name(i) for i in range(_n_gpus))
                 if _n_gpus > 0 else "CPU")

    logger.info("=" * 60)
    logger.info("START: INFERENCE [FIXED] — Musipainter")
    logger.info(f"timestamp             : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"GPU                   : {_n_gpus}  ({_gpu_info})")
    logger.info(f"base_model            : {args.pretrained_model_name_or_path}")
    logger.info(f"resolution            : {args.resolution}x{args.resolution}")
    logger.info(f"num_inference_steps   : {args.num_inference_steps}")
    logger.info(f"generation_steps      : {args.generation_steps}")
    logger.info(f"guidance_scale        : {args.guidance_scale}")
    logger.info(f"uncond_mode           : {args.uncond_mode}")
    logger.info(f"lora                  : {args.lora}")
    logger.info(f"embeddings_dir        : {args.embeddings_dir}")
    logger.info("=" * 60)

    args.learned_embeds = _resolve_checkpoint_path(
        explicit_path=args.learned_embeds,
        output_dir=args.output_dir,
        stem="learned_embeds",
        label="audio_guided_cross_attn",
    )
    logger.info(f"[CKPT] learned_embeds → {args.learned_embeds}")

    if args.lora:
        args.learned_embeds_lora = _resolve_checkpoint_path(
            explicit_path=args.learned_embeds_lora,
            output_dir=args.output_dir,
            stem="learned_embeds_lora_layers",
            label="lora",
        )
        logger.info(f"[CKPT] learned_embeds_lora → {args.learned_embeds_lora}")

    # Tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    else:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer"
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset
    test_dataset = Museart(args=args, tokenizer=tokenizer, logger=logger, size=args.resolution)

    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(
        args.mixed_precision, torch.float32
    )

    # Model
    model      = MusicTokenWrapper(args).eval().to(device)
    base_model = model

    # Scheduler
    from diffusers import EulerDiscreteScheduler
    scheduler = EulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler",
        cache_dir=args.hf_cache_dir,
    )
    logger.info(
        f"[SCHEDULER] {scheduler.__class__.__name__} | "
        f"prediction_type={scheduler.config.get('prediction_type', 'N/A')}"
    )

    t_a = _probe_audio_frame_count(test_dataset)

    uncond_embeddings = _build_uncond_embedding(
        args, base_model, tokenizer, t_a, weight_dtype, device
    )   # [1, 77, 1024]

    cond_input_ids = tokenizer(
        [args.prompt], padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    ).input_ids.to(device)

    gen = torch.Generator(device="cpu")
    if args.seed is not None:
        gen.manual_seed(args.seed)
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=True,
        num_workers=args.dataloader_num_workers, generator=gen,
    )

    for step, batch in enumerate(test_dataloader):
        if step >= args.generation_steps:
            break

        scheduler.set_timesteps(args.num_inference_steps)

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        aud_features = batch["audio_features"].to(dtype=torch.float32)

        if aud_features.shape[1] != t_a:
            logger.warning(
                f"[CFG] T_a mismatch at step {step}: "
                f"expected {t_a} but got {aud_features.shape[1]}. "
                "Skipping sample — check embeddings_dir for mixed strides."
            )
            continue

        with torch.no_grad():
            text_tokens = base_model._get_text_embeddings(cond_input_ids)  # [1, 77, 1024]

            fused_seq = base_model.early_fusion(
                audio_tokens=aud_features,
                text_tokens=text_tokens,
                return_audio_summary=False,
            )                                               # [1, T_a, 1024]

            cond_embeddings = base_model._resample_to_77(fused_seq).to(weight_dtype)  # [1, 77, 1024]

            text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

            seed = random.randint(0, 10000)
            generator = torch.Generator(device=device).manual_seed(seed)
            latents = torch.randn(
                (1, base_model.unet.config.in_channels,
                 args.resolution // 8, args.resolution // 8),
                generator=generator, device=device, dtype=weight_dtype,
            ) * scheduler.init_noise_sigma

            for t in scheduler.timesteps:
                latent_model_input = scheduler.scale_model_input(
                    torch.cat([latents] * 2), t
                )
                noise_pred = base_model.unet(
                    latent_model_input, t,
                    encoder_hidden_states=text_embeddings,
                ).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + args.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            _sf = base_model.vae.config.scaling_factor
            latents = latents / _sf
            base_model.vae.to(dtype=torch.float32)
            image_tensor = base_model.vae.decode(
                latents.to(base_model.vae.device).float()
            ).sample
            base_model.vae.to(dtype=weight_dtype)

            image_np = (
                ((image_tensor / 2 + 0.5).clamp(0, 1)
                 .cpu().permute(0, 2, 3, 1).float().numpy()[0]) * 255
            ).round().astype("uint8")

        from PIL import Image
        save_name = f'{batch["aud_id"][0]}_{batch["image_id"][0]}_{batch["label"][0]}.jpg'
        Image.fromarray(image_np).save(
            os.path.join(args.output_dir, "imgs", args.run_name, save_name)
        )
        logger.info(f"Saved: {save_name} ({step + 1}/{args.generation_steps})")

    _total = time.time() - _t_start
    n_gen  = min(step + 1, args.generation_steps)
    logger.info("=" * 60)
    logger.info("COMPLETED: INFERENCE [FIXED]")
    logger.info(f"total time          : {_total:.1f}s")
    logger.info(f"images generated    : {n_gen}")
    logger.info(f"avg time/image      : {_total / max(n_gen, 1):.1f}s")
    logger.info(f"output_dir          : {os.path.join(args.output_dir, 'imgs', args.run_name)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    inference(args)
