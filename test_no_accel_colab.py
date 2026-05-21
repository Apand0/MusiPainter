# @title test_no_accel_colab.py
"""Inference script for Musipainter."""

import argparse
import logging
import os
import random
import time

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
import datasets
import diffusers
import transformers
from accelerate import Accelerator
from diffusers import DDPMScheduler
from diffusers.utils import check_min_version
from transformers import CLIPTokenizer
from pathlib import Path

from dataloader_colab import Museart
from modules.MusicToken.MusicToken_no_accel import MusicTokenWrapper

check_min_version("0.12.0")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

#  ARG PARSING

def parse_args():
    def _str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', '1'):
            return True
        if v.lower() in ('no', 'false', '0'):
            return False
        raise argparse.ArgumentTypeError(f"Valore booleano atteso, ricevuto: '{v}'")
    parser = argparse.ArgumentParser(description="Testing script con pre-encoded embeddings")

    from modules.preprocess.argparse_multiembedding_patch import add_multiembedding_args
    add_multiembedding_args(parser)

    parser.add_argument("--learned_embeds", type=str, 
                        default='./output/learned_embeds.safetensors')
    parser.add_argument("--learned_vae", type=str,
                        default='./output/vae_learned_embeds.bin')
    parser.add_argument("--learned_aud_encoder", type=str,
                        default='./output/aud_encoder_learned_embeds.bin')
    parser.add_argument("--learned_unet", type=str,
                        default='./output/unet_learned_embeds.bin')
    parser.add_argument("--learned_embeds_lora", type=str,
                        default='./output/learned_embeds_lora_layers.safetensors')
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default='stabilityai/stable-diffusion-2')
    parser.add_argument("--revision", type=str, default=None, required=False)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="./Museart/")
    parser.add_argument("--embeddings_dir", type=str, default="./audio_embeddings/")
    parser.add_argument("--latents_dir", type=str, default="./image_latents/")
    parser.add_argument("--use_precomputed_embeddings", type=_str2bool, default=True)
    parser.add_argument("--placeholder_token", type=str, default="<*>")
    parser.add_argument("--output_dir", type=str, default="./output/test/")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--data_set", type=str, default='test',
                        choices=['train', 'validation', 'test'])
    parser.add_argument("--generation_steps", type=int, default=50)
    parser.add_argument("--run_name", type=str, default='MusicToken')
    parser.add_argument("--set_size", type=str, default='full')
    parser.add_argument("--prompt", type=str,
                        default='An art image of <*>')
    parser.add_argument("--input_length", type=int, default=30)
    parser.add_argument("--lora", type=_str2bool, default=False)
    parser.add_argument("--aud_encoder", type=_str2bool, default=False)
    parser.add_argument("--unet", type=_str2bool, default=False)
    parser.add_argument("--vae", type=_str2bool, default=False)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--center_crop", action="store_true", default=False)
    parser.add_argument("--hf_cache_dir", type=str, default="/tmp/hf_model_cache")
    parser.add_argument("--uncond_mode", type=str, default="zeros",
                        choices=["zeros", "text_only"],
                        help=(
                            "'zeros': unconditional embedding is all-zeros "
                            "(fastest). 'text_only': unconditional embedding "
                            "is the fused representation of the text prompt "
                            "with a silent (zero) audio feature."
                        ))

    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1:
        args.local_rank = env_local_rank

    if args.data_dir is None:
        raise ValueError("Specify --data_dir.")

    args.image_latents_dir = args.latents_dir
    return args


def _resolve_checkpoint_path(explicit_path, output_dir, stem, label):
    """Resolve best available checkpoint, supporting .bin → .safetensors fallback."""
    import glob
    candidates = [
        explicit_path,
        explicit_path.replace(".bin", ".safetensors")
            if explicit_path.endswith(".bin") else None,
        os.path.join(output_dir, f"{stem}.safetensors"),
        os.path.join(output_dir, f"{stem}.bin"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    glob_patterns = [
        os.path.join(output_dir, f"best_model_{label}_*.safetensors"),
        os.path.join(output_dir, f"best_model_{label}_*.bin"),
        os.path.join(output_dir, f"best_model_early_fusion_*.safetensors"),
        os.path.join(output_dir, "weights", f"*_{label}-step*.safetensors"),
        os.path.join(output_dir, "weights", f"*_early_fusion-step*.safetensors"),
        os.path.join(output_dir, "weights", f"*_{label}-step*.bin"),
    ]
    for pat in glob_patterns:
        hits = sorted(glob.glob(pat), key=os.path.getmtime, reverse=True)
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"[AUTO-CHECKPOINT] No checkpoint found for '{label}'. "
        f"Searched: {explicit_path} and glob patterns in {output_dir}."
    )


def inference(args):
    """Run Early Fusion inference loop and save generated images."""

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "imgs"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "imgs", args.run_name), exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        force=True,
    )
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()

    _t_start   = time.time()
    _n_gpus    = torch.cuda.device_count()
    _gpu_info  = (
        ", ".join(torch.cuda.get_device_name(i) for i in range(_n_gpus))
        if _n_gpus > 0 else "CPU"
    )

    logger.info("=" * 60)
    logger.info("START: INFERENCE")
    logger.info(f"timestamp           : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"GPU                 : {_n_gpus}  ({_gpu_info})")
    logger.info(f"base_model          : {args.pretrained_model_name_or_path}")
    logger.info(f"resolution          : {args.resolution}x{args.resolution}")
    logger.info(f"num_inference_steps : {args.num_inference_steps}")
    logger.info(f"generation_steps    : {args.generation_steps}")
    logger.info(f"guidance_scale      : {args.guidance_scale}")
    logger.info(f"uncond_mode         : {args.uncond_mode}")
    logger.info(f"prompt_template     : '{args.prompt}'")
    logger.info(f"learned_embeds      : {args.learned_embeds}")
    logger.info("=" * 60)

    # ── Auto-resolve checkpoint ──────────────────────────────────────────────
    args.learned_embeds = _resolve_checkpoint_path(
        explicit_path=args.learned_embeds,
        output_dir=args.output_dir,
        stem="learned_embeds",
        label="early_fusion",
    )
    logger.info(f"[AUTO-CHECKPOINT] Resolved: {args.learned_embeds}")
    if args.lora:
        args.learned_embeds_lora = _resolve_checkpoint_path(
            explicit_path=args.learned_embeds_lora,
            output_dir=args.output_dir,
            stem="learned_embeds_lora_layers",
            label="lora",
        )
        logger.info(f"[AUTO-CHECKPOINT] LoRA: {args.learned_embeds_lora}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    else:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer"
        )
    # and fed to the CLIP token embedding table inside MusicTokenWrapper.

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # ── Dataset ───────────────────────────────────────────────────────────────
    test_dataset = Museart(
        args=args, tokenizer=tokenizer, logger=logger, size=args.resolution,
    )
    dataloader_generator = torch.Generator(device="cpu")
    if args.seed is not None:
        dataloader_generator.manual_seed(args.seed)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True,
        num_workers=args.dataloader_num_workers,
        generator=dataloader_generator,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    weight_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(args.mixed_precision, torch.float32)

    model = MusicTokenWrapper(args).to(weight_dtype).eval().to(device)
    base_model = model


    # ── Scheduler (EulerDiscrete, v-prediction safe for SD 2.1) ───────────────
    from diffusers import EulerDiscreteScheduler
    scheduler = EulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        cache_dir=args.hf_cache_dir,
    )
    logger.info(
        f"[SCHEDULER] {scheduler.__class__.__name__} | "
        f"prediction_type={scheduler.config.get('prediction_type', 'N/A')}"
    )

    # ── Unconditional embedding for CFG ───────────────────────────────────────
    #
    # "zeros": uncond = zero vector of shape [1, 1, output_size].
    #   Fast and clean; the audio+text signal is amplified directly.
    #
    # "text_only": uncond = EarlyFusionEncoder(silent_audio, text_tokens).
    #   More semantically correct: CFG amplifies the *audio* contribution
    #   above the text-only baseline, which mirrors how FuseLIP operates.

    output_size = base_model.unet.config.cross_attention_dim
    with torch.no_grad():
        if args.uncond_mode == "zeros":
            uncond_embeddings = torch.zeros(
                1, 1, output_size, dtype=weight_dtype, device=device
            )
            logger.info("[CFG] uncond_mode=zeros: using zero vector.")
        else:
            # text_only: encode the prompt with silent (zero) audio features.
            # This lets CFG amplify exactly the audio contribution.
            _text_only_prompt = args.prompt.replace(args.placeholder_token, "").strip()
            _uncond_ids = tokenizer(
                [_text_only_prompt],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            # Determine audio temporal length from a dummy forward
            _T_a = 376  # 30s × 16kHz / 160 / 8 (same as EarlyFusionEncoder default)
            _silent_audio = torch.zeros(
                1, _T_a, 768 * 3, dtype=torch.float32, device=device
            )
            _text_tokens = base_model._get_text_embeddings(_uncond_ids)
            _uncond_fused = base_model.early_fusion(
                audio_tokens=_silent_audio,
                text_tokens=_text_tokens.float(),
            )  # [1, output_size]
            uncond_embeddings = _uncond_fused.unsqueeze(1).to(dtype=weight_dtype)
            # [1, 1, output_size]
            logger.info(
                f"[CFG] uncond_mode=text_only: "
                f"prompt='{_text_only_prompt}' + silent audio."
            )

    # ── Tokenise the conditional prompt (constant across images) ─────────────
    cond_input_ids = tokenizer(
        [args.prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)

    # ── Generation loop ───────────────────────────────────────────────────────
    for step, batch in enumerate(test_dataloader):
        if step >= args.generation_steps:
            break

        scheduler.set_timesteps(args.num_inference_steps)

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)

        aud_features = batch["audio_features"].to(dtype=weight_dtype)

        with torch.no_grad():
            # ── Conditional embedding (audio + text fused) ────────────────────
            # It internally: projects audio + text → shared Transformer → pool →
            # output_proj → [B, output_size].
            audio_feats_f32 = aud_features.float()
            text_tokens_f32 = base_model._get_text_embeddings(cond_input_ids)

            cond_fused = base_model.early_fusion(
                audio_tokens=audio_feats_f32,
                text_tokens=text_tokens_f32,
            )  # [1, output_size]
            cond_embeddings = cond_fused.unsqueeze(1).to(dtype=weight_dtype)
            # [1, 1, output_size]

            # CFG: stack uncond + cond
            text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])
            # [2, 1, output_size]

            # ── Latent diffusion ──────────────────────────────────────────────
            seed = random.randint(0, 10000)
            generator = torch.Generator(device=device).manual_seed(seed)
            latents = torch.randn(
                (1, base_model.unet.config.in_channels,
                 args.resolution // 8, args.resolution // 8),
                generator=generator,
                device=device,
                dtype=weight_dtype,
            )
            latents = latents * scheduler.init_noise_sigma

            for t in scheduler.timesteps:
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = scheduler.scale_model_input(
                    latent_model_input, t
                )
                noise_pred = base_model.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                ).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + args.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            # ── VAE decode ────────────────────────────────────────────────────
            _sf = base_model.vae.config.scaling_factor  # 0.18215 for SD 2.1
            latents = latents / _sf
            base_model.vae.to(dtype=torch.float32)
            latents_f32 = latents.to(base_model.vae.device).float()
            image_tensor = base_model.vae.decode(latents_f32).sample
            base_model.vae.to(dtype=weight_dtype)

            image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
            image_tensor = image_tensor.cpu().permute(0, 2, 3, 1).float().numpy()
            image_np = (image_tensor[0] * 255).round().astype("uint8")

        from PIL import Image
        image = Image.fromarray(image_np)
        save_name = (
            f'{batch["aud_id"][0]}_{batch["image_id"][0]}_{batch["label"][0]}.jpg'
        )
        image.save(
            os.path.join(args.output_dir, "imgs", args.run_name, save_name)
        )
        logger.info(f"Saved: {save_name} ({step + 1}/{args.generation_steps})")

    _total = time.time() - _t_start
    n_gen  = min(step + 1, args.generation_steps)
    logger.info("=" * 60)
    logger.info("COMPLETED: INFERENCE")
    logger.info(f"total time          : {_total:.1f}s")
    logger.info(f"images generated    : {n_gen}")
    logger.info(f"avg time/image      : {_total / max(n_gen, 1):.1f}s")
    logger.info(
        f"output_dir          : "
        f"{os.path.join(args.output_dir, 'imgs', args.run_name)}"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    inference(args)
