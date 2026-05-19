# @title test_no_accel_colab.py
"""
Inference script for Musipainter.
Generates images from audio embeddings using a frozen SD pipeline.
"""

import argparse
import logging
import os
import random
import time

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.utils.data import Dataset
import datasets
import diffusers
import transformers
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from diffusers.utils import check_min_version
from transformers import CLIPTokenizer
from pathlib import Path

from dataloader_colab import Museart
from modules.MusicToken.MusicToken_no_accel import MusicTokenWrapper

check_min_version("0.12.0")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


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
    parser = argparse.ArgumentParser(description="Testing script con pre-encoded embeddings")
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
    parser.add_argument("--guidance_scale", type=float, default=4.0,
                        help="CFG scale. 4.0-5.0 consigliato per embedder appena addestrato.")
    parser.add_argument("--center_crop", action="store_true", default=False)
    parser.add_argument("--hf_cache_dir", type=str, default="/tmp/hf_model_cache")

    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1:
        args.local_rank = env_local_rank

    if args.data_dir is None:
        raise ValueError("You must specify a data directory.")
        
    args.image_latents_dir = args.latents_dir

    return args


def _resolve_checkpoint_path(explicit_path: str, output_dir: str,
                               stem: str, label: str) -> str:
    """
    Resolve the best available checkpoint path for a given weight file.

    Search order:
      1. The explicit path as provided by the user.
      2. The .safetensors counterpart if a .bin path was given (format migration).
      3. <output_dir>/<stem>.safetensors
      4. <output_dir>/<stem>.bin
      5. Most recent best_model_<label>_*.safetensors in output_dir.
      6. Most recent best_model_<label>_*.bin in output_dir.
      7. Most recent weights/<run_name>_<label>-step*.safetensors in output_dir.
      8. Most recent weights/<run_name>_<label>-step*.bin in output_dir.

    Args:
        explicit_path: value of --learned_embeds / --learned_embeds_lora.
        output_dir:    value of --output_dir (parent of weights/).
        stem:          base filename without extension (e.g. "learned_embeds").
        label:         used in glob patterns ("embedder" or "lora").

    Returns:
        Absolute path to the first candidate that exists on disk.

    Raises:
        FileNotFoundError if none of the candidates is found.
    """
    import glob

    candidates = [
        explicit_path,
        # Transparent .bin → .safetensors upgrade for callers using old paths.
        explicit_path.replace(".bin", ".safetensors") if explicit_path.endswith(".bin") else None,
        os.path.join(output_dir, f"{stem}.safetensors"),
        os.path.join(output_dir, f"{stem}.bin"),
    ]

    for c in candidates:
        if c and os.path.exists(c):
            return c

    # Dynamic glob fallback: most-recently-modified best_model or step file.
    glob_patterns = [
        os.path.join(output_dir, f"best_model_{label}_*.safetensors"),
        os.path.join(output_dir, f"best_model_{label}_*.bin"),
        os.path.join(output_dir, "weights", f"*_{label}-step*.safetensors"),
        os.path.join(output_dir, "weights", f"*_{label}-step*.bin"),
    ]
    for pat in glob_patterns:
        hits = sorted(glob.glob(pat), key=os.path.getmtime, reverse=True)
        if hits:
            return hits[0]

    raise FileNotFoundError(
        f"[AUTO-CHECKPOINT] No checkpoint found for '{label}'. "
        f"Searched: {explicit_path} and glob patterns in {output_dir}. "
        "Train first or pass the correct --learned_embeds path."
    )


def inference(args):
    """Run inference loop and save generated images."""

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

    _t_start = time.time()
    _n_gpus = torch.cuda.device_count()
    _gpu_info = (
        ", ".join(torch.cuda.get_device_name(i) for i in range(_n_gpus))
        if _n_gpus > 0 else "CPU"
    )

    logger.info("=" * 60)
    logger.info("START: INFERENCE (TEST) PIPELINE")
    logger.info(f"timestamp           : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"GPU disponibili     : {_n_gpus}  ({_gpu_info})")
    logger.info(f"base_model          : {args.pretrained_model_name_or_path}")
    logger.info(f"resolution          : {args.resolution}x{args.resolution}")
    logger.info(f"num_inference_steps : {args.num_inference_steps}")
    logger.info(f"generation_steps    : {args.generation_steps}  (max immagini generate)")
    logger.info(f"guidance_scale      : {args.guidance_scale}")
    logger.info(f"prompt_template     : '{args.prompt}'")
    logger.info(f"uncond_prompt       : '{args.prompt.replace(args.placeholder_token, '').strip()}'")
    logger.info(f"placeholder_token   : {args.placeholder_token}")
    logger.info(f"mixed_precision     : {args.mixed_precision}")
    logger.info(f"precomputed_embeds  : {args.use_precomputed_embeddings}")
    logger.info(f"embeddings_dir      : {args.embeddings_dir}")
    logger.info(f"latents_dir         : {args.latents_dir}")
    logger.info(f"learned_embeds      : {args.learned_embeds}")
    logger.info(f"data_dir            : {args.data_dir}")
    logger.info(f"output_dir          : {args.output_dir}")
    logger.info(f"run_name            : {args.run_name}")
    logger.info(f"lora                : {args.lora}")
    if args.lora:
        logger.info(f"learned_embeds_lora : {args.learned_embeds_lora}")
        # Existence is verified later by _resolve_checkpoint_path (supports both .safetensors and .bin).
    logger.info(f"  seed                : {args.seed}")
    logger.info("=" * 60)

    # --- Auto-resolve checkpoint paths (handles .bin → .safetensors migration) ---
    # This runs before MusicTokenWrapper is instantiated so args already contains
    # the validated, existing path when __init__ calls _load_weights.
    args.learned_embeds = _resolve_checkpoint_path(
        explicit_path=args.learned_embeds,
        output_dir=args.output_dir,
        stem="learned_embeds",
        label="embedder",
    )
    logger.info(f"[AUTO-CHECKPOINT] Resolved embedder weights: {args.learned_embeds}")

    if args.lora:
        args.learned_embeds_lora = _resolve_checkpoint_path(
            explicit_path=args.learned_embeds_lora,
            output_dir=args.output_dir,
            stem="learned_embeds_lora_layers",
            label="lora",
        )
        logger.info(f"[AUTO-CHECKPOINT] Resolved LoRA weights: {args.learned_embeds_lora}")

    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    else:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer"
        )

    num_added_tokens = tokenizer.add_tokens(args.placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {args.placeholder_token}. "
            "Please pass a different `placeholder_token`."
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    test_dataset = Museart(
        args=args,
        tokenizer=tokenizer,
        logger=logger,
        size=args.resolution,
    )

    dataloader_generator = torch.Generator(device="cpu")
    if args.seed is not None:
        dataloader_generator.manual_seed(args.seed)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True,
        num_workers=args.dataloader_num_workers,
        generator=dataloader_generator,
    )

    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    model = MusicTokenWrapper(args).to(weight_dtype).eval()

    model = model.to(device)
    base_model = model

    placeholder_token_id = tokenizer.convert_tokens_to_ids(args.placeholder_token)

    base_model.text_encoder.resize_token_embeddings(len(tokenizer))
    base_model.set_placeholder_token_id(placeholder_token_id)

    _n_gpus_test = torch.cuda.device_count()
    if _n_gpus_test > 1:
        logger.info(
            f"[MULTI-GPU TEST] {_n_gpus_test} GPU disponibili. "
            "Inferenza su processo singolo (no DDP — batch_size=1). "
            "Strategia: UNet/text_encoder su cuda:0, VAE decode su cuda:1."
        )
        _vae_device = torch.device("cuda:1")
    else:
        _vae_device = device
        logger.info(f"[SINGLE-GPU TEST] {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    import logging as _logging
    _diffusers_logger = _logging.getLogger('diffusers')
    _prev_level = _diffusers_logger.level
    _diffusers_logger.setLevel(_logging.ERROR)

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        tokenizer=tokenizer,
        text_encoder=base_model.text_encoder,
        vae=base_model.vae,
        unet=base_model.unet,
        cache_dir=args.hf_cache_dir,
    ).to(device)

    _diffusers_logger.setLevel(_prev_level)
    pipeline.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

    if _n_gpus_test > 1:
        try:
            pipeline.vae = pipeline.vae.to(_vae_device)
            logger.info(f"[MULTI-GPU TEST] VAE spostato su {_vae_device} — "
                        "UNet su cuda:0, VAE su cuda:1 (decode parallelizzato)")
        except Exception as _e:
            logger.warning(f"[MULTI-GPU TEST] Impossibile spostare VAE su cuda:1: {_e}. "
                           "Uso pipeline standard su cuda:0.")

    from diffusers import EulerDiscreteScheduler

    scheduler = EulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        cache_dir=args.hf_cache_dir,
    )
    # scheduler.set_timesteps(args.num_inference_steps)
    logger.info(f"[FIX-A] Scheduler: {scheduler.__class__.__name__} | "
                f"prediction_type={scheduler.config.get('prediction_type', 'N/A')}")

    _uncond_prompt_text = args.prompt.replace(args.placeholder_token, "").strip()
    logger.info(f"[FIX-CFG] uncond_prompt: '{_uncond_prompt_text}'")
    uncond_input = tokenizer(
        [_uncond_prompt_text],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.to(device)

    prompt_input = tokenizer(
        [args.prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)

    with torch.no_grad():
        uncond_embeddings = base_model.text_encoder(
            None,
            input_ids=uncond_input,
        )[0].to(dtype=weight_dtype) 

    for step, batch in enumerate(test_dataloader):
        if step >= args.generation_steps:
            break

        # Resetta lo scheduler per la nuova immagine
        scheduler.set_timesteps(args.num_inference_steps)

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)

        aud_features = batch["audio_features"].to(dtype=weight_dtype)

        with torch.no_grad():
            audio_token = base_model.embedder(aud_features).to(dtype=weight_dtype)
            
            cond_embeddings = base_model.text_encoder(
                audio_token,
                input_ids=prompt_input,
            )[0].to(dtype=weight_dtype) 

            text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

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
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

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

            _sf = base_model.vae.config.scaling_factor  # 0.18215 per SD 2.1
            latents = latents / _sf
            
            # 1. Sposta temporaneamente il VAE in float32
            base_model.vae.to(dtype=torch.float32)
            
            latents = latents.to(base_model.vae.device).float()   # float32 critico
            image_tensor = base_model.vae.decode(latents).sample
            
            # 2. Riporta il VAE al tipo originale (es. fp16) per coerenza e risparmio VRAM
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

    _total_elapsed = time.time() - _t_start
    _h = int(_total_elapsed // 3600)
    _m = int((_total_elapsed % 3600) // 60)
    _s = int(_total_elapsed % 60)
    _elapsed_str = f"{_h}h {_m:02d}m {_s:02d}s" if _h else f"{_m}m {_s:02d}s"

    imgs_dir = os.path.join(args.output_dir, "imgs", args.run_name)
    n_generated = min(step + 1, args.generation_steps)

    output_files_info = []
    total_imgs_mb = 0.0
    if os.path.isdir(imgs_dir):
        for fname in sorted(os.listdir(imgs_dir)):
            fpath = os.path.join(imgs_dir, fname)
            if os.path.isfile(fpath):
                size_kb = os.path.getsize(fpath) / 1024
                total_imgs_mb += size_kb / 1024
                output_files_info.append((fname, size_kb))

    logger.info("=" * 60)
    logger.info("COMPLETED SUCCESSFULLY: INFERENCE (TEST) PIPELINE")
    logger.info(f"end timestamp       : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"total time          : {_elapsed_str}  ({_total_elapsed:.1f}s)")
    logger.info(f"images generated    : {n_generated}")
    logger.info(f"avg time/image      : {_total_elapsed / max(n_generated, 1):.1f}s")
    logger.info("  --- Generated images ---")
    for fname, size_kb in output_files_info:
        logger.info(f"    {fname:<50} {size_kb:>6.1f} KB")
    logger.info("  --- Total output ---")
    logger.info(f"  {len(output_files_info)} immagini  —  total space: {total_imgs_mb:.2f} MB")
    logger.info(f"output_dir          : {imgs_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    inference(args)