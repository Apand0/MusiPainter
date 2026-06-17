# @title test_no_accel_colab.py
"""
Inference script for Musipainter (Early Fusion branch).

Generates images from audio embeddings using EarlyFusionEncoder +
frozen Stable Diffusion UNet. Audio embeddings are loaded via
LazyEmbeddingIndex which accepts a comma-separated --embeddings_dir
pointing to one or more Kaggle Dataset directories.

[FIX-CFG-T_A] The unconditional embedding sequence length is now
inferred dynamically from the first sample of the dataset instead of
being hardcoded. This ensures torch.cat([uncond, cond]) never
fails with a size mismatch regardless of which temporal_pool_stride
was used during preprocessing.

Note on sequence lengths:
  EarlyFusionEncoder outputs [B, actual_T_a + 1 + T_t, output_size] where:
  actual_T_a = n_audio_queries (if >0) or T_a (if 0)
  1   = FuseLIP separator token
  T_t = 77 — CLIP text tokens
  The +1 separator is handled internally by EarlyFusionEncoder and is
  transparent to the UNet, which simply attends over the full sequence
  as encoder_hidden_states (LDM Sec. 3.3).
"""

import argparse
import logging
import os
import random
import time

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset, DataLoader
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
        raise argparse.ArgumentTypeError(f"Valore booleano atteso, ricevuto: '{v}'")

    parser = argparse.ArgumentParser(description="Testing script con pre-encoded embeddings")

    from modules.preprocess.argparse_multiembedding import add_multiembedding_args
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

    # Early Fusion hyper-parameters (must match training config)
    parser.add_argument("--ef_d_model", type=int, default=512,
                        help="EarlyFusionEncoder shared Transformer hidden dim.")
    parser.add_argument("--ef_nhead", type=int, default=8,
                        help="EarlyFusionEncoder number of attention heads.")
    parser.add_argument("--ef_num_layers", type=int, default=4,
                        help="EarlyFusionEncoder number of Transformer layers.")
    parser.add_argument("--ef_dropout", type=float, default=0.1,
                        help="EarlyFusionEncoder dropout rate.")
    parser.add_argument("--ef_n_audio_queries", type=int, default=1,
                        help="0=Full T_a (FuseLIP), 1=AttentivePooling (MusiPainter, default), >1=Resampler")
    parser.add_argument(
        "--uncond_mode", type=str, default="zeros",
        choices=["zeros", "text_only"],
        help=(
            "'zeros': unconditional embedding is all-zeros audio + empty text "
            "(corresponds to p(x) — the fully unconditional distribution trained "
            "via the ~3%% CFG dropout). "
            "'text_only': unconditional embedding uses the text prompt without "
            "audio (corresponds to p(x|text) — trained via the ~7%% audio-only "
            "dropout). CFG then amplifies exactly the audio contribution above "
            "the text-only baseline: ε̃ = ε_text + s*(ε_audio+text - ε_text)."
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
    Resolve best available checkpoint, supporting .bin → .safetensors fallback.

    Search order:
      1. explicit_path as provided
      2. .safetensors sibling of a .bin explicit_path (format migration)
      3. <output_dir>/<stem>.safetensors
      4. <output_dir>/<stem>.bin
      5. Most-recent best_model_<label>_*.safetensors in output_dir
      6. Most-recent best_model_early_fusion_*.safetensors in output_dir
      7. Most-recent weights/*_<label>-step*.safetensors in output_dir/weights/
      8. Most-recent weights/*_early_fusion-step*.safetensors
    """
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


# ─────────────────────────────────────────────────────────────────────────────
#  DYNAMIC T_A PROBE
# ─────────────────────────────────────────────────────────────────────────────

def _probe_audio_frame_count(dataset) -> int:
    """
    Load one sample from the dataset and return its audio frame count T_a.

    This is the only correct way to obtain T_a: reading it from the actual
    precomputed embeddings ensures the unconditional sequence always matches
    the conditional one regardless of which temporal_pool_stride was used.

    Note: T_a is the number of BEATs frames in the precomputed embeddings,
    NOT the output sequence length of EarlyFusionEncoder. UNet directly 
    processes actual_T_a + 1 + T_t tokens, preserving the full temporal resolution
    T_a is only needed here to size the silent audio tensor passed to
    EarlyFusionEncoder when building the unconditional CFG embedding.

    stride=1  → T_a ≈ 376
    stride=4  → T_a ≈ 94
    stride=8  → T_a ≈ 47
    stride=16 → T_a ≈ 23
    """
    probe_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    first_batch  = next(iter(probe_loader))
    t_a = first_batch["audio_features"].shape[1]
    logger.info(
        f"[CFG] Probed T_a={t_a} from first sample "
        f"(shape={list(first_batch['audio_features'].shape)}). "
        f"Silent audio tensor [1,{t_a},2304] will be used for unconditional CFG embedding."
    )
    return t_a


# ─────────────────────────────────────────────────────────────────────────────
#  UNCONDITIONAL EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────

def _build_uncond_embedding(
    args,
    base_model,
    tokenizer,
    t_a: int,
    weight_dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Build the unconditional embedding used for Classifier-Free Guidance (CFG).

    CFG formula (Ho & Salimans 2022, applied in LDM Sec. 4 / Appendix):
        ε̃ = ε_uncond + s * (ε_cond - ε_uncond)
    where s = guidance_scale and ε_uncond is produced by this function.

    The silent audio tensor has shape [1, t_a, 2304] where t_a is inferred
    dynamically from the dataset. The output retains the audio's native dimensions, 
    always resulting in: [1, actual_T_a + 1 + T_t, output_size]
    regardless of t_a. This matches the shape of the conditional embedding
    produced in the generation loop, ensuring torch.cat([uncond, cond]) always
    succeeds without a size mismatch.

    Args:
        args:         parsed CLI args (uncond_mode, prompt, placeholder_token).
        base_model:   MusicTokenWrapper instance.
        tokenizer:    CLIPTokenizer.
        t_a:          audio frame count read from the actual embeddings.
        weight_dtype: fp16 / bf16 / fp32.
        device:       CUDA or CPU device.

    Returns:
        uncond_embeddings: [1, actual_T_a + 1 + T_t, output_size] float tensor.
    """
    audio_dim     = 768 * 3          # BEATs layers 4+8+12 concatenated
    silent_audio  = torch.zeros(1, t_a, audio_dim, dtype=weight_dtype, device=device)

    with torch.no_grad():
        if args.uncond_mode == "zeros":
            # Empty text + silent audio → corresponds to p(x), the fully
            # unconditional distribution trained via the ~3% CFG dropout.
            uncond_ids = tokenizer(
                [""],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            uncond_text = base_model._get_text_embeddings(uncond_ids).to(weight_dtype)

            uncond_embeddings = base_model.early_fusion(
                audio_tokens=silent_audio,
                text_tokens=uncond_text,
            )   # [1, actual_T_a+1+T_t, output_size]

            logger.info(
                f"[CFG] uncond_mode=zeros | "
                f"silent audio shape=[1,{t_a},{audio_dim}] | "
                f"uncond_embeddings shape={list(uncond_embeddings.shape)}"
            )

        else:   # text_only
            # Prompt text without the placeholder + silent audio → corresponds
            # to p(x|text), the text-conditioned distribution trained via the
            # ~7% audio-only dropout. CFG with this baseline amplifies exactly
            # the audio contribution above the text-only representation:
            #   ε̃ = ε_text + s * (ε_audio+text - ε_text)
            text_only_prompt = args.prompt.replace(args.placeholder_token, "").strip()
            uncond_ids = tokenizer(
                [text_only_prompt],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            uncond_text = base_model._get_text_embeddings(uncond_ids).to(weight_dtype)

            uncond_embeddings = base_model.early_fusion(
                audio_tokens=silent_audio,
                text_tokens=uncond_text,
            )   # [1, actual_T_a+1+T_t, output_size]

            logger.info(
                f"[CFG] uncond_mode=text_only | "
                f"prompt='{text_only_prompt}' | "
                f"silent audio shape=[1,{t_a},{audio_dim}] | "
                f"uncond_embeddings shape={list(uncond_embeddings.shape)}"
            )

    return uncond_embeddings


# ─────────────────────────────────────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

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

    _t_start  = time.time()
    _n_gpus   = torch.cuda.device_count()
    _gpu_info = (
        ", ".join(torch.cuda.get_device_name(i) for i in range(_n_gpus))
        if _n_gpus > 0 else "CPU"
    )

    _n_q = getattr(args, 'ef_n_audio_queries', 1)
    _seq_desc = (
        f"T_a + 1 + 77 (Full Temporal Resolution)"
        if _n_q == 0 else
        f"{_n_q} + 1 + 77 (Resampler)" if _n_q > 1 else
        f"1 + 1 + 77 (AttentivePooling)"
    )

    logger.info("=" * 60)
    logger.info("START: INFERENCE — Early Fusion branch")
    logger.info(f"timestamp           : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"GPU                 : {_n_gpus}  ({_gpu_info})")
    logger.info(f"base_model          : {args.pretrained_model_name_or_path}")
    logger.info(f"resolution          : {args.resolution}x{args.resolution}")
    logger.info(f"num_inference_steps : {args.num_inference_steps}")
    logger.info(f"generation_steps    : {args.generation_steps}")
    logger.info(f"guidance_scale      : {args.guidance_scale}")
    logger.info(f"uncond_mode         : {args.uncond_mode}")
    logger.info(f"prompt_template     : '{args.prompt}'")
    logger.info(f"ef_d_model          : {args.ef_d_model}")
    logger.info(f"ef_nhead            : {args.ef_nhead}")
    logger.info(f"ef_num_layers       : {args.ef_num_layers}")
    logger.info(f"ef_dropout          : {args.ef_dropout}")
    logger.info(f"ef_n_audio_queries  : {_n_q}")
    logger.info(f"UNet seq length     : {_seq_desc}")
    logger.info(f"embeddings_dir      : {args.embeddings_dir}")
    logger.info(f"embeddings_preload  : {args.embeddings_preload_all}")
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # ── Dataset ───────────────────────────────────────────────────────────────
    logger.info(
        f"LazyEmbeddingIndex will scan: {args.embeddings_dir}  "
        f"[preload_all={args.embeddings_preload_all}, "
        f"max_sf_handles={args.embeddings_max_sf_handles}]"
    )
    test_dataset = Museart(
        args=args, tokenizer=tokenizer, logger=logger, size=args.resolution,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    weight_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(args.mixed_precision, torch.float32)

    model = MusicTokenWrapper(args).to(weight_dtype).eval().to(device)
    base_model = model

    # ── Scheduler (EulerDiscrete, v-prediction safe for SD 2.x) ──────────────
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

    # ── Dynamic T_a probe ─────────────────────────────────────────────────────
    # [FIX-CFG-T_A] Read the actual BEATs frame count from the first dataset
    # sample. This is needed only to size the silent audio tensor for the
    # unconditional CFG embedding; the EarlyFusionEncoder will compress it
    # internally based on n_audio_queries, so the UNet always sees actual_T_a+1+T_t tokens.
    t_a: int = _probe_audio_frame_count(test_dataset)

    # ── Unconditional embedding for CFG (built once, reused for all samples) ──
    uncond_embeddings = _build_uncond_embedding(
        args=args,
        base_model=base_model,
        tokenizer=tokenizer,
        t_a=t_a,
        weight_dtype=weight_dtype,
        device=device,
    )

    # ── Tokenise the conditional prompt (constant across all images) ──────────
    cond_input_ids = tokenizer(
        [args.prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)

    # ── DataLoader for generation loop ────────────────────────────────────────
    dataloader_generator = torch.Generator(device="cpu")
    if args.seed is not None:
        dataloader_generator.manual_seed(args.seed)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        generator=dataloader_generator,
    )

    # ── Generation loop ───────────────────────────────────────────────────────
    for step, batch in enumerate(test_dataloader):
        if step >= args.generation_steps:
            break

        scheduler.set_timesteps(args.num_inference_steps)

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)

        aud_features = batch["audio_features"].to(dtype=weight_dtype)

        # Sanity-check: the conditional T_a must match what we probed above.
        # If it doesn't, there are mixed-stride embeddings in the dataset —
        # which should never happen but is detected here explicitly.
        assert aud_features.shape[1] == t_a, (
            f"[CFG] Audio frame count mismatch at step {step}: "
            f"expected T_a={t_a} (probed from dataset) but got "
            f"T_a={aud_features.shape[1]}. "
            "Check that all embeddings were preprocessed with the same stride."
        )

        with torch.no_grad():
            # ── Conditional embedding (audio + text fused) ────────────────────
            # EarlyFusionEncoder: τ_θ(audio, text) → [1, actual_T_a+1+T_t, output_size]
            text_tokens = base_model._get_text_embeddings(cond_input_ids).to(weight_dtype)
            cond_embeddings = base_model.early_fusion(
                audio_tokens=aud_features,
                text_tokens=text_tokens,
            )   # [1, actual_T_a+1+T_t, output_size]

            # CFG (LDM Sec. 4 / Ho & Salimans 2022):
            # stack uncond + cond → [2, actual_T_a+1+T_t, output_size]
            # Both tensors have the same sequence length because the
            # EarlyFusionEncoder uses the same n_audio_queries for both.
            text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])
            # [2, actual_T_a+1+T_t, output_size]

            # ── Latent diffusion (LDM Sec. 3.2) ──────────────────────────────
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
                # CFG formula: ε̃ = ε_uncond + s * (ε_cond - ε_uncond)
                noise_pred = noise_pred_uncond + args.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            # ── VAE decode (LDM Sec. 3.1) ─────────────────────────────────────
            # Undo the 0.18215 scaling applied during encoding
            _sf = base_model.vae.config.scaling_factor    # 0.18215 for SD/SD2
            latents = latents / _sf
            base_model.vae.to(dtype=torch.float32)
            image_tensor = base_model.vae.decode(
                latents.to(base_model.vae.device).float()
            ).sample
            base_model.vae.to(dtype=weight_dtype)

            image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
            image_np = (
                image_tensor.cpu().permute(0, 2, 3, 1).float().numpy()[0] * 255
            ).round().astype("uint8")

        from PIL import Image
        image = Image.fromarray(image_np)
        save_name = (
            f'{batch["aud_id"][0]}_{batch["image_id"][0]}_{batch["label"][0]}.jpg'
        )
        image.save(os.path.join(args.output_dir, "imgs", args.run_name, save_name))
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