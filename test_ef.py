# @title test_ef.py
"""
test_ef.py — Inference script per Musipainter (Branch FuseLIP / Early Fusion).

NOTE: rinominato da test_no_accel_colab.py per convivere con la variante
Cross-Attention (test_ca.py) nello stesso repository, selezionabile tramite
musipainter_test.py --architecture Musipainter-EF.
Unica modifica rispetto all'originale: l'import di MusicTokenWrapper punta
al modulo rinominato modules.MusicToken.MusicToken_no_accel_ef.
Nessun'altra logica è stata alterata.
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
from modules.MusicToken.MusicToken_no_accel_ef import MusicTokenWrapper

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

    parser = argparse.ArgumentParser(description="Inference script — Musipainter FuseLIP branch")

    from modules.preprocess.argparse_multiembedding import add_multiembedding_args
    add_multiembedding_args(parser)

    parser.add_argument("--learned_embeds",      type=str, default='./output/learned_embeds.safetensors')
    parser.add_argument("--learned_vae",         type=str, default='./output/vae_learned_embeds.bin')
    parser.add_argument("--learned_aud_encoder", type=str, default='./output/aud_encoder_learned_embeds.bin')
    parser.add_argument("--learned_unet",        type=str, default='./output/unet_learned_embeds.bin')
    parser.add_argument("--learned_embeds_lora", type=str, default='./output/learned_embeds_lora_layers.safetensors')
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='stabilityai/stable-diffusion-2')
    parser.add_argument("--revision",            type=str, default=None, required=False)
    parser.add_argument("--tokenizer_name",      type=str, default=None)
    parser.add_argument("--data_dir",            type=str, default="./Museart/")
    parser.add_argument("--latents_dir",         type=str, default="./image_latents/")
    parser.add_argument("--use_precomputed_embeddings", type=_str2bool, default=True)
    parser.add_argument("--placeholder_token",   type=str, default="<*>")
    parser.add_argument("--output_dir",          type=str, default="./output/test/")
    parser.add_argument("--seed",                type=int, default=1234)
    parser.add_argument("--resolution",          type=int, default=768)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--logging_dir",         type=str, default="logs")
    parser.add_argument("--mixed_precision",     type=str, default="fp16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow_tf32",          action="store_true")
    parser.add_argument("--report_to",           type=str, default="tensorboard")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--data_set",            type=str, default='test',
                        choices=['train', 'validation', 'test'])
    parser.add_argument("--generation_steps",    type=int, default=50)
    parser.add_argument("--run_name",            type=str, default='MusicToken')
    parser.add_argument("--set_size",            type=str, default='full')
    parser.add_argument("--prompt",              type=str, default='An art image of <*>')
    parser.add_argument("--input_length",        type=int, default=30)
    parser.add_argument("--lora",                type=_str2bool, default=False)
    parser.add_argument("--aud_encoder",         type=_str2bool, default=False)
    parser.add_argument("--unet",                type=_str2bool, default=False)
    parser.add_argument("--vae",                 type=_str2bool, default=False)
    parser.add_argument("--guidance_scale",      type=float, default=4.0)
    parser.add_argument("--center_crop",         action="store_true", default=False)
    parser.add_argument("--hf_cache_dir",        type=str, default="/tmp/hf_model_cache")

    # Early Fusion hyper-parameters (devono corrispondere alla configurazione di training)
    parser.add_argument("--ef_d_model",          type=int,   default=512)
    parser.add_argument("--ef_nhead",            type=int,   default=8)
    parser.add_argument("--ef_num_layers",       type=int,   default=4)
    parser.add_argument("--ef_dropout",          type=float, default=0.1)
    parser.add_argument("--ef_n_audio_queries",  type=int,   default=1,
                        help="0=Full T_a (FuseLIP), 1=AttentivePooling (default), >1=Resampler")
    parser.add_argument(
        "--uncond_mode", type=str, default="zeros",
        choices=["zeros", "text_only"],
        help=(
            "'zeros' (default): uncond = audio silenzioso + testo vuoto → p(x). "
            "Corrisponde al ~3%% CFG dropout full-zero usato in training. "
            "'text_only': uncond = audio silenzioso + prompt testuale → p(x|text). "
        ),
    )

    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1:
        args.local_rank = env_local_rank

    if args.data_dir is None:
        raise ValueError("Specificare --data_dir.")

    args.image_latents_dir = args.latents_dir
    return args


# ─────────────────────────────────────────────────────────────────────────────
#  CHECKPOINT RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_checkpoint_path(explicit_path, output_dir, stem, label):
    """
    Risolve il miglior checkpoint disponibile, con fallback .bin → .safetensors.
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
        f"[AUTO-CHECKPOINT] Nessun checkpoint trovato per '{label}'. "
        f"Cercato: {explicit_path} e pattern glob in {output_dir}."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  DYNAMIC T_A PROBE
# ─────────────────────────────────────────────────────────────────────────────

def _probe_audio_frame_count(dataset) -> int:
    probe_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    first_batch  = next(iter(probe_loader))
    t_a = first_batch["audio_features"].shape[1]
    logger.info(
        f"[CFG] T_a probed={t_a} dal primo sample "
        f"(shape={list(first_batch['audio_features'].shape)}). "
        f"Il silent audio tensor per CFG avrà shape [1,{t_a},2304]."
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
    audio_dim    = 768 * 3
    silent_audio = torch.zeros(1, t_a, audio_dim, dtype=torch.float32, device=device)

    with torch.no_grad():
        if args.uncond_mode == "zeros":
            uncond_ids = tokenizer(
                [""],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            uncond_text = base_model._get_text_embeddings(uncond_ids)  # float32

            uncond_embeddings = base_model.early_fusion(
                audio_tokens=silent_audio,
                text_tokens=uncond_text,
            )

            logger.info(
                f"[CFG] uncond_mode=zeros | "
                f"silent_audio=[1,{t_a},{audio_dim}] | "
                f"uncond_embeddings shape={list(uncond_embeddings.shape)}"
            )

        else:  # text_only
            text_only_prompt = args.prompt.replace(args.placeholder_token, "").strip()
            uncond_ids = tokenizer(
                [text_only_prompt],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            uncond_text = base_model._get_text_embeddings(uncond_ids)  # float32

            uncond_embeddings = base_model.early_fusion(
                audio_tokens=silent_audio,
                text_tokens=uncond_text,
            )

            logger.info(
                f"[CFG] uncond_mode=text_only | "
                f"prompt='{text_only_prompt}' | "
                f"silent_audio=[1,{t_a},{audio_dim}] | "
                f"uncond_embeddings shape={list(uncond_embeddings.shape)}"
            )

    return uncond_embeddings.to(dtype=weight_dtype)


# ─────────────────────────────────────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def inference(args):
    """Loop di inferenza Early Fusion e salvataggio immagini generate."""

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
        f"T_a + 1 + 77 (Full Temporal — FuseLIP puro)"
        if _n_q == 0 else
        f"{_n_q} + 1 + 77 (Resampler)" if _n_q > 1 else
        f"1 + 1 + 77 (AttentivePooling)"
    )

    logger.info("=" * 60)
    logger.info("START: INFERENCE — FuseLIP / Early Fusion branch")
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
        f"LazyEmbeddingIndex: {args.embeddings_dir}  "
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

    model = MusicTokenWrapper(args).eval().to(device)
    base_model = model

    # ── Scheduler ─────────────────────────────────────────────────────────────
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
    t_a: int = _probe_audio_frame_count(test_dataset)

    # ── Unconditional embedding per CFG (costruito una volta sola) ────────────
    uncond_embeddings = _build_uncond_embedding(
        args=args,
        base_model=base_model,
        tokenizer=tokenizer,
        t_a=t_a,
        weight_dtype=weight_dtype,
        device=device,
    )

    # ── Tokenizza il prompt condizionale (costante per tutti i sample) ────────
    cond_input_ids = tokenizer(
        [args.prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)

    # ── DataLoader per il loop di generazione ─────────────────────────────────
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

        aud_features = batch["audio_features"].to(dtype=torch.float32)

        assert aud_features.shape[1] == t_a, (
            f"[CFG] Audio frame count mismatch al step {step}: "
            f"atteso T_a={t_a} (probed dal dataset) ma ricevuto "
            f"T_a={aud_features.shape[1]}. "
            "Verifica che tutti gli embeddings siano stati preprocessati con lo stesso stride."
        )

        with torch.no_grad():
            text_tokens = base_model._get_text_embeddings(cond_input_ids)  # float32
            cond_embeddings = base_model.early_fusion(
                audio_tokens=aud_features,
                text_tokens=text_tokens,
            )

            cond_embeddings = cond_embeddings.to(dtype=weight_dtype)

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

            _sf = base_model.vae.config.scaling_factor
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
        logger.info(f"Salvato: {save_name} ({step + 1}/{args.generation_steps})")

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
