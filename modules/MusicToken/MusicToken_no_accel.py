# @title modules/MusicToken/MusicToken_no_accel.py
"""
MusicToken_no_accel.py — Core model wrapper for the Musipainter pipeline.

Wraps all Stable Diffusion components (VAE, UNet, CLIP text encoder) together
with the BEATs audio encoder and the trainable FGAEmbedder. Handles:
  - Component loading in fp16 (frozen) and fp32 (trainable embedder)
  - Optional LoRA adapters on UNet attention layers
  - Gradient flow via a Straight-Through Estimator so the frozen UNet
    does not participate in the backward pass
  - Train / test mode switching with checkpoint load/save
"""

import gc
import torch
import torch.nn as nn
import logging
from diffusers.loaders import AttnProcsLayers
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import LoRAAttnProcessor

from modules.BEATs.BEATs import BEATs, BEATsConfig
from modules.MusicToken.embedder import FGAEmbedder
from modules.clip_text_model.modeling_clip import CLIPTextModel

logger = logging.getLogger(__name__)


def _load_weights(path, map_device) -> dict:
    """
    Load model weights from a .safetensors or legacy .bin/.pt file.

    Resolution order:
      1. If *path* ends with '.safetensors' → use load_safetensors (pickle-free).
      2. Otherwise → torch.load (legacy .bin or .pt checkpoint).
    If the exact path does not exist and a .bin was requested, the function
    tries the corresponding .safetensors path before raising FileNotFoundError.

    Args:
        path:       path string to the weight file.
        map_device: torch.device used as map_location.

    Returns:
        dict mapping parameter name → torch.Tensor.
    """
    import os
    resolved = str(path)

    # Transparent fallback: if a .bin is requested but only .safetensors exists.
    if not os.path.exists(resolved) and resolved.endswith(".bin"):
        sf_candidate = resolved[:-4] + ".safetensors"
        if os.path.exists(sf_candidate):
            logger.info(
                f"[_load_weights] .bin not found, auto-switching to: {sf_candidate}"
            )
            resolved = sf_candidate

    if not os.path.exists(resolved):
        raise FileNotFoundError(f"[_load_weights] Weight file not found: {resolved}")

    if resolved.endswith(".safetensors"):
        from modules.preprocess.utils import load_safetensors
        return load_safetensors(resolved, device=str(map_device))

    # Legacy torch.load path.
    return torch.load(resolved, map_location=map_device)


class MusicTokenWrapper(nn.Module):
    """
    Wraps VAE, UNet, CLIP text encoder, BEATs audio encoder and FGAEmbedder
    into a single nn.Module for training and inference.

    Only FGAEmbedder (and optionally LoRA layers) has requires_grad=True.
    All other components are frozen and kept in float16 to save VRAM.
    """

    def __init__(self, args):
        super().__init__()

        frozen_dtype = torch.float16

        # --- Load frozen SD components in fp16 on CPU ---
        # .to(device) is intentionally deferred to the training loop, which calls
        # model.to(device) AFTER wrapping with DDP/DataParallel so that all
        # parameters land on the GPU at the same time.
        _hf_cache = getattr(args, 'hf_cache_dir', '/tmp/hf_model_cache')

        self.text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=getattr(args, 'revision', None),
            torch_dtype=frozen_dtype,
            cache_dir=_hf_cache,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=getattr(args, 'revision', None),
            torch_dtype=frozen_dtype,
            cache_dir=_hf_cache,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=getattr(args, 'revision', None),
            torch_dtype=frozen_dtype,
            cache_dir=_hf_cache,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if hasattr(args, 'use_precomputed_embeddings') and args.use_precomputed_embeddings:
            logger.info("Pre-computed embeddings — BEATs not loaded")
            self.aud_encoder = None
        else:
            logger.info("Loading BEATs...")
            checkpoint = torch.load(
                'models/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt',
                map_location='cpu',
            )
            cfg = BEATsConfig(checkpoint['cfg'])
            self.aud_encoder = BEATs(cfg)
            self.aud_encoder.load_state_dict(checkpoint['model'])
            self.aud_encoder.predictor = None
            del checkpoint
            gc.collect()

        # FGAEmbedder input is the concatenation of BEATs layers 4, 8, 12 → dim 2304.
        input_size = 768 * 3
        if args.pretrained_model_name_or_path == "CompVis/stable-diffusion-v1-4":
            self.embedder = FGAEmbedder(input_size=input_size, output_size=768)
        else:
            self.embedder = FGAEmbedder(input_size=input_size, output_size=1024)

        # Set all SD components to eval mode; only the embedder will be trained.
        self.vae.eval()
        self.unet.eval()
        self.text_encoder.eval()
        if self.aud_encoder is not None:
            self.aud_encoder.eval()

        # --- Optional LoRA adapters on UNet attention layers ---
        # LoRA processors must be installed BEFORE the xformers/SDPA fallback so
        # that set_attn_processor() does not overwrite them.
        if hasattr(args, 'lora') and args.lora:
            lora_attn_procs = {}
            for name in self.unet.attn_processors.keys():
                cross_attention_dim = (
                    None if name.endswith("attn1.processor")
                    else self.unet.config.cross_attention_dim
                )
                if name.startswith("mid_block"):
                    hidden_size = self.unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = self.unet.config.block_out_channels[block_id]
                else:
                    hidden_size = self.unet.config.block_out_channels[0]
                lora_attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
                )
            self.unet.set_attn_processor(lora_attn_procs)
            self.lora_layers = AttnProcsLayers(self.unet.attn_processors)
            logger.info(
                f"[LORA-INIT] LoRAAttnProcessor installed on {len(lora_attn_procs)} "
                "UNet attention processors."
            )
        else:
            self.lora_layers = None

        # --- Attention backend: xformers → SDPA → vanilla fallback ---
        # When LoRA is active, skip the AttnProcessor2_0 fallback to avoid
        # overwriting the LoRA processors.
        try:
            self.unet.enable_xformers_memory_efficient_attention()
            logger.info("xformers memory efficient attention: ON")
        except Exception as e:
            logger.info(f"xformers not available ({e}), using torch SDPA")
            if not (hasattr(args, 'lora') and args.lora):
                try:
                    self.unet.set_attn_processor(
                        __import__('diffusers').models.attention_processor
                        .AttnProcessor2_0()
                    )
                    logger.info("torch SDPA (AttnProcessor2_0): ON")
                except Exception:
                    logger.info("Falling back to vanilla attention")
            else:
                logger.info(
                    "[LORA] AttnProcessor2_0 fallback skipped: LoRAAttnProcessor already active."
                )

        # --- Training vs test mode ---
        if args.data_set == 'train':
            self.vae.requires_grad_(False)
            self.unet.requires_grad_(False)
            self.unet.enable_gradient_checkpointing()
            self.text_encoder.requires_grad_(False)
            if self.aud_encoder is not None:
                self.aud_encoder.requires_grad_(False)
            self.embedder.requires_grad_(True)
            self.embedder.train()
            # torch.compile is applied in the training loop AFTER DDP wrap,
            # directly on base_model.embedder.
            if hasattr(args, 'lora') and args.lora:
                # Only put LoRA layers in train mode, not the full UNet backbone.
                self.lora_layers.requires_grad_(True)
                self.lora_layers.train()
                logger.info("[LORA-TRAIN] lora_layers.train() — UNet backbone remains frozen/eval.")

        elif args.data_set == 'test':
            map_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.embedder.eval()
            # Load via _load_weights to support both .safetensors and legacy .bin.
            _state = _load_weights(args.learned_embeds, map_device)
            # Strip _orig_mod. prefix produced by torch.compile checkpoints.
            if any(k.startswith("_orig_mod.") for k in _state.keys()):
                _state = {k[10:]: v for k, v in _state.items()}
            self.embedder.load_state_dict(_state)
            if hasattr(args, 'vae') and args.vae:
                # Load VAE weights (safetensors or legacy .bin).
                self.vae.load_state_dict(
                    _load_weights(args.learned_vae, map_device)
                )
            if hasattr(args, 'aud_encoder') and args.aud_encoder \
               and self.aud_encoder is not None:
                # Load audio encoder weights (safetensors or legacy .bin).
                self.aud_encoder.load_state_dict(
                    _load_weights(args.learned_aud_encoder, map_device)
                )
            if hasattr(args, 'unet') and args.unet:
                # Load UNet weights (safetensors or legacy .bin).
                self.unet.load_state_dict(
                    _load_weights(args.learned_unet, map_device)
                )
            if hasattr(args, 'lora') and args.lora:
                # AttnProcsLayers is a direct view of the UNet processor weights;
                # load_state_dict updates them in-place in the already-installed processors.
                if self.lora_layers is None:
                    raise RuntimeError(
                        "[LORA-TEST] lora_layers is None but --lora=True. "
                        "This should not happen: contact the developer."
                    )
                # Load LoRA weights (safetensors or legacy .bin).
                _lora_state = _load_weights(args.learned_embeds_lora, map_device)
                self.lora_layers.load_state_dict(_lora_state)
                self.lora_layers.eval()
                logger.info(
                    f"[LORA-TEST] LoRA weights loaded from: {args.learned_embeds_lora}"
                )

    def set_placeholder_token_id(self, token_id: int):
        """
        Set the real token ID of the <*> placeholder inside CLIPTextEmbeddings.

        Must be called immediately after tokenizer.add_tokens(placeholder_token) and
        text_encoder.resize_token_embeddings(len(tokenizer)), before any forward pass.
        This guarantees deterministic placeholder detection instead of the fragile
        input_ids.max() fallback.
        """
        embeddings_module = self.text_encoder.text_model.embeddings
        embeddings_module.placeholder_token_id = token_id
        logger.info(f"[FIX-PLACEHOLDER] placeholder_token_id set to {token_id}")

    def forward(self, audio_features, input_ids, noisy_latents, timesteps):
        """
        Args:
            audio_features : [B, T, 2304] float32
            input_ids      : [B, seq_len] long
            noisy_latents  : [B, 4, H/8, W/8] float16
            timesteps      : [B] long

        Returns:
            model_pred : [B, 4, H/8, W/8] float32
            audio_token: [B, output_size] float32
        """
        # Step 1: trainable FGAEmbedder → audio vector in CLIP embedding space
        audio_token = self.embedder(
            audio_features.float() if audio_features.dtype != torch.float32
            else audio_features
        )  # [B, output_size], requires_grad=True

        # Step 2: frozen CLIP text encoder injects audio_token at the <*> placeholder position
        encoder_hidden_states = self.text_encoder(
            audio_token,
            input_ids=input_ids,
        )[0].to(dtype=noisy_latents.dtype)
        # [B, seq_len, D], requires_grad=True (via the differentiable placeholder injection)

        # Step 3: frozen UNet forward pass (with gradient graph intact)
        # The UNet weights are frozen (requires_grad=False), but the graph is kept so
        # that gradients can flow back through encoder_hidden_states to the embedder.
        model_pred = self.unet(
            noisy_latents if noisy_latents.dtype == torch.float16
            else noisy_latents.to(dtype=torch.float16),
            timesteps,
            encoder_hidden_states,
        ).sample.float()
        # [B, 4, H/8, W/8], requires_grad=True (inherited from encoder_hidden_states)

        return model_pred, audio_token
