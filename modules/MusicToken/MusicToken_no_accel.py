# @title modules/MusicToken/MusicToken_no_accel.py
"""
MusicToken_no_accel.py — Core model wrapper for the Musipainter pipeline.

Wraps VAE, UNet, frozen CLIP token embeddings, BEATs audio encoder and the
trainable EarlyFusionEncoder into a single nn.Module for training and inference.
Only EarlyFusionEncoder (and optionally LoRA layers) has requires_grad=True.
All other components are frozen in float16 to save VRAM.
"""

import gc
import torch
import torch.nn as nn
import logging
from diffusers.loaders import AttnProcsLayers
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import LoRAAttnProcessor

from modules.BEATs.BEATs import BEATs, BEATsConfig
from modules.MusicToken.early_fusion_encoder import EarlyFusionEncoder

logger = logging.getLogger(__name__)


def _load_weights(path, map_device) -> dict:
    """
    Load model weights from a .safetensors or legacy .bin/.pt file.

    Resolution order:
      1. If *path* ends with '.safetensors' → use load_safetensors (pickle-free).
      2. Otherwise → torch.load (legacy .bin or .pt checkpoint).
    If the exact path does not exist and a .bin was requested, tries the
    corresponding .safetensors path before raising FileNotFoundError.
    """
    import os
    resolved = str(path)

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

    return torch.load(resolved, map_location=map_device)


class MusicTokenWrapper(nn.Module):
    """
    Wraps VAE, UNet, CLIP token embeddings, BEATs audio encoder and
    EarlyFusionEncoder into a single nn.Module for training and inference.

    Only EarlyFusionEncoder (and optionally LoRA layers) has requires_grad=True.
    All other components are frozen in float16 to save VRAM.
    """

    def __init__(self, args):
        super().__init__()

        frozen_dtype = torch.float16
        _hf_cache = getattr(args, 'hf_cache_dir', '/tmp/hf_model_cache')

        # ── Frozen SD components ──────────────────────────────────────────────
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

        # ── CLIP token embedding table (frozen) ───────────────────────────────
        # Only the nn.Embedding weight is retained; the full CLIP transformer
        # is discarded after extraction (~400 MB VRAM saved).
        from transformers import CLIPTextModel as _CLIPTextModel
        _clip = _CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=getattr(args, 'revision', None),
            torch_dtype=frozen_dtype,
            cache_dir=_hf_cache,
        )
        self.token_embedding = _clip.text_model.embeddings.token_embedding
        self._text_dim: int = self.token_embedding.embedding_dim
        del _clip
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── BEATs audio encoder (frozen) ──────────────────────────────────────
        if hasattr(args, 'use_precomputed_embeddings') and args.use_precomputed_embeddings:
            logger.info("Pre-computed embeddings — BEATs not loaded.")
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

        # ── Dimensions ────────────────────────────────────────────────────────
        audio_dim   = 768 * 3
        output_size = self.unet.config.cross_attention_dim
        # Legge i parametri FuseLIP dagli argomenti CLI
        d_model     = getattr(args, 'ef_d_model', 512)
        nhead       = getattr(args, 'ef_nhead', 8)
        num_layers  = getattr(args, 'ef_num_layers', 12)
        dropout     = getattr(args, 'ef_dropout', 0.1)

        logger.info(
            f"EarlyFusionEncoder: audio_dim={audio_dim}, text_dim={self._text_dim}, "
            f"output_size={output_size}, d_model={d_model}, nhead={nhead}"
        )

        # ── Trainable EarlyFusionEncoder ──────────────────────────────────────
        self.early_fusion = EarlyFusionEncoder(
            audio_dim=audio_dim,
            text_dim=self._text_dim,
            output_size=output_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )
        # Alias kept so checkpoint helpers (save_progress, _unwrap_compiled)
        # in the training script work without modification.
        self.embedder = self.early_fusion

        # ── Eval mode for frozen components ───────────────────────────────────
        self.vae.eval()
        self.unet.eval()
        self.token_embedding.eval()
        if self.aud_encoder is not None:
            self.aud_encoder.eval()

        # ── Optional LoRA adapters on UNet attention layers ───────────────────
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
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                )
            self.unet.set_attn_processor(lora_attn_procs)
            self.lora_layers = AttnProcsLayers(self.unet.attn_processors)
            logger.info(
                f"LoRAAttnProcessor installed on "
                f"{len(lora_attn_procs)} UNet attention processors."
            )
        else:
            self.lora_layers = None

        # ── Attention backend ─────────────────────────────────────────────────
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
                    "[LORA] AttnProcessor2_0 fallback skipped: "
                    "LoRAAttnProcessor already active."
                )

        # ── Training vs test mode ─────────────────────────────────────────────
        if args.data_set == 'train':
            self.vae.requires_grad_(False)
            self.unet.requires_grad_(False)
            self.unet.enable_gradient_checkpointing()
            self.token_embedding.requires_grad_(False)
            if self.aud_encoder is not None:
                self.aud_encoder.requires_grad_(False)
            self.early_fusion.requires_grad_(True)
            self.early_fusion.train()
            if hasattr(args, 'lora') and args.lora:
                self.lora_layers.requires_grad_(True)
                self.lora_layers.train()
                logger.info(
                    "lora_layers.train() — "
                    "UNet backbone remains frozen/eval."
                )

        elif args.data_set == 'test':
            map_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.early_fusion.eval()
            _state = _load_weights(args.learned_embeds, map_device)
            if any(k.startswith("_orig_mod.") for k in _state.keys()):
                _state = {k[10:]: v for k, v in _state.items()}
            self.early_fusion.load_state_dict(_state)
            if hasattr(args, 'vae') and args.vae:
                self.vae.load_state_dict(_load_weights(args.learned_vae, map_device))
            if hasattr(args, 'aud_encoder') and args.aud_encoder \
               and self.aud_encoder is not None:
                self.aud_encoder.load_state_dict(
                    _load_weights(args.learned_aud_encoder, map_device)
                )
            if hasattr(args, 'unet') and args.unet:
                self.unet.load_state_dict(_load_weights(args.learned_unet, map_device))
            if hasattr(args, 'lora') and args.lora:
                if self.lora_layers is None:
                    raise RuntimeError(
                        "lora_layers is None but --lora=True."
                    )
                _lora_state = _load_weights(args.learned_embeds_lora, map_device)
                self.lora_layers.load_state_dict(_lora_state)
                self.lora_layers.eval()
                logger.info(
                    f"LoRA weights loaded from: {args.learned_embeds_lora}"
                )

    # ──────────────────────────────────────────────────────────────────────────
    def set_placeholder_token_id(self, token_id: int):
        """
        No-op kept for CLI backward-compatibility.

        Placeholder injection into CLIP is not used in the Early Fusion
        architecture.  The token embedding table is queried directly for all
        text tokens; fusion is handled by EarlyFusionEncoder.
        """
        pass

    # ──────────────────────────────────────────────────────────────────────────
    def _get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up CLIP token embeddings for a batch of token-ID sequences.

        Args:
            input_ids: [B, seq_len] long

        Returns:
            [B, seq_len, text_dim] float32
        """
        with torch.no_grad():
            text_embeds = self.token_embedding(input_ids).float()
        return text_embeds

    # ──────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        audio_features: torch.Tensor,
        input_ids:      torch.Tensor,
        noisy_latents:  torch.Tensor,
        timesteps:      torch.Tensor,
    ):
        """
        Args:
            audio_features : [B, T_a, 2304]    float32  — BEATs concat(4,8,12)
            input_ids      : [B, seq_len]       long     — CLIP token IDs
            noisy_latents  : [B, 4, H/8, W/8]  float16
            timesteps      : [B]                long

        Returns:
            model_pred  : [B, 4, H/8, W/8] float32
            fused_embed : [B, output_size]  float32
        """
        text_tokens = self._get_text_embeddings(input_ids)

        audio_feats = (
            audio_features.float()
            if audio_features.dtype != torch.float32
            else audio_features
        )
        fused_embed = self.early_fusion(
            audio_tokens=audio_feats,
            text_tokens=text_tokens,
        )  # [B, output_size]

        encoder_hidden_states = fused_embed.unsqueeze(1).to(dtype=noisy_latents.dtype)
        # [B, 1, output_size]

        model_pred = self.unet(
            noisy_latents if noisy_latents.dtype == torch.float16
            else noisy_latents.to(dtype=torch.float16),
            timesteps,
            encoder_hidden_states,
        ).sample.float()

        return model_pred, fused_embed
