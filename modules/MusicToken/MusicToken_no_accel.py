# @title modules/MusicToken/MusicToken_no_accel.py
"""
MusicToken_no_accel.py — Core model wrapper per Musipainter (Branch FuseLIP / Strada B).
"""

import gc
import torch
import torch.nn as nn
import logging
from diffusers import AutoencoderKL, UNet2DConditionModel

from modules.BEATs.BEATs import BEATs, BEATsConfig
from modules.fusion.early_fusion_encoder import EarlyFusionEncoder

logger = logging.getLogger(__name__)


def _load_weights(path, map_device) -> dict:
    import os
    resolved = str(path)
    if not os.path.exists(resolved) and resolved.endswith(".bin"):
        sf_candidate = resolved[:-4] + ".safetensors"
        if os.path.exists(sf_candidate):
            logger.info(f"[_load_weights] .bin not found, auto-switching to: {sf_candidate}")
            resolved = sf_candidate
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"[_load_weights] Weight file not found: {resolved}")
    if resolved.endswith(".safetensors"):
        from modules.preprocess.utils import load_safetensors
        return load_safetensors(resolved, device=str(map_device))
    return torch.load(resolved, map_location=map_device)


class MusicTokenWrapper(nn.Module):
    """
    Wraps VAE, UNet, CLIP token embeddings, BEATs e EarlyFusionEncoder.

    Solo EarlyFusionEncoder (e LoRA opzionale) ha requires_grad=True.
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

        # ── CLIP token embedding (frozen, full transformer discarded) ─────────
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

        # ── BEATs audio encoder (frozen, optional) ────────────────────────────
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
        audio_dim       = 768 * 3
        output_size     = self.unet.config.cross_attention_dim
        d_model         = getattr(args, 'ef_d_model', 512)
        nhead           = getattr(args, 'ef_nhead', 8)
        num_layers      = getattr(args, 'ef_num_layers', 4)
        dropout         = getattr(args, 'ef_dropout', 0.1)
        n_audio_queries = getattr(args, 'ef_n_audio_queries', 1)

        logger.info(
            f"EarlyFusionEncoder (FuseLIP / Strada B): "
            f"n_audio_queries={n_audio_queries} (0=Full, 1=Pooling, >1=Resampler), "
            f"audio_dim={audio_dim}, text_dim={self._text_dim}, "
            f"output_size={output_size}, d_model={d_model}, "
            f"nhead={nhead}, num_layers={num_layers}, dropout={dropout}"
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
            n_audio_queries=n_audio_queries,
        )
        self.embedder = self.early_fusion  # alias per checkpoint helpers

        # ── Eval mode per frozen components ───────────────────────────────────
        self.vae.eval()
        self.unet.eval()
        self.token_embedding.eval()
        if self.aud_encoder is not None:
            self.aud_encoder.eval()

        # ── [FIX-BUG2] LoRA — parsing robusto + LoRAAttnProcessor2_0 + ordine corretto
        if hasattr(args, 'lora') and args.lora:
            from diffusers.models.attention_processor import LoRAAttnProcessor2_0, LoRAAttnProcessor
            from diffusers.loaders import AttnProcsLayers

            logger.info("--- CONFIGURAZIONE LORA (FuseLIP — fix Diffusers 0.25.1) ---")

            lora_attn_procs = {}

            # Raccoglie le chiavi dei processor attuali
            attn_proc_keys = list(self.unet.attn_processors.keys())

            if attn_proc_keys:
                logger.info(f"[LORA] Primi 3 processor names: {attn_proc_keys[:3]}")
                logger.info(f"[LORA] Totale processor keys: {len(attn_proc_keys)}")
            else:
                # Fallback: costruisce i nomi dai moduli attention dell'UNet
                logger.warning("[LORA] attn_processors.keys() vuoto — fallback su named_modules()")
                for name, module in self.unet.named_modules():
                    if hasattr(module, 'to_q') and hasattr(module, 'to_k') and hasattr(module, 'to_v'):
                        attn_proc_keys.append(name + ".processor")
                logger.info(f"[LORA] Totale keys dopo fallback: {len(attn_proc_keys)}")

            for name in attn_proc_keys:
                cross_attention_dim = (
                    None if name.endswith("attn1.processor")
                    else self.unet.config.cross_attention_dim
                )

                # [FIX-BUG2a] Parsing robusto con try/except
                hidden_size = None
                try:
                    parts = name.split(".")
                    if "down_blocks" in name:
                        idx = parts.index("down_blocks")
                        block_id = int(parts[idx + 1])
                        hidden_size = self.unet.config.block_out_channels[block_id]
                    elif "up_blocks" in name:
                        idx = parts.index("up_blocks")
                        block_id = int(parts[idx + 1])
                        hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
                    elif "mid_block" in name:
                        hidden_size = self.unet.config.block_out_channels[-1]
                    else:
                        logger.warning(f"[LORA] Blocco non standard ignorato: {name}")
                        continue
                except (ValueError, IndexError, AttributeError) as e:
                    logger.error(f"[LORA] Errore parsing {name}: {e}")
                    continue

                if hidden_size is None:
                    logger.warning(f"[LORA] hidden_size None per {name}, skip")
                    continue

                # [FIX-BUG2b] LoRAAttnProcessor2_0 con fallback a LoRAAttnProcessor
                try:
                    proc = LoRAAttnProcessor2_0(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        rank=getattr(args, 'lora_rank', 8),
                    )
                except Exception as e:
                    logger.warning(f"[LORA] LoRAAttnProcessor2_0 fallito per {name}: {e}")
                    try:
                        proc = LoRAAttnProcessor(
                            hidden_size=hidden_size,
                            cross_attention_dim=cross_attention_dim,
                            rank=getattr(args, 'lora_rank', 8),
                        )
                    except Exception as e2:
                        logger.error(f"[LORA] Anche LoRAAttnProcessor fallito per {name}: {e2}")
                        continue

                lora_attn_procs[name] = proc

            # [FIX-BUG2c] Ordine corretto: set_attn_processor → AttnProcsLayers → (poi xformers a parte)
            if lora_attn_procs:
                self.unet.set_attn_processor(lora_attn_procs)
                self.lora_layers = AttnProcsLayers(self.unet.attn_processors)
                n_params = sum(p.numel() for p in self.lora_layers.parameters())
                logger.info(
                    f"[LORA] Installati {len(lora_attn_procs)} processori "
                    f"({n_params:,} parametri). SDPA (PyTorch native): ON"
                )
            else:
                logger.error(
                    "[LORA] CRITICAL: LoRA installed on 0 processors! "
                    "Versione Diffusers incompatibile."
                )
                self.lora_layers = None

            logger.info("----------------------------------------------")

        else:
            self.lora_layers = None
            # xformers solo quando LoRA NON è attivo
            try:
                self.unet.enable_xformers_memory_efficient_attention()
                logger.info("xformers memory efficient attention: ON")
            except Exception as e:
                logger.info(f"xformers non disponibile ({e}). Fallback a PyTorch 2.0 SDPA.")
                try:
                    from diffusers.models.attention_processor import AttnProcessor2_0
                    self.unet.set_attn_processor(AttnProcessor2_0())
                    logger.info("torch SDPA (AttnProcessor2_0): ON")
                except Exception:
                    logger.info("Falling back to vanilla attention")

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
            if hasattr(args, 'lora') and args.lora and self.lora_layers is not None:
                self.lora_layers.requires_grad_(True)
                self.lora_layers.train()
                logger.info("lora_layers.train() — UNet backbone rimane frozen/eval.")

        elif args.data_set == 'test':
            # [FIX-BUG3] NON chiamare .to(weight_dtype) qui:
            # UNet e VAE sono già in fp16 grazie a from_pretrained(torch_dtype=frozen_dtype).
            # early_fusion rimane in float32 — il cast avviene nel forward() dove necessario.
            map_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.early_fusion.eval()

            _state = _load_weights(args.learned_embeds, map_device)

            # [FIX-BUG4] Pulizia prefisso _orig_mod. aggiunto da torch.compile.
            # k.startswith() NON funziona perché torch.compile genera chiavi come
            # "early_fusion._orig_mod.audio_proj.weight" (prefisso in mezzo).
            # k.replace() rimuove _orig_mod. ovunque si trovi nella stringa.
            _state = {k.replace("_orig_mod.", ""): v for k, v in _state.items()}

            missing, unexpected = self.early_fusion.load_state_dict(_state, strict=True)
            if missing:
                logger.warning(f"[CKPT] Missing keys ({len(missing)}): {missing[:5]}")
            if unexpected:
                logger.warning(f"[CKPT] Unexpected keys ({len(unexpected)}): {unexpected[:5]}")

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
                    raise RuntimeError("lora_layers is None ma --lora=True.")
                _lora_state = _load_weights(args.learned_embeds_lora, map_device)
                self.lora_layers.load_state_dict(_lora_state)
                self.lora_layers.eval()
                logger.info(f"LoRA weights loaded from: {args.learned_embeds_lora}")

    # ──────────────────────────────────────────────────────────────────────────
    def set_placeholder_token_id(self, token_id: int):
        """No-op mantenuto per compatibilità CLI."""
        pass

    # ──────────────────────────────────────────────────────────────────────────
    def _get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup CLIP token embeddings [B, seq_len] → [B, seq_len, text_dim].
        """
        with torch.no_grad():
            text_embeds = self.token_embedding(input_ids).float()
        return text_embeds

    # ──────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        audio_features: torch.Tensor,   # [B, T_a, 2304]
        input_ids:      torch.Tensor,   # [B, seq_len]
        noisy_latents:  torch.Tensor,   # [B, 4, H/8, W/8]
        timesteps:      torch.Tensor,   # [B]
    ):
        """
        Forward pass.

        LDM conditioning (Rombach et al., Sec. 3.3):
          EarlyFusionEncoder agisce come encoder τ_θ che mappa il segnale
          multimodale (audio + testo) in τ_θ(y) ∈ R^{M×d_τ} passato alle
          cross-attention layers della UNet come encoder_hidden_states.

        Returns:
            model_pred    : [B, 4, H/8, W/8]  float32
            fused_seq     : [B, actual_T_a + 1 + T_t, output_size]  float32
            audio_summary : [B, 1, output_size]  float32
                            Token PRE-transformer dell'attentive pooling,
                            proiettato via loss_proj. Usato per cosine loss.
        """
        text_tokens = self._get_text_embeddings(input_ids)  # [B, T_t, text_dim]

        audio_feats = (
            audio_features.float()
            if audio_features.dtype != torch.float32
            else audio_features
        )

        fused_seq, audio_summary = self.early_fusion(
            audio_tokens=audio_feats,
            text_tokens=text_tokens,
            return_audio_summary=True,
        )

        encoder_hidden_states = fused_seq.to(dtype=noisy_latents.dtype)

        model_pred = self.unet(
            noisy_latents if noisy_latents.dtype == torch.float16
            else noisy_latents.to(dtype=torch.float16),
            timesteps,
            encoder_hidden_states,
        ).sample.float()

        return model_pred, fused_seq, audio_summary