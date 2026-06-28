# @title modules/MusicToken/MusicToken_no_accel.py
"""
MusicToken_no_accel.py — Core model wrapper per Musipainter (VERSIONE DEFINITIVA SDPA)
"""

import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from diffusers.loaders import AttnProcsLayers
from diffusers import AutoencoderKL, UNet2DConditionModel

from modules.BEATs.BEATs import BEATs, BEATsConfig
from modules.fusion.cross_attention_encoder import FullAudioGuidedCrossAttentionEncoder

logger = logging.getLogger(__name__)

CLIP_SEQ_LEN = 77

def _load_weights(path, map_device) -> dict:
    import os
    resolved = str(path)
    if not os.path.exists(resolved) and resolved.endswith(".bin"):
        sf_candidate = resolved[:-4] + ".safetensors"
        if os.path.exists(sf_candidate):
            logger.info(f"[_load_weights] .bin not found → {sf_candidate}")
            resolved = sf_candidate
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"[_load_weights] Not found: {resolved}")
    if resolved.endswith(".safetensors"):
        from modules.preprocess.utils import load_safetensors
        return load_safetensors(resolved, device=str(map_device))
    return torch.load(resolved, map_location=map_device)


class MusicTokenWrapper(nn.Module):
    def __init__(self, args):
        super().__init__()

        frozen_dtype = torch.float16
        _hf_cache = getattr(args, 'hf_cache_dir', '/tmp/hf_model_cache')

        # ── Frozen SD components ──────────────────────────────────────────────
        self.unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet",
            revision=getattr(args, 'revision', None),
            torch_dtype=frozen_dtype, cache_dir=_hf_cache,
        )
        gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

        self.vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae",
            revision=getattr(args, 'revision', None),
            torch_dtype=frozen_dtype, cache_dir=_hf_cache,
        )
        gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # ── Full CLIP text transformer ────────────────────────────────────────
        from transformers import CLIPTextModel as _CLIPTextModel
        self.text_encoder = _CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder",
            revision=getattr(args, 'revision', None),
            torch_dtype=frozen_dtype, cache_dir=_hf_cache,
        )
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)
        self.token_embedding = self.text_encoder.text_model.embeddings.token_embedding
        self._text_dim: int = self.text_encoder.config.hidden_size  # 1024 for SD2
        gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # ── BEATs (optional) ──────────────────────────────────────────────────
        if getattr(args, 'use_precomputed_embeddings', True):
            logger.info("Pre-computed embeddings — BEATs not loaded.")
            self.aud_encoder = None
        else:
            logger.info("Loading BEATs...")
            checkpoint = torch.load(
                'models/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt',
                map_location='cpu')
            cfg = BEATsConfig(checkpoint['cfg'])
            self.aud_encoder = BEATs(cfg)
            self.aud_encoder.load_state_dict(checkpoint['model'])
            self.aud_encoder.predictor = None
            del checkpoint; gc.collect()

        # ── Dimensions ────────────────────────────────────────────────────────
        audio_dim   = 768 * 3
        output_size = self.unet.config.cross_attention_dim   # 1024 for SD2
        d_model     = getattr(args, 'ef_d_model', 512)
        nhead       = getattr(args, 'ef_nhead', 8)
        num_layers  = getattr(args, 'ef_num_layers', 4)
        dropout     = getattr(args, 'ef_dropout', 0.1)

        logger.info(
            f"FullAudioGuidedCrossAttentionEncoder: "
            f"audio_dim={audio_dim}, text_dim={self._text_dim}, "
            f"output_size={output_size}, d_model={d_model}, "
            f"nhead={nhead}, num_layers={num_layers}, dropout={dropout}"
        )

        # ── Cross-attention encoder (trainable) ───────────────────────────────
        self.early_fusion = FullAudioGuidedCrossAttentionEncoder(
            audio_dim=audio_dim, text_dim=self._text_dim,
            output_size=output_size, d_model=d_model,
            nhead=nhead, num_layers=num_layers, dropout=dropout,
            max_audio_len=1024,
        )
        self.embedder = self.early_fusion

        # ── Sequence resampler T_a → 77 (trainable) ───────────────────────────
        self.seq_resampler   = nn.Linear(output_size, output_size)
        self.resampler_queries = nn.Parameter(torch.randn(CLIP_SEQ_LEN, output_size) * 0.02)
        self.resampler_attn  = nn.MultiheadAttention(
            embed_dim=output_size, num_heads=8, dropout=dropout, batch_first=True)
        self.resampler_norm  = nn.LayerNorm(output_size)
        self.loss_proj_head  = nn.Linear(output_size, output_size)

        # ── Frozen components eval ────────────────────────────────────────────
        self.vae.eval(); self.unet.eval(); self.text_encoder.eval()
        if self.aud_encoder is not None:
            self.aud_encoder.eval()

        # ── CONFIGURAZIONE LORA ROBUSTA (SDPA) ──────────────────────────────
        if getattr(args, 'lora', False):
            from diffusers.models.attention_processor import LoRAAttnProcessor2_0, LoRAAttnProcessor
            from diffusers.loaders import AttnProcsLayers

            logger.info("--- CONFIGURAZIONE LORA (Diffusers 0.25.1) ---")

            lora_attn_procs = {}

            # FIX: Ottieni i processor attuali. In Diffusers 0.25.1, attn_processors
            # può restituire un dict vuoto se non ci sono processor custom.
            # Usiamo un approccio ibrido: prima proviamo con attn_processors.keys(),
            # se vuoto, iteriamo sui moduli direttamente.
            attn_proc_keys = list(self.unet.attn_processors.keys())

            # DEBUG: logga i primi nomi trovati
            if attn_proc_keys:
                logger.info(f"[LORA-DEBUG] Primi 3 processor names: {attn_proc_keys[:3]}")
                logger.info(f"[LORA-DEBUG] Totale processor keys: {len(attn_proc_keys)}")
            else:
                logger.warning("[LORA-DEBUG] attn_processors.keys() è vuoto! Provo fallback...")

            # Se il dict è vuoto, costruisci i nomi dai moduli attention dell'UNet
            if not attn_proc_keys:
                for name, module in self.unet.named_modules():
                    # Cerca moduli attention che hanno processor
                    if hasattr(module, 'to_q') and hasattr(module, 'to_k') and hasattr(module, 'to_v'):
                        processor_name = name + ".processor"
                        attn_proc_keys.append(processor_name)
                        logger.info(f"[LORA-DEBUG] Trovato modulo attention: {processor_name}")
                logger.info(f"[LORA-DEBUG] Totale keys dopo fallback: {len(attn_proc_keys)}")

            for name in attn_proc_keys:
                # Determina cross_attention_dim
                cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim

                # FIX: Parsing più robusto del nome per hidden_size
                hidden_size = None
                try:
                    parts = name.split(".")
                    if "down_blocks" in name:
                        # Trova l'indice numerico dopo "down_blocks"
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
                        logger.warning(f"[LORA] Ignorato blocco non standard: {name}")
                        continue
                except (ValueError, IndexError, AttributeError) as e:
                    logger.error(f"[LORA] Errore nel parsing di {name}: {e}")
                    continue

                if hidden_size is None:
                    logger.warning(f"[LORA] hidden_size None per {name}, skip")
                    continue

                # FIX: Prova LoRAAttnProcessor2_0, fallback a LoRAAttnProcessor
                try:
                    proc = LoRAAttnProcessor2_0(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        rank=getattr(args, 'lora_rank', 8)
                    )
                except Exception as e:
                    logger.warning(f"[LORA] LoRAAttnProcessor2_0 fallito per {name}: {e}")
                    try:
                        proc = LoRAAttnProcessor(
                            hidden_size=hidden_size,
                            cross_attention_dim=cross_attention_dim,
                            rank=getattr(args, 'lora_rank', 8)
                        )
                    except Exception as e2:
                        logger.error(f"[LORA] Anche LoRAAttnProcessor fallito per {name}: {e2}")
                        continue

                lora_attn_procs[name] = proc

            # Applica i processor LoRA
            if lora_attn_procs:
                self.unet.set_attn_processor(lora_attn_procs)
                self.lora_layers = AttnProcsLayers(self.unet.attn_processors)

                n_lora = len(lora_attn_procs)
                n_params = sum(p.numel() for p in self.lora_layers.parameters())
                logger.info(f"Installati {n_lora} processori LoRA ({n_params:,} parametri).")
                logger.info("SDPA (PyTorch native): ON — xformers skipped to preserve LoRA")
            else:
                logger.error("CRITICAL: LoRA installed on 0 processors! Diffusers version may be incompatible.")
                self.lora_layers = None

            logger.info("----------------------------------------------")

        else:
            self.lora_layers = None
            try:
                import xformers
                self.unet.enable_xformers_memory_efficient_attention()
                logger.info("xformers memory efficient attention: ON")
            except Exception as e:
                logger.info(f"xformers non abilitato ({e}). Falling back to PyTorch 2.0 SDPA.")
                try:
                    from diffusers.models.attention_processor import AttnProcessor2_0
                    self.unet.set_attn_processor(AttnProcessor2_0())
                except Exception:
                    logger.info("Falling back to vanilla attention")

        # ── Training vs test ──────────────────────────────────────────────────
        if args.data_set == 'train':
            self.vae.requires_grad_(False)
            self.unet.requires_grad_(False)
            self.unet.enable_gradient_checkpointing()
            self.text_encoder.requires_grad_(False)
            if self.aud_encoder is not None:
                self.aud_encoder.requires_grad_(False)
            self.early_fusion.requires_grad_(True);    self.early_fusion.train()
            self.seq_resampler.requires_grad_(True)
            self.resampler_queries.requires_grad_(True)
            self.resampler_attn.train()
            self.resampler_norm.train()
            self.loss_proj_head.requires_grad_(True)
            if getattr(args, 'lora', False) and self.lora_layers is not None:
                self.lora_layers.requires_grad_(True)
                self.lora_layers.train()

        elif args.data_set == 'test':
            map_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.early_fusion.eval(); self.resampler_attn.eval()

            _state = _load_weights(args.learned_embeds, map_device)
            _state = {k.replace("_orig_mod.", ""): v for k, v in _state.items()}

            missing, unexpected = self.load_trainable_state_dict(_state)
            if missing:
                logger.warning(f"[CKPT] Missing keys ({len(missing)}): {missing[:3]}...")
            if unexpected:
                logger.warning(f"[CKPT] Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")

            if getattr(args, 'vae', False):
                self.vae.load_state_dict(_load_weights(args.learned_vae, map_device))
            if getattr(args, 'aud_encoder', False) and self.aud_encoder is not None:
                self.aud_encoder.load_state_dict(_load_weights(args.learned_aud_encoder, map_device))
            if getattr(args, 'unet', False):
                self.unet.load_state_dict(_load_weights(args.learned_unet, map_device))
            if getattr(args, 'lora', False):
                if self.lora_layers is None:
                    raise RuntimeError("lora_layers is None but --lora=True.")
                self.lora_layers.load_state_dict(_load_weights(args.learned_embeds_lora, map_device))
                self.lora_layers.eval()
                logger.info(f"LoRA weights loaded from: {args.learned_embeds_lora}")

    # ── State dict helpers ────────────────────────────────────────────────────

    def trainable_state_dict(self) -> dict:
        state = {}
        state.update({f"early_fusion.{k}": v
                      for k, v in self.early_fusion.state_dict().items()})
        state.update({f"seq_resampler.{k}": v
                      for k, v in self.seq_resampler.state_dict().items()})
        state["resampler_queries"] = self.resampler_queries.data
        state.update({f"resampler_attn.{k}": v
                      for k, v in self.resampler_attn.state_dict().items()})
        state.update({f"resampler_norm.{k}": v
                      for k, v in self.resampler_norm.state_dict().items()})
        state.update({f"loss_proj_head.{k}": v
                      for k, v in self.loss_proj_head.state_dict().items()})
        return state

    def load_trainable_state_dict(self, state_dict: dict):
        TRAINABLE_PREFIXES = (
            "early_fusion.", "seq_resampler.", "resampler_attn.",
            "resampler_norm.", "loss_proj_head.", "resampler_queries",
        )
        FROZEN_PREFIXES = ("unet.", "vae.", "text_encoder.", "aud_encoder.")

        state_dict = {
            k: v for k, v in state_dict.items()
            if not any(k.startswith(fp) for fp in FROZEN_PREFIXES)
        }

        has_prefix = any(
            any(k.startswith(p) for p in TRAINABLE_PREFIXES)
            for k in state_dict
        )

        if has_prefix:
            own_params = dict(self.named_parameters())
            own_params["resampler_queries"] = self.resampler_queries

            missing, unexpected = [], []
            for name, param in state_dict.items():
                if name in own_params:
                    try:
                        own_params[name].data.copy_(param.data
                            if isinstance(param, nn.Parameter) else param)
                    except Exception as e:
                        logger.warning(f"[CKPT] Shape mismatch for {name}: {e}")
                        unexpected.append(name)
                else:
                    unexpected.append(name)

            for name in own_params:
                if name not in state_dict:
                    missing.append(name)
            return missing, unexpected

        else:
            logger.info("[CKPT] Legacy checkpoint detected (early_fusion keys only).")
            missing, unexpected = self.early_fusion.load_state_dict(state_dict, strict=False)
            return list(missing), list(unexpected)

    # ── Forward helpers ───────────────────────────────────────────────────────

    def set_placeholder_token_id(self, token_id: int):
        pass

    def _get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.text_encoder(
                input_ids=input_ids, attention_mask=None,
                output_hidden_states=False, return_dict=True,
            )
            return out.last_hidden_state.float()

    def _resample_to_77(self, seq: torch.Tensor) -> torch.Tensor:
        B, T_a, D = seq.shape
        if T_a == CLIP_SEQ_LEN:
            return seq
        if T_a <= 150:
            resampled = F.adaptive_avg_pool1d(
                seq.transpose(1, 2), CLIP_SEQ_LEN
            ).transpose(1, 2)
        else:
            queries = self.resampler_queries.unsqueeze(0).expand(B, -1, -1)
            resampled, _ = self.resampler_attn(query=queries, key=seq, value=seq)
        return self.resampler_norm(resampled)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        audio_features: torch.Tensor,   # [B, T_a, 2304]
        input_ids:      torch.Tensor,   # [B, 77]
        noisy_latents:  torch.Tensor,   # [B, 4, H/8, W/8]
        timesteps:      torch.Tensor,   # [B]
    ):
        text_tokens = self._get_text_embeddings(input_ids)

        audio_feats = audio_features.float() if audio_features.dtype != torch.float32 \
                      else audio_features

        fused_seq, audio_summary = self.early_fusion(
            audio_tokens=audio_feats, text_tokens=text_tokens,
            return_audio_summary=True,
        )

        fused_77 = self._resample_to_77(fused_seq)

        model_pred = self.unet(
            noisy_latents.to(dtype=torch.float16),
            timesteps,
            fused_77.to(dtype=noisy_latents.dtype),
        ).sample.float()

        return model_pred, fused_seq, audio_summary