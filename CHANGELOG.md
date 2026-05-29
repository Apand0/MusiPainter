# CHANGELOG — MusiPainter Pipeline

This file provides a comprehensive documentation of the architecture, fixes,
optimizations, and evolution of every module in the MusiPainter pipeline, from
the Late Fusion version (v8) to the currently-in-production Early Fusion version (v9).

---

## Table of Contents

- [Architectural Overview](#architectural-overview)
- [modules/MusicToken/MusicToken_no_accel.py](#modulesmusictokenmusictoken_no_accelpy)
- [modules/fusion/early_fusion_encoder.py](#modulesfusionearly_fusion_encoderpy)
- [modules/MusicToken/embedder.py](#modulesmustokenbedderpy)
- [modules/clip_text_model/modeling_clip.py](#modulesclip_text_modelmodeling_clippy)
- [modules/BEATs/BEATs.py](#modulesbeatsbeatsspy)
- [modules/BEATs/Tokenizers.py](#modulesbeatsTokenizerspy)
- [modules/BEATs/backbone.py](#modulesbeatsbackbonepy)
- [modules/BEATs/quantizer.py](#modulesbeatsquantizerpy)
- [modules/BEATs/modules.py](#modulesbeatsmodulespy)
- [modules/FGA/fga_model.py and atten.py](#modulesfgafga_modelpy-and-attenpy)
- [modules/preprocess/preprocess_audio_embeddings_colab.py](#modulespreprocesspreprocess_audio_embeddings_colabpy)
- [modules/preprocess/preprocess_manager.py](#modulespreprocesspreprocess_managerpy)
- [modules/preprocess/preprocess_image_latents_colab.py](#modulespreprocesspreprocess_image_latents_colabpy)
- [modules/preprocess/utils.py](#modulespreprocessutilspy)
- [modules/preprocess/argparse_multiembedding.py](#modulespreprocessargparse_multiembeddingpy)
- [dataloader_colab.py](#dataloader_colabpy)
- [train_validation_no_accel_colab.py](#train_validation_no_accel_colabpy)
- [test_no_accel_colab.py](#test_no_accel_colabpy)
- [Weight Format Migration: .bin → .safetensors](#weight-format-migration)
- [Architectural Evolution: Late Fusion v8 → Early Fusion v9](#architectural-evolution)

---

## Architectural Overview

MusiPainter generates artistic images from audio tracks by conditioning
Stable Diffusion with audio embeddings extracted from BEATs. The main flow is:

```
Audio WAV
    │
    ▼
[BEATs encoder] (frozen) — layers 4+8+12 concatenated → dim 2304
    │
    ▼
[B, T_a, 2304]  (precomputed on disk, float16)
    │
    ├──────────────────────────────────────────────┐
    │                                              │
    ▼                                              ▼
[audio_tokens]                        [CLIP token_embedding] (frozen)
                                      [B, T_t, 768] ← input_ids
    └──────────────────┬───────────────────────────┘
                       ▼
            [EarlyFusionEncoder] (trainable)
                       │
                       ▼
            [B, T_a+T_t, output_size]  fused_seq
                       │
                       ▼
            [UNet2DConditionModel] (frozen, + optional LoRA)
            encoder_hidden_states = fused_seq
                       │
                       ▼
                [VAE decoder]
                       │
                       ▼
               Generated image
```

**Main dependencies:** torch==2.5.1, diffusers==0.25.1, transformers==4.38.2,
safetensors, BEATs (Microsoft), Stable Diffusion 2.1 (stabilityai).

---

## modules/MusicToken/MusicToken_no_accel.py

### Description

This is the central wrapper of the entire system. `MusicTokenWrapper` is an `nn.Module`
that contains all frozen components and the trainable module, and implements
the complete forward-pass logic for both training and inference.

### Frozen components (float16, requires_grad=False)

- `self.vae`: `AutoencoderKL` — encodes images into latents and decodes them back.
  Scaling factor 0.18215 applied during encoding.
- `self.unet`: `UNet2DConditionModel` — main denoiser. Receives
  `encoder_hidden_states` as the fused sequence `[B, T_a+T_t, output_size]`.
- `self.token_embedding`: only the `nn.Embedding` table extracted from
  `CLIPTextModel`, which is then discarded (~400 MB VRAM saved compared
  to keeping the full CLIP Transformer).
- `self.aud_encoder`: BEATs — loaded only if `use_precomputed_embeddings=False`.

### Trainable component (float32)

- `self.early_fusion`: `EarlyFusionEncoder` — the only module with
  `requires_grad=True`.
- `self.embedder`: alias of `self.early_fusion` for compatibility with
  helper checkpoints.
- `self.lora_layers`: `AttnProcsLayers` on the UNet, optional.

### Key methods

**`_get_text_embeddings(input_ids)`** — lookup `[B, seq_len] → [B, seq_len, text_dim]`
in the frozen CLIP table, without passing through the CLIP Transformer. Returns float32.

**`forward(audio_features, input_ids, noisy_latents, timesteps)`**:
1. Calls `_get_text_embeddings` → `text_tokens [B, T_t, 768]`
2. Calls `early_fusion(audio_tokens, text_tokens)` → `fused_seq [B, T_a+T_t, output_size]`
3. Passes `fused_seq` directly as `encoder_hidden_states` to the UNet
4. Returns `(model_pred [B,4,H/8,W/8], fused_seq)` — the second is needed
   by the training loop for auxiliary losses

**`set_placeholder_token_id(token_id)`** — no-op kept for CLI compatibility
with the old Late Fusion architecture.

### LoRA management

`LoRAAttnProcessor` modules are installed on the UNet attention layers **before**
any xformers/AttnProcessor2_0 fallback. During training only
`lora_layers.train()` is called; the UNet backbone remains in `eval()`.

### Fixes and optimizations

**[FIX-DP]** Frozen components loaded in float16 on CPU; `EarlyFusionEncoder`
initialized in float32 (PyTorch default). DataParallel/DDP requires all
parameters on the same device at wrapping time. Solution: `__init__`
does not call `.to(device)`; the training loop does it after DDP wrapping. The VAE
is accessed via `base_model.vae` to avoid "buffer registered on cuda:0 but found on cpu" errors.

**[OPT-UNET-NOGRAD]** The UNet has 859M frozen parameters. The forward passes
`encoder_hidden_states` (requires_grad=True) directly to the UNet, keeping the graph
only through that tensor — a narrower and cheaper path compared to the full graph.

**[OPT-HF-CACHE-TMP]** HF cache redirected to `/tmp` (RAM tmpfs on Kaggle)
to avoid disk quota issues.

**[LORA-INIT]** `LoRAAttnProcessor` installed on the UNet before the xformers/AttnProcessor2_0 fallback.

**[LORA-TRAIN]** Only `lora_layers.train()` is called, not `unet.train()`.

**[LORA-TEST]** `AttnProcsLayers` is a direct view of the UNet processor weights. `load_state_dict()` updates them in-place.

**[FIX-COMPILE-LOAD]** `torch.compile()` adds the `_orig_mod.` prefix to
parameter names in the `state_dict`. It is removed during loading so that compiled
and non-compiled checkpoints are interchangeable.

---

## modules/fusion/early_fusion_encoder.py

### Description

Implements early fusion in the style of FuseLIP. It is the main trainable module,
introduced with the Early Fusion (v9) migration. Replaces `FGAEmbedder`.

### Internal architecture

```
audio_tokens [B, T_a, 2304]
    → Linear(2304, d_model)
    → + modality_emb_audio (parameter [1,1,d_model])
    → + audio_pos_emb(arange(T_a))

text_tokens [B, T_t, 768]
    → Linear(768, d_model)
    → + modality_emb_text (parameter [1,1,d_model])
    → + text_pos_emb(arange(T_t))

cat(dim=1) → fused [B, T_a+T_t, d_model]
    ↓
TransformerEncoder — Pre-LN (norm_first=True)
  num_layers × TransformerEncoderLayer(
    d_model, nhead, ffn=d_model×4, gelu, batch_first=True
  )
    ↓
LayerNorm(d_model) → Linear(d_model, output_size)
    ↓
[B, T_a+T_t, output_size]  ← full sequence returned
```

### Design properties

**Modality embeddings** — scalar parameters `[1,1,d_model]` learned separately
for audio and text, allowing the Transformer to distinguish the modality of each
token at every layer.

**Separate positional embeddings** — distinct `nn.Embedding` tables for audio
(`max_audio_len=512`, covers stride=1 with ~376 frames) and text (`max_text_len=77`).
Audio tokens come first in the sequence so their positional indices are stable
regardless of text length.

**Pre-LN (norm_first=True)** — numerically more stable with mixed-modality inputs
compared to standard Post-LN.

**Full sequence return** — unlike the previous version which returned a single
pooled vector `[B, D]`, it now returns `[B, T_a+T_t, output_size]`. This allows the
UNet to attend to different tokens for different spatial regions of the image, fully
exploiting the original cross-attention mechanism of LDM.

**Masking** — `src_key_padding_mask` to handle padding for both audio and text.
Padding positions are zeroed out in the output before returning the sequence.

**Stride compatibility** — stride=1 → T_a≈376 (total ~453), stride=4 → T_a≈94
(total ~171), stride=8 → T_a≈47 (total ~124), stride=16 → T_a≈23 (total ~100).
SD accepts any sequence length, not just 77.

### [EARLY-FUSION-ENCODER]

Architecture that replaces FGAEmbedder. Every audio token can attend to every
text token at every layer of the Transformer (Early Fusion), unlike the original
Late Fusion which injected a single placeholder token after CLIP had already
processed the text independently.

---

## modules/MusicToken/embedder.py

### Description

Implemented the audio projection in the old Late Fusion architecture.
**No longer imported by `MusicTokenWrapper`** since the introduction of Early Fusion.
Kept as reference.

### Architecture

```
[B, T, 2304]
    → Linear(2304, 2304) → GELU → Linear(2304, output_size)
    → [B, T, output_size]

Attentive Pooling:
    → Linear(output_size, output_size//2) → Tanh → Linear(→1)
    → Softmax(dim=T)   (sums to 1 over T)
    → weighted sum → [B, output_size]
```

`output_size = 768` for SD v1-4, `output_size = 1024` for SD 2.1.

### [FIX-EMBEDDER]

The old embedder used the `Atten` module from `modules.FGA.atten`, designed
for multi-modal VQA. With a single audio modality and `pairwise_flag=False`,
the output shape was not guaranteed to be `[B, 768]` — it could return
`[B, T, 768]` with the temporal dimension still present, breaking the injection
into the CLIP placeholder. Standard Attentive Pooling completely solves the
problem and matches the architecture described in the paper (Section 3.3, Fig. 3).

---

## modules/clip_text_model/modeling_clip.py

### Description

Custom version of the HuggingFace `CLIPTextModel` which, in the old Late Fusion
architecture, injected the audio embedding in place of the `<*>` placeholder token.
**No longer imported by `MusicTokenWrapper`** in the Early Fusion architecture.
Kept as reference.

### Changes from the original HuggingFace

**`CLIPTextEmbeddings.forward()` — differentiable injection:**

Uses explicit `placeholder_token_id` instead of `input_ids.max()`, which caused
erroneous injection when other tokens (e.g. "1960" in "20th century") had higher
IDs than the placeholder.

Instead of an in-place assignment `inputs_embeds[indices] = audio_e` (which
would break the autograd chain because `base_embeds.requires_grad=False`),
it uses a linear combination:

```python
inputs_embeds = base_embeds * (1 - mask) + audio_expanded * mask
```

This preserves the graph: loss → encoder_hidden_states → inputs_embeds →
audio_expanded → audio_e → FGAEmbedder. Full chain.

Saves `_last_input_ids_for_pooling` with the placeholder lowered to 5 for
correct EOS pooling (argmax must not confuse placeholder with EOS).

**`CLIPTextModel.forward()` — dtype guard:**

If the first argument is an integer tensor (as when the Diffusers pipeline
calls `text_encoder(input_ids, ...)` positionally), it is reinterpreted as
`input_ids` and `audio_e` is set to `None`. This guarantees backward-compatibility
with `StableDiffusionPipeline`.

### Applied fixes

**[FIX-PLACEHOLDER]** Replaced `input_ids.max()` with explicit `placeholder_token_id`
set via `set_placeholder_token_id()`.

**[FIX-GRAD]** Differentiable linear combination instead of in-place assignment
to preserve the autograd graph through the frozen embeddings.

**[OPT-CLONE]** Single `input_ids.clone()` at the beginning instead of multiple clones.

**[FIX-EOS]** EOS pooling uses the saved copy of `input_ids` with the placeholder
lowered to 5, preventing argmax() from finding the placeholder instead of EOS.

**[FIX-COMPAT]** Dtype guard on `audio_e` for backward-compatibility with
the positional call of `StableDiffusionPipeline`.

---

## modules/BEATs/BEATs.py

### Description

Implements the BEATs model (Audio Pre-Training with Acoustic Tokenizers,
Microsoft Research). Used as a frozen audio encoder to extract semantic features
from WAV tracks.

### Feature extraction pipeline

1. **`preprocess()`** — waveform → Mel filterbank (128 bins, 16kHz, 25ms frame,
   10ms shift), normalization with mean 15.41663 and std 6.55582×2.
2. **`patch_embedding`** — Conv2D that discretizes the spectrogram into patches
   of size `input_patch_size`.
3. **`TransformerEncoder`** — 12 layers, `encoder_embed_dim=768`.
4. **`extract_features()`** — returns `(x, layers_cat, layers)` where
   `layers_cat` is the **concatenation** of layers 4, 8, 12 → dimension `768×3=2304`.

The concatenation of layers instead of the mean (`[FIX-2304]`) captures features
at different levels of abstraction: local (layer 4), semi-global (layer 8),
global (layer 12).

When `cfg.finetuned_model=True`, it adds a linear predictor for classification.
In MusiPainter `predictor=None` is explicitly set after checkpoint loading.

---

## modules/BEATs/Tokenizers.py

### Description

Variant of BEATs that includes a vector quantizer to produce discrete tokens.
Used for BEATs pre-training with the acoustic token prediction objective.
Not used directly in the MusiPainter pipeline, which only uses `BEATs.extract_features()`.

### Differences from BEATs.py

Adds:
- `NormEMAVectorQuantizer` — codebook with 1024 entries of dimension 256,
  updated with L2-normalized EMA.
- `quantize_layer` — `Linear(768) → Tanh → Linear(256)` to project features
  into the codebook space.
- **`extract_labels()`** — returns the discrete codebook indices (`embed_ind`)
  instead of continuous features.

---

## modules/BEATs/backbone.py

### Description

Implements the BEATs Transformer backbone with `TransformerEncoder`,
`TransformerSentenceEncoderLayer`, and specialized `MultiheadAttention`.

### TransformerEncoder

- **Convolutional positional encoding** — `pos_conv`: Conv1D with weight_norm,
  `conv_pos_groups` groups, SamePad padding, GELU activation.
- **Layer drop** — during training, each layer is skipped with probability
  `encoder_layerdrop`.
- **Layer-wise gradient decay** — `GradMultiply.apply(x, ratio)` scales
  gradients per layer, useful for stable fine-tuning.
- **Relative position embedding** — T5-style with distance buckets, optional.
  Weights are shared across all layers via direct assignment.
- **Layer 4, 8, 12 collection** — `extract_features()` saves hidden states
  at indices 3, 7, 11 in `layers[]` and concatenates them into `layers_cat [B,T,2304]`.

### MultiheadAttention

- **Scaling with alpha=32** — `q *= 1/alpha` before the dot product, then
  `attn_weights * alpha` after, for numerical stability with large embeddings.
- **GRU relative positional bias** — learned gate on query to modulate the relative
  positional bias, optional (`gru_rel_pos`).
- **`quant_noise` wrapper** — adds quantization noise to weights during training
  for post-training quantization robustness.
- **`incremental_state` support** — for autoregressive decoding with key and value caching.

### TransformerSentenceEncoderLayer

Supports both Pre-LN (`layer_norm_first=True`) and Post-LN, Deep Norm
(`deep_norm_alpha = (2N)^(1/4)`, scales residuals), and GLU activation.

---

## modules/BEATs/quantizer.py

### Description

Implements `NormEMAVectorQuantizer` with `EmbeddingEMA`. Used in BEATs Tokenizers pre-training.

### Main mechanisms

**L2 normalization** — input and codebook vectors are normalized before distance
computation. `l2norm()` = `F.normalize(t, p=2, dim=-1)`.

**EMA codebook update** — centroids are updated with normalized EMA (`norm_ema_inplace`)
instead of backpropagation, for stability during the quantization phase.

**K-means initialization** — if `kmeans_init=True`, the codebook is initialized
with k-means on the first batches of data.

**Straight-Through Estimator (STE)** — `z_q = z + (z_q - z).detach()`
preserves gradients through the discrete lookup operation: the gradient passes
as if `z_q = z`.

**Commitment loss** — `beta * MSE(z_q.detach(), z)` pushes input vectors toward
the codebook centroids.

**DDP synchronization** — `all_reduce` on `bins` and `embed_sum` when
`distributed.is_initialized()` for consistent EMA updates across GPUs.

---

## modules/BEATs/modules.py

### Description

Support modules for the BEATs backbone.

**`GradMultiply`** — `torch.autograd.Function` that scales gradients by a
factor `scale` during backward (forward = identity). Used for layer-wise gradient decay.

**`SamePad`** — padding for Conv1D that guarantees the output has the same length
as the input. Causal mode (`causal=True`) removes `kernel_size-1` elements from the end.

**`GLU_Linear`** — Gated Linear Unit: `Linear(input, output*2)` followed by split and gating.
Supports sigmoid, swish, relu, gelu, bilinear activations.

**`quant_noise`** — `register_forward_pre_hook` that during training randomly masks
blocks of weights with probability `p`, scaling the remaining ones by `1/(1-p)`.
Supports Linear, Embedding, Conv2D.

**Activation functions** — `gelu`, `gelu_accurate` (numerical approximation),
`get_activation_fn()` for selection from string.

---

## modules/FGA/fga_model.py and atten.py

### Note

Modules from the original VQA research that inspired the "FGA" naming in MusiPainter.
**Not imported in the current pipeline.** Kept for reference.

`FGA` implements a multi-modal model for Visual Question Answering with LSTM
for questions, answers, conversational history, and captions, plus multi-factor
cross-modal attention. `Atten` implements factorial graph attention with unary,
pairwise, and prior potentials, with support for weight sharing between repeated modules.

---

## modules/preprocess/preprocess_audio_embeddings_colab.py

### Description

Pipeline to pre-compute all BEATs embeddings and save them to disk before training.
Designed to run on Kaggle with a ~20 GB disk quota.

### Main flow

1. **Recursive scan** of `.wav` files under `audio_dir`
2. **Parallel loading** with `ThreadPoolExecutor(io_workers)` — overlaps
   disk I/O and GPU compute
3. **Resample** to 16kHz mono, trim/pad to `duration_seconds` seconds (default 30)
4. **Batch encoding** with BEATs — `extract_features()` returns `layers_cat [B,T,2304]`
5. **`temporal_pool(features, stride)`** — non-overlapping average-pool of
   `stride` consecutive frames: `[T, 2304] → [T//stride, 2304]`. stride=1 is a
   no-op; stride=8 reduces ~375 frames to ~47
6. **Periodic flush** every `FLUSH_EVERY_N_BATCHES=150` batches into
   safetensors chunks per class in `chunks/<class>_chunk_NNNN.safetensors`
7. **Automatic resume** — reads only the headers of existing chunks (O(1),
   without loading tensors) to determine already-encoded IDs
8. **Automatic stop** when free space drops below `critical_free_gb`
   (default 2 GB), printing resume instructions to screen
9. **Streaming final merge** — deletes each source chunk immediately after reading,
   so the extra space needed is only the largest chunk + 200 MB

### Output structure

```
output_dir/
  chunks/
    Train_Hard_Rock_chunk_0000.safetensors
    Train_Jazz_chunk_0000.safetensors
    ...
  audio_embeddings.safetensors   (after merge, optional)
  metadata.json
```

### Fixes and optimizations

**[MEM-1]** No RAM accumulation — flush every N batches.

**[MEM-2]** Separate chunks per class — never overwritten.

**[MEM-3]** O(1) resume — reads only chunk headers.

**[MEM-4]** Streaming merge — maximum extra space = largest chunk.

**[SPEED-1]** `gc.collect()` + `torch.cuda.empty_cache()` after every flush.

**[SPEED-2]** DataParallel BEATs on multi-GPU — `_BEATsDP` wrapper for
DataParallel compatibility with the `extract_features()` signature.

**[SPEED-3]** Parallel I/O via `ThreadPoolExecutor`.

**[FIX-2304]** Full concatenation of layers 4+8+12 → 2304-D instead of the mean.
Matches the input expected by `EarlyFusionEncoder`.

**[FIX-OOM]** OOM during encoding: halve the batch and retry. If even batch_size=1
OOMs, skip the batch with an error log.

---

## modules/preprocess/preprocess_manager.py

### Description

High-level orchestrator for audio preprocessing on Kaggle.
Supports encoding at multiple `temporal_pool_stride` values in a single session
and granular disk space management.

### Features

**[KAGGLE-DISK]** Checks free space before every class batch. Stops automatically
when it drops below `critical_free_gb` (default 2 GB), printing complete
upload and resume instructions.

**[KAGGLE-CLASS-BATCH]** Processes audio classes in batches of `class_batch_size`
(default 5). Between batches: disk check, `gc.collect()`. This provides clean
stop points before running out of disk.

**[KAGGLE-STRIDE-SWEEP]** `--strides` accepts a list of stride values.
Each stride gets its own `stride<N>/` subdirectory. Allows precomputing multiple
levels of temporal compression in one session.

**[KAGGLE-RESUME]** `--existing_datasets` accepts comma-separated Kaggle paths from
previous runs. Already-encoded audio IDs are automatically skipped, enabling incremental
preprocessing across sessions.

**[KAGGLE-SKIP-MERGE]** `--skip_merge` avoids the final merge, saving the space of
the largest chunk. The dataloader reads class chunks directly.

**[KAGGLE-INSTRUCTIONS]** `_print_upload_instructions()` generates the complete
resume command with all pre-compiled flags, including the path of the new dataset.

---

## modules/preprocess/preprocess_image_latents_colab.py

### Description

Analogous to `preprocess_audio_embeddings_colab.py` but for VAE latents of
images. Pre-encodes all images in the dataset as float16 latents `[4, H/8, W/8]`
to eliminate the VAE bottleneck during training.

### Flow

- PIL images → optional center crop → bicubic resize → normalization `[-1, 1]`
- VAE encode with `autocast` float16 → `latent * 0.18215` → float16
- `DataLoader` with custom `collate_fn` that skips corrupted images
- DataParallel VAE on multi-GPU via `VAEDataParallelWrapper`
- Periodic safetensors chunks, identical streaming final merge as the audio pipeline

### Fixes and optimizations

**[MEM-1/2/3/4]** Same chunked strategy as the audio pipeline.

**[OPT-1]** DataParallel VAE on multi-GPU.

**[OPT-2]** CPU workers for parallel PIL decode.

**[OPT-3]** `pin_memory=True` when `num_workers > 0`.

**[OPT-4]** `autocast` float16 for VAE encoding.

**[FIX-OOMAUTO]** OOM: halve batch, retry, skip batch on unrecoverable failure.

---

## modules/preprocess/utils.py

### Description

Helpers for tensor serialization. Two levels:

### Safetensors (preferred format)

**`save_safetensors(tensors, path, metadata)`** — saves with explicit `.contiguous()`
and `fsync` for GCS/Kaggle filesystems. Optional str→str metadata.

**`open_safetensors(path)`** — opens in lazy mmap mode via `safe_open`.
Returns a handle with `.keys()`, `.get_tensor(key)`, `.metadata()`.
Not a context manager: must be kept as an attribute.

**`load_safetensors(path, device)`** — loads everything into memory. Pickle-free,
zero-copy, lazy-capable. Preferred for weight files.

### Legacy torch.save (backward compat)

**`save_chunk_safe(chunk, path)`** — `torch.save` with `fsync`.

**`safe_torch_load(path, map_location)`** — with fallback from `weights_only=True`
to `False` for compatibility with older checkpoints.

---

## modules/preprocess/argparse_multiembedding.py

### Description

Provides `LazyEmbeddingIndex`, a unified index over N directories of precomputed
embeddings, and `add_multiembedding_args()` to register CLI flags.

### LazyEmbeddingIndex

**Automatically detected layouts (in priority order):**
1. `<dir>/audio_embeddings.safetensors` — single merged file
2. `<dir>/chunks/*.safetensors` — per-class chunks
3. `<dir>/*.safetensors` — generic
4. `<dir>/audio_embeddings_*.pt` — legacy

**Preload mode** (`preload_all=True`) — all tensors loaded into RAM at
`_build_index()`. O(1) access via `_flat_cache`.

**Lazy mmap mode** (`preload_all=False`) — LRU cache of `max_sf_handles`
`safe_open` file handles. When the cache is full, the oldest handle is evicted.
For legacy `.pt` files, LRU cache of `max_cached_chunks=4` fully loaded dictionaries.

**Deduplication** — duplicate IDs across different directories: last-write-wins.

### CLI flags registered by add_multiembedding_args()

- `--embeddings_dir` — comma-separated paths to embedding directories
- `--embeddings_preload_all` — bool: RAM preload or lazy mmap
- `--embeddings_max_sf_handles` — LRU cache size for handles

### [MULTIEMB-BOOL]

`_str2bool()` accepts `'yes/true/1'` and `'no/false/0'` for flexible CLI parsing.

---

## dataloader_colab.py

### Description

Main PyTorch Dataset (`Museart`) and latent index (`LazyLatentIndex`) for the
MusiPainter pipeline.

### LazyLatentIndex

Provides access to precomputed VAE latents `[4, H/8, W/8]` float16.
Prefers `image_latents.safetensors`; fallback reads legacy `.pt` chunks with
an LRU cache of 4 chunks.

In `preload_all=True` mode, copies all tensors into `_flat_cache` and releases
the `safe_open` handle. In lazy mode, keeps the handle open for mmap access.

### Museart Dataset

Each sample pairs a WAV file with a WikiArt image of the same class.

**Initialization:**
- Single filesystem walk with `followlinks=True`, populates
  `audio_id_to_path` and `image_id_to_path`
- `prepare_dataset()`: join CSV audio ↔ images per class
- Pre-tokenization of all templates in `__init__` →
  `_cached_input_ids` (list of contiguous tensors) — avoids tokenization overhead
  per sample (~0.5 ms/call saved)

**`__getitem__`** — retry loop up to `num_samples` attempts to handle missing
latents/images without crashing the DataLoader worker:
- `txt_proc()` — selects a random tokenized template with `random.randrange`
  (no temporary list)
- `aud_proc_beats()` — resolves embedding via `LazyEmbeddingIndex.get()`,
  guarantees contiguous tensor
- Precomputed latents: bypasses `img_proc()` and VAE encoding, reads directly
  from `LazyLatentIndex`
- `is_precomputed_latent` flag tells the training loop whether `pixel_values`
  is already a float16 latent

### Fixes and optimizations

**[OPT-GETITEM-CACHE]** `_TRUE` / `_FALSE` pre-allocated as constant `torch.tensor`
in `__init__`.

**[OPT-TXTCACHE]** All templates tokenized once in `__init__`.

**[OPT-TXTCACHE-CONTIGUOUS]** `_cached_input_ids` tensors made contiguous in `__init__`.

**[OPT-TXTCACHE-IDX]** `random.randrange` instead of `random.choice`.

**[OPT-PANDAS]** `set_index('id')` for O(1) lookup, `groupby('class')` for per-class image lists.

**[CHUNK-SF]** Chunk safetensors mode: supports N independent chunk files.

**[CHUNK-SF-FIX]** Single pass over `sf.keys()` avoids double safetensors header scan.

**[FIX-GETITEM-ROBUST]** Retry up to `num_samples` for missing samples.

**[FIX-IMGID / FIX-SLUGIFY]** Image lookup with fallback: exact ID,
lowercase+underscore, then disk search with dash variants.

**[FIX-3]** `LazyLatentIndex` bypasses `img_proc()` + VAE encoding in training.

**[PERF-CONTIGUOUS]** Audio tensors made contiguous in the DataLoader worker.

**[OPT-CV2]** Removed redundant `cv2.cvtColor` round-trip in `img_proc()`.

**Early Fusion note:** `dataloader_colab.py` is **unchanged** by the Early Fusion migration.
The `<*>` placeholder appears in templates and is tokenized as a normal vocabulary token —
its embedding is included in the text sequence passed to `EarlyFusionEncoder` instead of
being replaced by audio.

---

## train_validation_no_accel_colab.py

### Description

Training and validation script with DDP (DistributedDataParallel) support.
Supports single-GPU and multi-GPU modes via `torchrun`.

### DDP setup

Reads `RANK`, `LOCAL_RANK`, `WORLD_SIZE` from the environment (torchrun), initializes
NCCL. Only rank 0 saves checkpoints and writes to TensorBoard.
`_barrier(is_ddp)` synchronizes processes at critical points.

### PrefetchLoader

Overlaps the GPU transfer of the next batch (on a separate CUDA stream) with the
compute of the current batch. Reduces H2D transfer waiting time.

### build_label_embedding_cache()

Pre-computes for every artistic class the average vector of the CLIP embeddings of the
label words. Uses `base_model.token_embedding` (frozen) instead of the full `CLIPTextModel`.
Result: `label_matrix [N_labels, text_dim]` used by the cosine loss.

### Training loop

**CFG dropout (10%)** — with probability 0.10, zeros out `audio_features` and
`input_ids` (text replaced with `pad_token_id`). This teaches the model to generate
without conditioning, making Classifier-Free Guidance effective at inference. Without this
step, `guidance_scale > 1` produces OOD images.

**Composite loss:**
- MSE on predicted noise (main denoising loss)
- L1 regularization: `lambda_a * mean(|fused_pooled|)`
- Optional L2 regularization: `lambda_b * mean(||fused_pooled||²)`
- Cosine loss: `lambda_c * (1 - cos(fused_pooled, label_embedding))²`

`fused_pooled = fused_seq.mean(dim=1)` — pooling only for auxiliary losses;
the UNet receives the full sequence.

**Gradient accumulation** — `loss / gradient_accumulation_steps` before `.backward()`.
Unscale → clip_grad_norm(1.0) → step → scaler update.

### Checkpoint management

Atomic write with `.tmp` + `os.replace()` to avoid corrupted checkpoints in case of interruption.
Only the embedder weights are saved in `.safetensors`. Tag `"arch": "early_fusion_v9"` for version
identification. Automatic rotation of the N most recent checkpoints (`keep_last_n_checkpoints`).

### Fixes and optimizations

**[DDP-MIGRATION]** Migration from DataParallel to DistributedDataParallel.

**[FIX-DOUBLE-VAL-MIDLOOP]** Avoids both mid-epoch and end-of-epoch validation on the same
step via `_last_validated_step`.

**[FIX-ATOMIC-SAVE]** Atomic write `.tmp` + `os.replace()`.

**[FIX-RESUME-BEST]** `best_model_path` validated on disk after resume.

**[OPT-NOISE-PREALLOCATE]** Noise and timestep buffers allocated once and reused across steps.

**[OPT-PREFETCH]** `PrefetchLoader` with separate CUDA stream.

**[OPT-ZERO-GRAD]** `optimizer.zero_grad(set_to_none=True)`.

**[MEM-1/2/3/4/5/6]** Label cache, delete previous best, TensorBoard rotation,
VAE offload to CPU, `LazyLatentIndex`, worker reduction after VAE precompute.

**[LORA-DDP]** `find_unused_parameters=True` when LoRA is active.

**[EARLY-FUSION-TRAIN]** Changes for Early Fusion:
- Removed `tokenizer.add_tokens()` and `resize_token_embeddings()`
- Removed `set_placeholder_token_id()`
- `build_label_embedding_cache()` receives `base_model.token_embedding` instead of `text_encoder`
- `trainable_params = list(base_model.early_fusion.parameters()) [+ LoRA]`
- `torch.compile` applied to `base_model.early_fusion`
- New CLI args: `--ef_d_model`, `--ef_nhead`, `--ef_num_layers`, `--ef_dropout`
- Warning emitted when loading a late-fusion checkpoint (arch != "early_fusion_v9")

---

## test_no_accel_colab.py

### Description

Inference script for the Early Fusion pipeline. Generates images from audio embeddings
with EarlyFusionEncoder + frozen UNet. Does not use DDP (batch_size=1).

### Automatic checkpoint resolution

`_resolve_checkpoint_path()` looks for the checkpoint in this order:
1. Explicit path provided
2. Sibling `.safetensors` if the provided path was `.bin`
3. `<output_dir>/learned_embeds.safetensors`
4. Glob `best_model_early_fusion_*.safetensors` (most recent by mtime)
5. Glob `weights/*_early_fusion-step*.safetensors`

### Unconditional mode for CFG

**`--uncond_mode zeros`** — silent audio and text (zero tensor). The model has seen
these during training with CFG dropout, so it recognizes them as the "no conditioning"
signal. Fast.

**`--uncond_mode text_only`** — silent audio + prompt text. CFG amplifies exactly
the audio contribution above the textual baseline. More expensive but semantically precise.

In both cases, `uncond_embeddings` has the same shape as `cond_embeddings`
`[1, T_a+T_t, output_size]` for correct CFG stacking.

### Generation loop

1. `_get_text_embeddings(cond_input_ids)` → `text_tokens`
2. `early_fusion(audio_feats, text_tokens)` → `cond_embeddings [1, T, output_size]`
3. `cat([uncond_embeddings, cond_embeddings])` → `text_embeddings [2, T, output_size]`
4. DDIM/Euler loop with CFG:
   `uncond + guidance_scale * (cond - uncond)`
5. VAE decode in float32 for stability, JPEG save

### Applied fixes

**[FIX-1]** Removed runtime BEATs call; uses `batch["audio_features"]` precomputed.

**[FIX-A]** `EulerDiscreteScheduler` for SD 2.1 with v-prediction.

**[FIX-CFG]** Unconditional prompt is base text without `<*>`, not an empty string.

**[FIX-VAE]** Cast to float32 for stable VAE decode in fp16 mode.

**[EARLY-FUSION-TEST]** Changes for Early Fusion:
- Removed `tokenizer.add_tokens()` and `resize_token_embeddings()`
- `StableDiffusionPipeline` no longer used for conditioning
- `_resolve_checkpoint_path()` updated for `early_fusion` glob
- Support for `--uncond_mode` for both CFG strategies

---

## Weight Format Migration

**[FMT-SAFETENSORS]** Weight files migrated from `torch.save` (`.bin`) to
safetensors. Scope: embedder, LoRA, best-model snapshot.

The resume checkpoint (`.pt`) remains unchanged because it contains
optimizer/scaler/scheduler state not compatible with safetensors.

All loading paths have transparent fallback `.bin` → `.safetensors`.
Old `.bin` checkpoints continue to work.

Advantages: zero-copy, pickle-free, lazy mmap, security (no arbitrary code execution
on load), multi-framework compatibility.

---

## Architectural Evolution

### Late Fusion v8 (previous architecture)

```
BEATs features [B, T, 2304]
    → FGAEmbedder → [B, 768]   (single pooled vector)
    → injected into the <*> placeholder inside CLIPTextModel
    → CLIPTextModel processes the sequence [B, 77, 768] with audio in place of <*>
    → encoder_hidden_states [B, 77, 768]
    → UNet
```

**Limitations solved by Early Fusion:**
- CLIP processed text and audio separately — audio entered only as a single
token, without cross-modal interaction during text encoding
- `input_ids.max()` to find the placeholder was fragile with high vocab IDs
- FGAEmbedder used VQA `Atten` with non-guaranteed output shape
- Single vector `[B, 768]` limited the richness of the conditioning to the UNet

### Early Fusion v9 (current architecture)

```
BEATs features [B, T_a, 2304]  +  CLIP token_embedding [B, T_t, 768]
    → EarlyFusionEncoder (shared Transformer)
    → fused_seq [B, T_a+T_t, output_size]   (full sequence)
    → encoder_hidden_states → UNet cross-attention
```

**Advantages:**
- Every audio token can attend to every text token at every layer of the Transformer
- Full sequence `[T_a+T_t, D]` instead of a single vector `[1, D]`
- UNet can attend to different tokens for different spatial regions of the image
- No dependency on the placeholder token or CLIP injection logic
- Full CLIPTextModel discarded (~400 MB VRAM saved)

| | Late Fusion v8 | Early Fusion v9 |
|---|---|---|
| Trainable component | FGAEmbedder | EarlyFusionEncoder |
| Audio-text fusion | after CLIP (single token) | before the Transformer (sequence) |
| encoder_hidden_states | `[B, 77, 768]` | `[B, T_a+T_t, 768]` |
| CLIP required | yes (full Transformer) | only token_embedding |
| Placeholder `<*>` | necessary | no-op |
| Checkpoint tag | `arch: late_fusion_v8` | `arch: early_fusion_v9` |
