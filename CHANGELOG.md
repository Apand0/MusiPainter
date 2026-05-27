# CHANGELOG — Musipainter Pipeline

This file collects all [FIX-*], [OPT-*], [LORA-*], [PERF-*], [MEM-*], [DDP-*]
and [EARLY-FUSION-*] annotations extracted from the source files.

# =============================================================================
# embedder.py
# =============================================================================

[FIX-EMBEDDER]
  Replaced the old FGA (Factor Graph Attention) module with standard Attentive
  Pooling, matching the architecture in Musipainter paper Section 3.3, Figure 3.

  Problem with old code:
    The old embedder used `Atten` from `modules.FGA.atten`, a module designed
    for multi-modal VQA (questions, answers, images, dialogue history). With a
    single audio modality and pairwise_flag=False, the output shape was not
    guaranteed to be [B, 768]: it could return [B, T, 768] with the temporal
    dimension still present, breaking CLIP placeholder injection. It also required
    the external `modules.FGA.atten` dependency.

  Correct architecture (paper Section 3.3):
    phi(a) ∈ R^{T × 2304}   (BEATs layers 4, 8, 12 concatenated)
    W1: Linear(2304 → 2304) + GELU
    W2: Linear(2304 → output_size)
    Attentive Pooling: [B, T, D] → [B, D]
    output_size = 768  for SD v1-4
    output_size = 1024 for SD 2

  NOTE: FGAEmbedder (embedder.py) is superseded by EarlyFusionEncoder
  (early_fusion_encoder.py) as of the Early Fusion migration. The file is
  retained for reference but is no longer imported by MusicTokenWrapper.


# =============================================================================
# MusicToken_no_accel.py
# =============================================================================

[FIX-PLACEHOLDER]
  CLIPTextEmbeddings used input_ids.max() to locate the <*> placeholder token.
  In batches where another token (e.g. "1960" for "20th century") had a higher
  ID than the placeholder, this silently injected audio at the wrong position,
  replacing valid text tokens and making learning impossible.
  Fix: placeholder_token_id is set explicitly via set_placeholder_token_id()
  after tokenizer.add_tokens("<*>") and propagated to the embeddings module.
  NOTE: set_placeholder_token_id() is a no-op as of the Early Fusion migration.

[FIX-2304]
  input_size set to 768 × 3 = 2304 to match the BEATs preprocessing, which
  now returns the concatenation of layers 4, 8, and 12 instead of their mean.

[FIX-DP] (DataParallel / DDP compatibility)
  Frozen SD components are loaded in float16 on CPU.
  FGAEmbedder is initialised in float32 (PyTorch default).
  DataParallel/DDP requires all parameters on the same device at wrap time.
  Fix:
    a) __init__ does NOT call .to(device); the training loop does so AFTER
       DataParallel/DDP wrapping.
    b) forward() casts input tensors, not module parameters.
    c) VAE is never moved off the MusicTokenWrapper module directly; the
       training loop accesses it via base_model.vae to avoid the
       "buffer registered on cuda:0 but found on cpu" error.

[FIX-forward]
  Mixed fp16/fp32 dtype handling in forward() revised alongside OPT-DETACH-UNET.

[OPT-UNET-NOGRAD]
  UNet has 859 M frozen parameters. Without no_grad(), PyTorch still builds
  the full computation graph and stores all intermediate activations for backward,
  costing ~1.5–2 GB VRAM and ~1800 ms/step of wasted backward time.
  The current forward() passes encoder_hidden_states (requires_grad=True) directly
  to the UNet, so the graph is retained only through that tensor — a narrower
  and cheaper path.

[OPT-STE-UNET] (Straight-Through Estimator)
  Gradient flow analysis for the frozen UNet — see CHANGELOG entry for full
  derivation. The current implementation passes encoder_hidden_states directly;
  the STE identities show why gradient flow is preserved even with the frozen UNet.

[FIX-SURROGATE-SPATIAL]
  Old surrogate used audio_token.mean() as the differentiable bridge, causing
  the embedder to collapse to a block of similar values. Fixed with a
  normalised dot-product surrogate (superseded by the Early Fusion architecture).

[OPT-HF-CACHE-TMP]
  HF model cache redirected to /tmp (RAM tmpfs on Kaggle) to avoid quota issues.

[LORA-INIT]
  LoRAAttnProcessor instances must be installed on the UNet BEFORE the
  xformers / AttnProcessor2_0 fallback.

[LORA-TRAIN]
  Only lora_layers.train() is called, not unet.train().

[LORA-TEST]
  AttnProcsLayers is a direct view of the UNet processor weights.
  load_state_dict() updates them in-place.

[FIX-COMPILE-LOAD]
  torch.compile() prepends "_orig_mod." to all parameter names in the saved
  state_dict. Stripped on load so compiled and non-compiled checkpoints are
  interchangeable.


# =============================================================================
# early_fusion_encoder.py  (NEW — Early Fusion migration)
# =============================================================================

[EARLY-FUSION-ENCODER]
  New module implementing the FuseLIP-style Early Fusion architecture.
  Replaces FGAEmbedder (embedder.py) as the trainable component.

  Architecture:
    audio_tokens [B, T_a, 2304] ──► nn.Linear(2304, d_model) ──► + modality_emb + pos_emb
                                                                         │
                                                               torch.cat(dim=1)
                                                                         │
    text_tokens  [B, T_t, D]   ──► nn.Linear(D, d_model)    ──► + modality_emb + pos_emb

    fused [B, T_a+T_t, d_model]
      ──► nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
              d_model, nhead, dim_feedforward=d_model*4,
              activation='gelu', batch_first=True, norm_first=True
            ),
            num_layers=4
          )
      ──► mean-pool over sequence
      ──► nn.LayerNorm(d_model)
      ──► nn.Linear(d_model, output_size)
      ──► [B, output_size]

  Key properties:
    - Every audio token attends to every text token at every Transformer layer
      (Early Fusion), unlike the original Late Fusion which replaced a single
      placeholder token after CLIP had already processed text independently.
    - norm_first=True (Pre-LN) for numerical stability with mixed modalities.
    - Learnable modality-type embeddings distinguish audio from text tokens.
    - Separate learnable positional embeddings for each modality.
    - Masking supported via src_key_padding_mask (PAD-aware mean pooling).

  References:
    FuseLIP paper; PyTorch docs:
      https://docs.pytorch.org/docs/2.12/generated/torch.nn.TransformerEncoderLayer.html
      https://docs.pytorch.org/docs/2.12/generated/torch.nn.Linear.html


# =============================================================================
# MusicToken_no_accel.py  (Early Fusion migration changes)
# =============================================================================

[EARLY-FUSION-WRAPPER]
  MusicTokenWrapper updated for the Early Fusion architecture:

  Removed:
    - Full CLIPTextModel (12-layer transformer, ~400 MB VRAM) — replaced by
      lightweight token_embedding (nn.Embedding, ~3 MB).
    - FGAEmbedder import and instantiation.
    - set_placeholder_token_id() logic (now a no-op for CLI compatibility).
    - The <*> placeholder injection mechanism entirely.

  Added:
    - self.token_embedding: frozen CLIP nn.Embedding extracted from CLIPTextModel
      then discarded; used by _get_text_embeddings() to look up token vectors.
    - self.early_fusion: EarlyFusionEncoder (trainable).
    - self.embedder: alias to self.early_fusion for checkpoint-helper compatibility.
    - _get_text_embeddings(input_ids): returns [B, seq_len, text_dim] float32.

  forward() change:
    OLD: embedder(audio_features) → audio_token [B, 768]
         text_encoder(audio_token, input_ids) → encoder_hidden_states [B, seq_len, D]
    NEW: _get_text_embeddings(input_ids) → text_tokens [B, T_t, D]
         early_fusion(audio_tokens, text_tokens) → fused [B, output_size]
         fused.unsqueeze(1) → encoder_hidden_states [B, 1, output_size]


# =============================================================================
# dataloader_colab.py
# =============================================================================

[OPT-GETITEM-CACHE]
  self._TRUE / self._FALSE pre-allocated in __init__ as constant torch.tensor
  scalars.

[OPT-TXTCACHE-CONTIGUOUS]
  _cached_input_ids tensors are made contiguous once in __init__.

[OPT-TXTCACHE-IDX]
  txt_proc() uses random.randrange instead of random.choice.

[OPT-PANDAS]
  prepare_dataset() uses set_index('id') for O(1) per-sample lookup and
  groupby for per-class image lists.

[OPT-TXTCACHE]
  All template strings are tokenised once in __init__.

[CHUNK-SF]
  Chunked safetensors mode: supports N independent chunk files in chunks/.

[CHUNK-SF-FIX]
  Single pass over sf.keys() avoids scanning the safetensors header twice.

[FIX-GETITEM-ROBUST]
  __getitem__ retries up to num_samples times on missing image/latent.

[FIX-IMGID]
  Image ID lookup tries exact ID, lowercased+underscored, then disk search.

[FIX-IMGID-LATENT]
  LazyLatentIndex key lookup uses the stem of the file path.

[FIX-FOLDER / FIX-SLUGIFY]
  Folder names derived from label with spaces→underscores and slashes→dashes.

[FIX-3]
  LazyLatentIndex bypasses img_proc() + VAE encoding at training time.

[PERF-CONTIGUOUS]
  Audio tensors made contiguous in the DataLoader worker via .contiguous().

[OPT-CV2]
  Removed redundant cv2.cvtColor round-trip in img_proc().

  NOTE: dataloader_colab.py is UNCHANGED by the Early Fusion migration.
  The placeholder_token still appears in imagenet_templates_small and is
  tokenised as a normal vocabulary token — its embedding is included in the
  text sequence fed to EarlyFusionEncoder rather than being replaced by audio.


# =============================================================================
# modeling_clip.py
# =============================================================================

[FIX-PLACEHOLDER] (CLIPTextEmbeddings)
  Replaced input_ids.max() placeholder detection with an explicit
  placeholder_token_id attribute.

[FIX-GRAD] (CLIPTextEmbeddings)
  Differentiable linear combination replaces in-place assignment to preserve
  the autograd graph through frozen token embeddings.

[OPT-CLONE] (CLIPTextEmbeddings)
  Single input_ids.clone() at the top.

[FIX-EOS] (CLIPTextTransformer)
  EOS pooling uses a saved copy of input_ids with placeholder lowered to 5.

[FIX-COMPAT] (CLIPTextModel)
  Integer-dtype guard on audio_e ensures backward-compatibility with the
  Diffusers StableDiffusionPipeline positional call convention.

  NOTE: modeling_clip.py is NOT imported by MusicTokenWrapper as of the Early
  Fusion migration. The full CLIPTextModel is replaced by a direct token
  embedding lookup. The file is retained as a reference implementation.


# =============================================================================
# modules/FGA/  (fga_model.py, atten.py)
# =============================================================================

  NOTE: fga_model.py and atten.py are standalone Factor Graph Attention modules
  used in the original VQA research that inspired the FGA naming in Musipainter.
  They are NOT imported anywhere in the current pipeline and are retained for
  reference only.


# =============================================================================
# train_validation_no_accel_colab.py
# =============================================================================

[DDP-MIGRATION]
  Migrated from DataParallel to DistributedDataParallel. See inline annotations
  [DDP-1] through [DDP-10] in the source file for full details.

[FIX-DOUBLE-VAL-MIDLOOP]
  Prevents running both mid-epoch and end-of-epoch validation on the same step.

[FIX-ATOMIC-SAVE]
  Checkpoints written to .tmp then atomically renamed with os.replace().

[FIX-RESUME-BEST]
  best_model_path validated on disk after checkpoint resume.

[FIX-COMPILE-DP→DDP]
  _orig_mod. prefix stripped on load; checkpoints interchangeable between
  compiled and non-compiled runs.

[OPT-NOISE-PREALLOCATE]
  Noise and timestep buffers allocated once and reused across steps.

[OPT-PREFETCH]
  PrefetchLoader pins next batch to GPU via a separate CUDA stream.

[OPT-ZERO-GRAD]
  optimizer.zero_grad(set_to_none=True).

[MEM-1] Embedding label cache pre-computed once before training.
[MEM-2] Previous best model file deleted when new best is saved.
[MEM-3] TensorBoard rotation when runs/ exceeds max_tb_size_mb.
[MEM-4] VAE offloaded to CPU after latent precomputation.
[MEM-5] LazyLatentIndex: memory-mapped safetensors chunks.
[MEM-6] Worker count reduced after VAE precomputation.

[RESUME]
  Full checkpoint: global_step, best_vloss, embedder, optimizer, scaler,
  lr_scheduler, optional lora. Tag "arch" written to identify architecture.

[LORA-DDP]  find_unused_parameters=True when LoRA is active.
[LORA-TRAIN-MODE]  lora_layers.train(), unet.eval() kept separate.
[LORA-BEST-MID]  Dual save: embedder + LoRA on new best (mid-epoch).
[LORA-BEST]      Dual save: embedder + LoRA on new best (end-of-epoch).

[EARLY-FUSION-TRAIN]
  Training script changes for the Early Fusion migration:
    - tokenizer.add_tokens() and text_encoder.resize_token_embeddings() removed.
    - set_placeholder_token_id() call removed.
    - build_label_embedding_cache() receives base_model.token_embedding
      (nn.Embedding) instead of text_encoder; semantics unchanged (mean of
      CLIP token embeddings for label words).
    - trainable_params = list(base_model.early_fusion.parameters()) [+ LoRA].
    - torch.compile applied to base_model.early_fusion.
    - New CLI args: --ef_d_model, --ef_nhead, --ef_num_layers, --ef_dropout.
    - Checkpoint tag "arch": "early_fusion_v9" for version identification.
    - Warning emitted when loading a late-fusion checkpoint into this script.


# =============================================================================
# test_no_accel_colab.py
# =============================================================================

[FIX-1] Removed runtime BEATs call; uses precomputed batch["audio_features"].
[FIX-2] Removed redundant second embedder call.
[FIX-3] Placeholder injection replaced by EarlyFusionEncoder call.
[FIX-4] StableDiffusionPipeline built once outside the generation loop.
[FIX-5] center_crop=False default.
[FIX-6] LOCAL_RANK guard in parse_args().
[FIX-PROMPT]    Default prompt includes <*> placeholder.
[FIX-LATENTS-DIR] --latents_dir bypasses PIL decode + VAE encode.
[DDP-TEST]      Inference does NOT use DDP (batch_size=1).
[FIX-A]         EulerDiscreteScheduler for SD 2.1 v-prediction.
[FIX-CFG]       Unconditional prompt is base text without <*>, not empty string.
[FIX-VAE]       Float32 cast for VAE decode stability in fp16 mode.

[EARLY-FUSION-TEST]
  Inference script changes for the Early Fusion migration:
    - tokenizer.add_tokens() and resize_token_embeddings() removed.
    - StableDiffusionPipeline no longer used for conditioning.
    - Conditional embedding:
        text_tokens  = base_model._get_text_embeddings(cond_input_ids)
        cond_fused   = base_model.early_fusion(audio_tokens, text_tokens)
        cond_embed   = cond_fused.unsqueeze(1)   → [1, 1, output_size]
    - Unconditional embedding for CFG — two modes (--uncond_mode):
        "zeros"     : zero tensor [1, 1, output_size]. Fast and clean.
        "text_only" : EarlyFusionEncoder(silent_audio=zeros, text_tokens).
                      CFG amplifies exactly the audio contribution above
                      the text-only baseline.
    - _resolve_checkpoint_path() updated to glob best_model_early_fusion_*.


# =============================================================================
# preprocess_image_latents_colab.py
# =============================================================================

[MEM-1] No RAM accumulation — flush every FLUSH_EVERY_N_BATCHES batches.
[MEM-2] Distinct chunk files — never overwritten.
[MEM-3] O(1) resume — reads only chunk headers.
[MEM-4] Streaming merge — max extra disk = largest chunk.
[OPT-1] DataParallel VAE on multi-GPU.
[OPT-2] CPU workers for parallel PIL decode.
[OPT-3] pin_memory=True when workers > 0.
[OPT-4] autocast float16 for VAE encoding.
[OPT-6] Throughput log every 10×batch_size images.
[FIX-OOMAUTO] OOM: halve batch, retry, skip on failure.


# =============================================================================
# preprocess_audio_embeddings_colab.py
# =============================================================================

[MEM-1] No RAM accumulation — flush every FLUSH_EVERY_N_BATCHES batches.
[MEM-2] Distinct chunk files.
[MEM-3] O(1) resume — reads only chunk headers.
[MEM-4] Streaming merge.
[SPEED-1] GC + empty_cache after every flush.
[SPEED-2] DataParallel BEATs on multi-GPU.
[SPEED-3] Parallel audio I/O via ThreadPoolExecutor.
[FIX-2304] Full layer concatenation (layers 4+8+12 → 2304-D) instead of mean.


# =============================================================================
# preprocess_manager.py
# =============================================================================

[KAGGLE-DISK]
  Monitors free disk space before every class batch. Stops
  automatically when free space falls below --critical_free_gb (default 2.0 GB),
  preventing data loss from a full disk. Prints step-by-step instructions for
  uploading the partial output as a new Kaggle Dataset and resuming with
  --existing_datasets on the next kernel run.

[KAGGLE-CLASS-BATCH]
  Processes audio classes in batches of size --class_batch_size (default 5).
  Between batches, disk space is checked and gc.collect() is triggered. This
  gives finer-grained control than processing all classes at once, which could
  exceed the 20 GB quota mid-run without a clean stopping point.

[KAGGLE-STRIDE-SWEEP]
  Supports encoding at multiple temporal_pool_stride values in a single session
  via --strides. Each stride gets its own output subdirectory (stride<N>/), so
  multiple compression levels can be precomputed before uploading. After all
  strides complete (or disk runs out), the user uploads the results and re-runs
  with --existing_datasets to continue any unfinished strides.

[KAGGLE-RESUME]
  The --existing_datasets flag accepts comma-separated paths to read-only Kaggle
  Dataset directories containing chunks from previous runs. Already-encoded audio
  IDs are skipped automatically, enabling incremental preprocessing across
  multiple kernel sessions without reprocessing.

[KAGGLE-SKIP-MERGE]
  --skip_merge prevents the final streaming merge into a single safetensors file.
  On Kaggle this saves ~largest_chunk of extra disk space. The dataloader reads
  per-class chunks in chunks/ directly, so training is unaffected.

[KAGGLE-INSTRUCTIONS]
  When disk runs low, _print_upload_instructions() generates a complete resume
  command with all flags pre-filled, including the new dataset path. The user
  only needs to copy-paste and re-run.


# =============================================================================
# argparse_multiembedding.py
# =============================================================================

[MULTIEMB-ARGS]
  add_multiembedding_args(parser) registers three new CLI flags:
    --embeddings_dir            : comma-separated paths to embedding directories
    --embeddings_preload_all    : bool, True = RAM preload, False = lazy mmap
    --embeddings_max_sf_handles : int, LRU cache size for open file handles
  Safe to call multiple times (uses conflict_handler='resolve' internally).

[MULTIEMB-INDEX]
  build_embedding_index(args, logger) constructs a LazyEmbeddingIndex from the
  parsed args. Returns None when args.embeddings_dir is falsy. The Museart Dataset
  class reads args.embeddings_dir, args.embeddings_preload_all, and
  args.embeddings_max_sf_handles directly, so explicit construction is optional.

[MULTIEMB-BOOL]
  _str2bool() accepts 'yes/true/1' and 'no/false/0' for flexible boolean parsing
  from CLI strings.


# =============================================================================
# dataloader_colab.py  (additional functional notes)
# =============================================================================

[LAZY-EMB-MULTI]
  LazyEmbeddingIndex._build_index() scans each directory independently and merges
  audio IDs into a single in-memory index. Duplicate IDs across directories are
  resolved with last-write-wins semantics. The index stores either True (tensor
  in _flat_cache for preload_all=True) or an absolute file path string (for
  preload_all=False lazy mmap).

[LAZY-EMB-LRU]
  In lazy mode, _get_from_sf() maintains an LRU cache of open safetensors file
  handles capped at max_sf_handles. When the cache is full, the oldest handle is
  evicted via pop() before opening the new file. Accessing an existing handle
  moves it to the most-recently-used position.

[LAZY-EMB-LEGACY]
  _index_legacy_pt() provides backward compatibility for .pt chunk files. When
  preload_all=True, tensors are loaded into _flat_cache immediately. When False,
  only the file path is stored and an LRU cache of loaded chunk dicts (max 4)
  is maintained by _get_from_pt().

[LAZY-LATENT-SF]
  LazyLatentIndex._build_index() prefers a single merged image_latents.safetensors
  file. If absent, it falls back to legacy image_latents_*.pt chunks. In preload
  mode all tensors are copied to _flat_cache and the safe_open handle is released.
  In lazy mode the handle is kept open for mmap access.

[LAZY-LATENT-LRU]
  For legacy .pt chunks, LazyLatentIndex uses the same LRU pattern as
  LazyEmbeddingIndex: a max_cached_chunks=4 cache of loaded chunk dictionaries.

[MUSEART-PREPARE]
  prepare_dataset() builds three parallel lists (audio_path, image_path, label)
  by joining the audio CSV against the image CSV on the 'class' column. Image IDs
  are matched via set_index('id') for O(1) lookup and groupby('class') for per-class
  candidate pools. Missing images trigger a filesystem search with both underscore
  and dash filename variants.

[MUSEART-RETRY]
  __getitem__ implements up to num_samples retry attempts when an image or latent
  is missing. This prevents a single missing file from crashing the DataLoader
  worker process. The retry uses modulo indexing to wrap around the dataset.

[MUSEART-TXT-CACHE]
  All template strings are tokenised once in __init__ and stored as contiguous
  tensors in _cached_input_ids. txt_proc() selects one via random.randrange,
  avoiding per-sample tokenisation overhead (~0.5 ms/call).

[MUSEART-AUD-EMB]
  aud_proc_beats() resolves the audio embedding via LazyEmbeddingIndex.get() when
  available, otherwise falls back to a dict lookup. All returned tensors are
  squeezed (if dim==3) and made contiguous before returning to the collate function.

[MUSEART-IMG-LATENT]
  When image_latents is configured, __getitem__ bypasses img_proc() and VAE
  encoding entirely, reading precomputed float16 latents from disk. This cuts
  per-batch CPU time by ~60% during training and inference.


# =============================================================================
# safetensors weight format migration
# =============================================================================

[FMT-SAFETENSORS]
  Weight output files migrated from torch.save (.bin) to safetensors format.
  Scope: embedder, LoRA, best-model snapshots.
  Resume checkpoint (.pt) unchanged (contains optimizer/scaler/scheduler state).
  Transparent .bin → .safetensors fallback in all load paths.
  See previous CHANGELOG entry for full details.
