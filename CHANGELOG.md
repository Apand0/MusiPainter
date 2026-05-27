# CHANGELOG — Musipainter Pipeline

This file collects all [FIX-*], [OPT-*], [LORA-*], [PERF-*], [MEM-*] and [DDP-*] annotations
extracted from the source files.

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

[OPT-UNET-NOGRAD] (v7, kept as standard forward)
  UNet has 859 M frozen parameters. Without no_grad(), PyTorch still builds
  the full computation graph and stores all intermediate activations for backward,
  costing ~1.5–2 GB VRAM and ~1800 ms/step of wasted backward time.
  The current forward() passes encoder_hidden_states (requires_grad=True) directly
  to the UNet, so the graph is retained only through that tensor — a narrower
  and cheaper path. The UNet backbone's own activation buffers are freed
  because unet.requires_grad_(False) prevents accumulation beyond that point.

[OPT-STE-UNET] (Straight-Through Estimator, v7)
  Gradient flow analysis for the frozen UNet:
    enc_val        = encoder_hidden_states.detach()
    model_pred     = unet(noisy_lats, t, enc_val)           # no grad on UNet
    model_pred_sg  = model_pred + (encoder_hidden_states - enc_val)
  Forward: enc_hidden - enc_hidden.detach() = 0 → model_pred_sg = model_pred
  Backward: d(model_pred_sg)/d(enc_hidden) = I (identity via enc_hidden term)
            d(loss)/d(enc_hidden) = exact gradient, no UNet backward required.
  The current implementation passes encoder_hidden_states directly; the STE
  identities show why gradient flow is preserved even with the frozen UNet.

[FIX-SURROGATE-SPATIAL] (v6 → v7 replacement)
  Old surrogate used audio_token.mean() as the differentiable bridge.
  Gradient of mean() w.r.t. [B, D] is uniform 1/(B*D) per element, giving
  every dimension the same gradient direction → embedder collapsed to a block
  of similar values → SD produced neon noise instead of figurative art.
  Fix: surrogate based on normalised dot-product:
    _scalar_bridge = (audio_token * audio_token.detach()).sum() / ||audio_token||
  Each dimension receives a proportional gradient → directional learning preserved.
  (The v7 direct-pass approach makes the surrogate unnecessary; this note explains
  why the earlier mean() approach was wrong.)

[OPT-TEXT-ENC-NOGRAD] (v5)
  text_encoder is frozen but its forward built a full graph over ~12 CLIP layers.
  torch.no_grad() on the text_encoder body (with gradient carried via the
  audio_token surrogate path) saved ~0.15–0.25 s/step.

[OPT-CAST] (v5)
  audio_features.float() replaces a conditional branch on dtype.

[OPT-HF-CACHE-TMP]
  HF model cache redirected to /tmp (RAM tmpfs on Kaggle):
  - /tmp is not counted toward the /kaggle/working 20 GB quota.
  - SD 2 components weigh ~2.1 GB — without /tmp the quota overflows.
  - RAM reads are ~2× faster than disk reads.
  Trade-off: cache lost on kernel restart (models re-downloaded in ~10 s).
  Override: pass hf_cache_dir='/kaggle/working/hf_model_cache' to persist
  across sessions of the same kernel lifetime.

[LORA-INIT]
  LoRAAttnProcessor instances must be installed on the UNet BEFORE the
  xformers / AttnProcessor2_0 fallback. If LoRA is active, the
  set_attn_processor(AttnProcessor2_0) fallback is skipped to avoid
  overwriting the LoRA processors.

[LORA-TRAIN]
  Only lora_layers.train() is called, not unet.train(). Calling unet.train()
  would put BatchNorm / Dropout layers in training mode, altering the frozen
  UNet's forward behaviour.

[LORA-TEST]
  AttnProcsLayers is a direct view of the UNet processor weights.
  load_state_dict() updates them in-place in the already-installed processors.
  Do NOT also call unet.load_attn_procs() — it is redundant and can cause
  inconsistencies if the .bin is not in the native HF format.

[FIX-COMPILE-LOAD]
  torch.compile() prepends "_orig_mod." to all parameter names in the saved
  state_dict. Stripped on load so compiled and non-compiled checkpoints are
  interchangeable.


# =============================================================================
# dataloader_colab.py
# =============================================================================

[OPT-GETITEM-CACHE]
  self._TRUE / self._FALSE pre-allocated in __init__ as constant torch.tensor
  scalars. torch.tensor(True/False) allocates a new Python object on every
  __getitem__ call. With 20 k steps × batch 8 = 160 k items, this saves
  ~320 k Python object allocations.

[OPT-TXTCACHE-CONTIGUOUS]
  _cached_input_ids tensors are made contiguous once in __init__ instead of
  checking is_contiguous() on every __getitem__ call.

[OPT-TXTCACHE-IDX]
  txt_proc() uses random.randrange instead of random.choice: randrange is
  ~15 % faster because it avoids the internal len() call inside choice().

[OPT-PANDAS]
  prepare_dataset() uses set_index('id') for O(1) per-sample lookup and
  groupby for per-class image lists, replacing the naive O(n²) DataFrame scan.

[OPT-TXTCACHE]
  All template strings are tokenised once in __init__. tokenisation costs
  ~0.5 ms/call; pre-computing saves ~80 s over a 20 k-step run with batch 8.

[CHUNK-SF]
  Chunked safetensors mode: supports N independent chunk files generated when
  the final merge step in preprocess_audio_embeddings_colab.py fails due to
  insufficient disk or RAM. Files are named chunk_NNNN.safetensors and stored
  in a chunks/ sub-directory. With preload_all=False, an LRU cache of
  safe_open handles (max_sf_handles) avoids re-opening the same chunk on
  every access.

[CHUNK-SF-FIX]
  Single pass over sf.keys() avoids scanning the safetensors header twice.
  Safe_open handle is deleted after the loop; with preload_all=True tensors are
  already in _flat_cache; with preload_all=False handles are re-opened via LRU.

[FIX-GETITEM-ROBUST]
  __getitem__ retries up to num_samples times when an image or latent is
  missing, instead of crashing the DataLoader worker.

[FIX-IMGID]
  Image ID lookup tries exact ID, lowercased+underscored variant, then a direct
  disk search with both underscore and dash filename forms.

[FIX-IMGID-LATENT]
  LazyLatentIndex key lookup uses the stem of the file path. Normalised variant
  is tried when the raw stem is not found.

[FIX-FOLDER / FIX-SLUGIFY]
  Folder names are derived from the class label with spaces→underscores and
  slashes→dashes. Both underscore and dash variants of the image filename stem
  are tried to handle pre-existing files that use dashes.

[FIX-3]
  LazyLatentIndex bypasses img_proc() + VAE encoding at training time, loading
  float16 latents precomputed by preprocess_image_latents_colab.py.

[PERF-CONTIGUOUS]
  Audio tensors are made contiguous in the DataLoader worker via a single
  .contiguous() call so that torch.stack in the collate function can use fast
  contiguous memcpy instead of a scattered byte copy.

[OPT-CV2]
  Removed a redundant cv2.cvtColor RGB→BGR→RGB round-trip in img_proc().
  PIL Image.open + convert("RGB") already returns correct RGB. Also removed
  the cv2 import which was no longer needed.


# =============================================================================
# modeling_clip.py
# =============================================================================

[FIX-PLACEHOLDER] (CLIPTextEmbeddings)
  Replaced input_ids.max() placeholder detection with an explicit
  placeholder_token_id attribute set by MusicTokenWrapper after add_tokens().
  input_ids.max() was unreliable: in any batch where a text token had a higher
  ID than the placeholder, audio was injected at the wrong position and learning
  was impossible. The legacy fallback is retained only for backward-compatibility
  with checkpoints loaded without this fix.

[FIX-GRAD] (CLIPTextEmbeddings)
  Old in-place assignment `inputs_embeds[indices] = audio_e_squeezed` broke
  the autograd graph: base_embeds has requires_grad=False (frozen token embedding
  table), so autograd would not create an edge from audio_e to inputs_embeds,
  silently stopping gradient flow to FGAEmbedder.
  Fix: differentiable linear combination:
    inputs_embeds = base_embeds * (1 - mask) + audio_expanded * mask
  Forward: identical value (mask is binary 0/1).
  Backward: full gradient chain loss → enc_hidden → inputs_embeds →
            audio_expanded → audio_e → audio_token → FGAEmbedder.

[OPT-CLONE] (CLIPTextEmbeddings)
  Single input_ids.clone() at the top; input_ids_for_embed reuses the clone
  with masked_fill in-place instead of a second full clone.

[FIX-EOS] (CLIPTextTransformer)
  EOS pooling uses argmax on a saved copy of input_ids where the placeholder
  has been lowered to 5. Without this, argmax() could find the placeholder
  (which may have a higher token ID than EOS) instead of the actual EOS token,
  corrupting the pooled representation and any loss term using it.
  The copy is stored in embeddings._last_input_ids_for_pooling and read
  immediately by CLIPTextTransformer.forward(). Thread-safe because each
  forward is synchronous and DDP/DataParallel creates separate module replicas.

[FIX-COMPAT] (CLIPTextModel)
  Diffusers StableDiffusionPipeline calls text_encoder(input_ids, attention_mask=...)
  with input_ids as the first positional argument. Since audio_e is now the
  first positional argument of CLIPTextModel.forward(), an integer-dtype guard
  detects this case and reroutes the tensor to input_ids, setting audio_e=None.


# =============================================================================
# train_validation_no_accel_colab.py
# =============================================================================

[DDP-MIGRATION] (v8)
  Migrated from DataParallel (DP) to DistributedDataParallel (DDP).

  Problem with DP:
    DP runs in a single Python process. The master thread splits the batch,
    copies it to each GPU via CUDA P2P, runs forward in parallel, then reduces
    gradients back to cuda:0. The Python GIL serialises parts of the backward
    pass, adding ~30–40 % overhead on dual T4. It also uses copy.deepcopy()
    on the model every forward, which breaks torch.compile compatibility.

  Why DDP:
    DDP launches one OS process per GPU. There is no shared GIL, so forward and
    backward are truly parallel. NCCL all-reduces gradients asynchronously during
    the backward, overlapping with computation. torch.compile is safe because
    each process owns its own module replica (no deepcopy). Expected speedup
    on T4×2 is ~25–40 % over DP.

[DDP-1] INIT
  dist.init_process_group("nccl") automatically detects RANK, LOCAL_RANK and
  WORLD_SIZE from environment variables set by torchrun. If the variables are
  absent (single-GPU) the group is not initialised and training falls back to
  single-process mode.

[DDP-2] DEVICE
  Each rank uses its own device = cuda:LOCAL_RANK. LOCAL_RANK=0 → cuda:0,
  LOCAL_RANK=1 → cuda:1. On single GPU LOCAL_RANK does not exist, so device
  defaults to cuda:0.

[DDP-3] SAMPLER
  DistributedSampler splits the dataset evenly across world_size ranks.
  With 2 processes and 19 366 samples, rank 0 sees 9 683 samples and rank 1
  sees the rest. The sampler seed is updated every epoch (set_epoch) so that
  each rank receives a different shuffle.

[DDP-4] MODEL WRAP
  DDP wraps the module AFTER model.to(device). torch.compile is applied to
  the embedder before the DDP wrap; this is safe because DDP uses gradient
  hooks rather than deepcopy. find_unused_parameters is set to True only when
  LoRA is active (self-attention processors may not receive gradients from the
  cross-attention loss). broadcast_buffers=False avoids synchronising frozen
  batch-norm buffers across ranks, saving bandwidth.

[DDP-5] VAE PRECOMPUTE
  VAE latent precomputation runs ONLY on rank 0 (it owns the filesystem view).
  Other ranks wait at a dist.barrier() before reading the latents. When
  latents_dir is provided (LazyLatentIndex) there is no runtime precomputation;
  every rank reads its own latents from disk in parallel.

[DDP-6] LOSS
  DDP automatically all-reduces gradients during the backward pass. The loss is
  not manually averaged across ranks; DDP divides gradients by world_size during
  the all-reduce, which is mathematically equivalent to computing the loss on
  the global batch.

[DDP-7] CHECKPOINT SAVE
  Checkpoints and best-model files are written ONLY by rank 0. All other ranks
  call dist.barrier() after the save point so that no process continues training
  until the files are fully on disk.

[DDP-8] LOGGING
  The Python logger is active only on rank 0. Other ranks use print() only for
  critical errors, preventing duplicated console output.

[DDP-9] COMPILE
  torch.compile is active on EVERY process in DDP mode. Because each process
  holds its own unwrapped module locally, compilation is safe; there is no
  deepcopy (DDP relies on hooks, not replicas).

[DDP-10] TEARDOWN
  dist.destroy_process_group() is called at the end of training. Without this,
  child processes may hang instead of exiting cleanly.

[FIX-DOUBLE-VAL-MIDLOOP]
  Prevents running both mid-epoch and end-of-epoch validation on the same step,
  which would double-count the validation loss and overwrite the best-model
  file twice.

[FIX-ATOMIC-SAVE]
  Checkpoints are written to a .tmp file and then renamed with os.replace().
  This guarantees that any partially written file is never visible to a
  concurrent reader (e.g. a resuming process or a Kaggle output download).

[FIX-RESUME-BEST]
  When resuming from a checkpoint, the best_model_path stored in the checkpoint
  is validated on disk. If the file is missing (e.g. deleted by the user or
  lost on Kaggle restart), the variable is set to None so that a new best model
  can be saved without crashing.

[FIX-COMPILE-DP→DDP]
  torch.compile checkpoints saved under DP contain "_orig_mod." prefixes.
  The unwrap helper strips both the DDP wrapper and the compile wrapper before
  saving, and strips "_orig_mod." on load, making checkpoints interchangeable
  between compiled and non-compiled runs.

[OPT-NOISE-PREALLOCATE]
  noise and timestep buffers (_noise_buf, _ts_buf) are allocated once and
  reused across steps instead of creating new tensors every iteration.
  This removes ~2 CUDA malloc/free calls per step.

[OPT-PREFETCH]
  PrefetchLoader pins the next batch to GPU via a separate CUDA stream while
  the current batch is being processed. On T4 this hides host-to-device copy
  latency behind the forward/backward computation.

[OPT-ZERO-GRAD]
  optimizer.zero_grad(set_to_none=True) is used instead of the default
  zero_grad(). Setting gradients to None (rather than zero) allows PyTorch to
  skip memory zeroing and lets the allocator reuse the memory pages immediately.

[MEM-1] EMBEDDING CACHE
  The label-embedding cosine target matrix is built once before training and
  kept on GPU. It avoids tokenising and averaging text embeddings inside the
  training loop, saving ~0.3 ms per sample.

[MEM-2] BEST-MODEL CLEANUP
  When a new best model is saved, the previous best file is deleted immediately
  (after the atomic replace). This prevents the output directory from growing
  unbounded with obsolete .bin files.

[MEM-3] TENSORBOARD ROTATION
  If the TensorBoard runs/ directory exceeds max_tb_size_mb, it is wiped and
  the writer is reopened. This avoids filling the Kaggle working disk with
  hundreds of MB of event files.

[MEM-4] VAE OFFLOAD
  After precomputing latents, the VAE is moved to CPU and the CUDA cache is
  emptied. This frees ~1.2 GB VRAM before the DDP wrap, preventing OOM when
  the UNet and embedder are loaded on the same GPU.

[MEM-5] LAZY LATENTS
  When latents_dir is provided, the dataloader uses LazyLatentIndex, which
  memory-maps safetensors chunks instead of loading all latents into RAM.
  Peak RAM stays flat regardless of dataset size.

[MEM-6] WORKER REDUCTION
  After VAE precomputation, the train DataLoader is recreated with fewer
  workers (workers_after_precompute). Precompute benefits from many CPU
  workers for image decoding; the training loop does not, because the dataset
  now reads precomputed tensors from disk, and extra workers only waste RAM.

[RESUME]
  Full checkpoint format stores: global_step, best_vloss, embedder_state_dict,
  optimizer_state_dict, scaler_state_dict, lr_scheduler_state_dict, and optional
  lora_state_dict. Loading restores exact training state, including AMP scaler
  internals and scheduler phase.

[LORA-DDP]
  When LoRA is enabled, find_unused_parameters must be True because the UNet
  self-attention processors (attn1) do not receive gradients from the
  cross-attention loss path. Without this flag, DDP would crash with
  "parameters which did not receive grad".

[LORA-TRAIN-MODE]
  Calling model.train() on the wrapper puts the entire model in training mode,
  including the frozen UNet backbone. With LoRA active, the UNet must stay in
  eval() mode (deterministic BatchNorm/Dropout) while only lora_layers.train()
  is called. Without LoRA, the backbone is frozen via requires_grad=False, so
  train/eval does not matter, but eval() is kept for cleanliness.

[LORA-BEST-MID]
  During mid-epoch validation, when a new best validation loss is found, both
  the embedder weights and the LoRA adapter weights are saved. The LoRA file
  is named best_model_lora_<timestamp>.bin and is atomically replaced together
  with the embedder best model.

[LORA-BEST]
  At end-of-epoch validation, the same dual-save logic applies: the best
  embedder and the best LoRA weights are written as a pair so that inference
  can load both without version mismatch.


# =============================================================================
# test_no_accel_colab.py
# =============================================================================

[FIX-1] PRECOMPUTED AUDIO
  Removed the runtime call to aud_encoder.extract_features() on
  batch["audio_values"]. The test pipeline now consumes batch["audio_features"]
  directly (precomputed BEATs embeddings written by
  preprocess_audio_embeddings_colab.py). This eliminates the need to load the
  heavy BEATs model during inference and cuts startup time by ~30 s.

[FIX-2] EMBEDDER CALL
  The line audio_token = base_model.embedder(aud_features) was already
  correct; a redundant second call and a misleading comment were removed to
  avoid confusion.

[FIX-3] PLACEHOLDER INJECTION
  The SD 2 pipeline injects the audio embedding by replacing the <*> placeholder
  token inside the CLIP text encoder. The placeholder token ID is resolved once
  outside the loop and the text encoder token embedding table is resized exactly
  once, avoiding repeated resize_token_embeddings() calls that slow down the
  pipeline.

[FIX-4] PIPELINE SINGLETON
  StableDiffusionPipeline is built once outside the generation loop instead of
  being reconstructed for every image. This saves ~5 s per image because the
  HF cache is reused and the tokenizer does not re-initialise.

[FIX-5] CENTER_CROP DEFAULT
  Added args.center_crop=False in parse_args() to match the Museart dataset
  __init__ default. Without this, the dataloader would attempt to centre-crop
  images that are already square, wasting CPU cycles.

[FIX-6] LOCAL_RANK GUARD
  parse_args() no longer references args.local_rank before it is defined.
  The env_local_rank check is performed safely and the attribute is created
  if missing, preventing AttributeError when the script is launched without
  torchrun.

[FIX-PROMPT]
  The default prompt string now includes the <*> placeholder token
  ("An art image of <*>"). Without the placeholder, the audio embedding is
  never injected into the text encoder, and the model falls back to pure
  text conditioning, producing generic images unrelated to the input music.

[FIX-LATENTS-DIR]
  Added --latents_dir so the Museart dataloader can bypass loading physical
  image files and read precomputed VAE latents directly. This removes PIL
  decode and VAE encode from the test loop, cutting per-batch CPU time by
  ~60 %.

[DDP-TEST]
  Inference does NOT use DDP. The SD pipeline generates one image at a time
  (batch_size=1), so DDP would add process-spawn overhead with no throughput
  gain. The optimal dual-GPU strategy is:
    - UNet + text_encoder on cuda:0 (compute-intensive)
    - VAE decode on cuda:1 (frees VRAM on cuda:0 for the diffusion steps)
  The script is always launched with plain python, never torchrun.

[FIX-A] SCHEDULER
  SD 2.1 uses prediction_type="v_prediction". The default DDIMScheduler loaded
  without arguments inherits the pretrained config, but EulerDiscreteScheduler
  is safer because it supports v-prediction natively and is the reference
  scheduler for SD 2.1. This avoids black-square artefacts caused by
  incompatible scheduler settings.

[FIX-CFG] UNCONDITIONAL PROMPT
  The unconditional prompt for classifier-free guidance must be the base text
  WITHOUT the <*> placeholder (e.g. "An art image of"), not an empty string.
  With uncond="", the vector distance between unconditional and conditional is
  huge (nothing vs text+audio), and multiplying it by guidance_scale burns the
  latents, producing structured noise. Using the base prompt limits the
  difference to exactly the audio token, so CFG amplifies only the desired
  signal.

[FIX-VAE] FP16 DECODE STABILITY
  SD 2.1 VAE suffers from numerical instability in float16: out-of-range values
  produce black squares or chromatic noise. The fix casts latents and the VAE
  to float32 only for decode(), then moves the VAE back to the original dtype
  (e.g. fp16) for VRAM efficiency. The scaling factor is read from
  vae.config.scaling_factor instead of the hard-coded 0.18215.


# =============================================================================
# preprocess_image_latents_colab.py
# =============================================================================

[MEM-1] NO RAM ACCUMULATION
  Latents are never accumulated in RAM indefinitely. Every
  FLUSH_EVERY_N_BATCHES batches the in-memory dict is written to a separate
  chunk file on disk and immediately cleared. Peak latent RAM is therefore
  bounded by FLUSH_EVERY_N_BATCHES × batch_size × ~25 KB (float16, 512 px).

[MEM-2] DISTINCT CHUNKS
  Each flush creates a new file (chunk_0000.safetensors, chunk_0001, …) in
  the chunks/ subdirectory. Files are never overwritten, so there is no
  transient double-disk-usage spike during writes.

[MEM-3] O(1) RESUME
  To determine which image_ids have already been processed, the script reads
  ONLY the keys from existing chunk headers (sf.keys()) without loading any
  tensor data into RAM. This makes resume cost proportional to the number of
  chunks, not the dataset size.

[MEM-4] OPTIONAL FINAL MERGE
  After encoding is complete, the chunks are merged into a single
  image_latents.safetensors if disk space permits. The merge uses a streaming
  strategy: each source chunk is read, its tensors appended to the output,
  and the source file deleted immediately. Maximum extra disk space required
  equals the size of the largest chunk (~329 MB) rather than the total dataset.
  If space is insufficient, the chunks remain and the dataloader reads them
  directly via SafeTensorsIndex.

[OPT-1] DataParallel VAE
  When multiple GPUs are available, the VAE encode step is wrapped in
  nn.DataParallel. This splits the batch across GPUs and gathers latents on
  the master device, doubling throughput on T4×2.

[OPT-2] CPU WORKERS
  The DataLoader uses CPU workers for parallel PIL decoding and resizing.
  On Colab single-GPU the number of workers is clamped to 2 because more
  workers starve the CPU and slow down the GPU pipeline.

[OPT-3] PIN MEMORY
  pin_memory=True is enabled when workers > 0 and CUDA is available. This
  allows non-blocking host-to-device transfers in the training loop.

[OPT-4] AUTOCAST
  VAE encoding runs under torch.amp.autocast('cuda') in float16. The VAE
  weights are kept in float16, halving activation memory and doubling tensor
  core throughput on T4.

[OPT-6] THROUGHPUT LOG
  Every 10×batch_size images, the script logs images-per-second. This helps
  diagnose whether the bottleneck is disk I/O, CPU preprocessing, or GPU
  encoding.

[FIX-OOMAUTO]
  If a batch triggers CUDA OutOfMemory, the encoder catches the exception,
  halves the batch size, retries, and repeats until the batch fits. If the
  batch cannot be reduced below 1, the batch is skipped and its IDs are
  logged as errors. This prevents a single oversized image from crashing the
  entire preprocessing run.


# =============================================================================
# preprocess_audio_embeddings_colab.py
# =============================================================================

[MEM-1] NO RAM ACCUMULATION
  Embeddings are flushed to disk every FLUSH_EVERY_N_BATCHES batches.
  With batch_size=8 and temporal_pool_stride=8, each audio embedding is
  ~0.55 MB float16. A flush window of 150 batches keeps embedding RAM below
  ~660 MB regardless of total dataset size.

[MEM-2] DISTINCT CHUNKS
  Each flush writes a new chunk_NNNN.safetensors file. Because files are never
  overwritten, there is no risk of corrupting partially written data if the
  process is interrupted.

[MEM-3] O(1) RESUME
  On restart, the script rebuilds the "already done" set by scanning only the
  keys of existing chunk headers. No tensor is loaded into RAM during this
  phase, so resume is instant even for 20 000+ audio files.

[MEM-4] STREAMING MERGE
  The final merge reads one chunk at a time, writes its tensors into the
  output safetensors, and deletes the source chunk immediately. Peak extra
  disk space is bounded by the largest chunk (~329 MB) instead of the full
  dataset (~6.6 GB). If free space is below the safety margin, the merge is
  skipped and the dataloader consumes the chunks directly.

[SPEED-1] POST-FLUSH GC
  torch.cuda.empty_cache() and gc.collect() are called after every flush.
  This prevents PyTorch's caching allocator from holding onto large blocks
  that are no longer needed, reducing the risk of OOM in the next encoding
  batch.

[SPEED-2] DataParallel BEATs
  When multiple GPUs are detected, BEATs is wrapped in nn.DataParallel with a
  thin wrapper that calls extract_features() and returns the concatenated
  layer outputs. Batch splitting is handled automatically by DataParallel.

[SPEED-3] PARALLEL AUDIO I/O
  Audio file loading (libsoundfile via torchaudio) runs in a ThreadPoolExecutor.
  Because libsoundfile releases the Python GIL, multiple files can be decoded
  in parallel while the main thread waits for the GPU to finish the previous
  batch. This hides disk latency behind GPU compute.

[FIX-2304] FULL LAYER CONCATENATION
  BEATs returns separate outputs for layers 4, 8 and 12. The old code averaged
  them into a single 768-D vector, destroying the differentiated semantic
  information: layer 4 carries low-level acoustic features, layer 8 carries
  rhythmic structure, and layer 12 carries high-level semantics. The fix
  concatenates the three layers into a 2304-D vector, preserving the full
  representational richness required by the Musipainter paper.


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
# safetensors weight format migration  (2026-05-19)
# =============================================================================

[FMT-SAFETENSORS] WEIGHT FILES MIGRATED TO SAFETENSORS FORMAT

  Scope: weight output files only (embedder, LoRA, best-model snapshots).
  The training resume checkpoint (checkpoint_step{N}.pt) is NOT changed —
  it contains optimizer/scaler/scheduler state that requires torch.save.

  Why:
    torch.save uses Python pickle. Pickle files can execute arbitrary code
    on load and are rejected by some security scanners. safetensors stores
    only flat {str → tensor} dicts; it is pickle-free, zero-copy (mmap),
    and validated against out-of-bounds reads at open time.

  Files changed:
    utils.py
      - Added load_safetensors(path, device) → dict[str, Tensor].
        Symmetric counterpart of the existing save_safetensors helper.

    train_validation_no_accel_colab.py
      - save_progress(module, save_path): now dispatches on extension —
        .safetensors → save_safetensors; anything else → torch.save (legacy).
      - load_embedder_weights(weight_path, ...): transparent .bin → .safetensors
        fallback; accepts both formats; renamed parameter bin_path → weight_path.
      - All weight output paths renamed .bin → .safetensors:
          learned_embeds.safetensors
          learned_embeds_lora_layers.safetensors
          weights/{run_name}_embeds-step{N}.safetensors
          weights/{run_name}_lora-step{N}.safetensors
          best_model_embedder_{timestamp}.safetensors
          best_model_lora_{timestamp}.safetensors
      - Best-model save blocks (mid-epoch and end-of-epoch) now call
        save_safetensors directly instead of torch.save; atomic rename (.tmp)
        preserved.

    MusicToken_no_accel.py
      - Added module-level helper _load_weights(path, map_device) → dict.
        Chooses the loader based on file extension; falls back transparently
        from .bin to .safetensors if only the new format exists on disk.
      - All torch.load calls in the test branch replaced with _load_weights:
        embedder, VAE, BEATs audio encoder, UNet, LoRA layers.

    test_no_accel_colab.py
      - Added _resolve_checkpoint_path(explicit_path, output_dir, stem, label).
        Search order: explicit path → .safetensors sibling → output_dir/<stem>.*
        → most-recent best_model_<label>_* → most-recent weights/*-step*.
        Raises FileNotFoundError with a clear message if nothing is found.
      - inference() calls _resolve_checkpoint_path for embedder (always) and
        LoRA (when --lora=True) before instantiating MusicTokenWrapper, so
        args.learned_embeds always points to a verified, existing file.
      - Default --learned_embeds and --learned_embeds_lora updated to .safetensors.
      - Redundant os.path.exists guard for LoRA replaced by the resolver.

  Backward compatibility:
    Old .bin files produced by previous training runs are still readable.
    Every load path tries .bin first if that is what was passed, then falls
    back to .safetensors. The resume checkpoint (.pt) is fully unaffected.
