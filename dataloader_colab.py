# @title dataloader_colab.py
"""
dataloader_colab.py — Dataset and lazy-loading indices for the Musipainter pipeline.

Provides:
  LazyEmbeddingIndex  — lazy or preloaded index for audio embeddings
                        (.safetensors, chunked .safetensors, or legacy .pt)
  LazyLatentIndex     — lazy or preloaded index for precomputed VAE latents
  Museart             — PyTorch Dataset pairing audio files with WikiArt images
                        by shared thematic class
"""

import collections
import PIL
import random
import numpy as np
from packaging import version
from PIL import Image
import os
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
import logging

from utils import safe_torch_load as _safe_torch_load, open_safetensors

logger = logging.getLogger(__name__)

# Resolve the correct PIL resampling constants for PIL >= 9.1.
if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }

imagenet_templates_small = ["an art image of {}"]


class LazyEmbeddingIndex:
    """
    Provides access to precomputed BEATs audio embeddings from disk.

    Supports three storage formats (auto-detected in priority order):
      1. audio_embeddings.safetensors      — single safetensors file
      2. chunks/chunk_*.safetensors        — chunked safetensors (when merge failed)
      3. audio_embeddings_*.pt             — legacy torch.save chunks

    Two access modes (controlled by preload_all):
      preload_all=True  (default) — entire dataset loaded into RAM at init;
                                    zero I/O during training. Dict is shared
                                    across DataLoader workers via fork (copy-on-write).
      preload_all=False           — lazy mmap via safe_open; each get_tensor()
                                    reads only that tensor's bytes. Ideal when
                                    available RAM < ~2 GB.

    For chunked safetensors with preload_all=False, an LRU cache of file handles
    (max_sf_handles) avoids re-opening the same chunk file on every access.
    """

    def __init__(self, embeddings_dir: str, preload_all: bool = True,
                 max_sf_handles: int = 4):
        self.embeddings_dir  = Path(embeddings_dir)
        self._preload_all    = preload_all
        self._index: dict    = {}   # audio_id → chunk_path (str) or True
        self._flat_cache: dict = {} # audio_id → tensor  (when preload_all=True)
        self._sf             = None # SafeOpen handle for single-file mode
        self._lru_cache: dict          = {}
        self._lru: collections.deque   = collections.deque()
        self._max_cached_chunks: int   = 4
        self._sf_handles: dict         = {}   # chunk_path → SafeOpen handle (chunked mode)
        self._sf_lru: collections.deque = collections.deque()
        self._max_sf_handles: int      = max_sf_handles
        self._mode: str                = "unknown"
        self._build_index()

    def _build_index(self):
        sf_file      = self.embeddings_dir / "audio_embeddings.safetensors"
        chunks_dir   = self.embeddings_dir / "chunks"
        chunk_files  = sorted(chunks_dir.glob("chunk_*.safetensors")) if chunks_dir.exists() else []
        legacy_files = sorted(self.embeddings_dir.glob("audio_embeddings_*.pt"))

        # ── SAFETENSORS (single file) ──────────────────────────────────────
        if sf_file.exists():
            self._mode     = "safetensors"
            total_size_mb  = sf_file.stat().st_size / (1024 ** 2)
            mode_label     = "preload RAM" if self._preload_all else "lazy mmap"
            logger.info(
                f"LazyEmbeddingIndex [safetensors | {mode_label}]: "
                f"{sf_file.name}  ({total_size_mb:.1f} MB)"
            )
            self._sf = open_safetensors(sf_file)
            keys     = list(self._sf.keys())
            sample_shape = None

            if self._preload_all:
                for audio_id in keys:
                    t = self._sf.get_tensor(audio_id)
                    t = t.squeeze(0) if t.dim() == 3 else t
                    self._flat_cache[audio_id] = t
                    self._index[audio_id]      = True
                    if sample_shape is None:
                        sample_shape = list(t.shape)
                self._sf = None  # handle no longer needed
            else:
                for audio_id in keys:
                    self._index[audio_id] = True
                if keys:
                    sample_shape = list(self._sf.get_tensor(keys[0]).shape)

            logger.info("=" * 60)
            logger.info("[AUDIO EMBEDDINGS] safetensors loaded successfully:")
            logger.info(f"Format        : safetensors")
            logger.info(f"Mode          : {mode_label}")
            logger.info(f"Total audio   : {len(self._index)}")
            logger.info(f"File size     : {total_size_mb:.1f} MB")
            logger.info(f"Tensor shape  : {sample_shape}  (dtype: float16)")
            logger.info(f"File          : {sf_file}")
            if self._preload_all:
                logger.info(f"In RAM        : {len(self._flat_cache)} embeddings")
            logger.info("=" * 60)

        # ── CHUNKED SAFETENSORS (chunks/chunk_NNNN.safetensors) ───────────
        elif chunk_files:
            self._mode = "chunked_safetensors"
            total_size_bytes = sum(f.stat().st_size for f in chunk_files)
            total_size_mb    = total_size_bytes / (1024 ** 2)
            mode_label       = "preload RAM" if self._preload_all else "lazy mmap (LRU handle)"
            logger.info(
                f"LazyEmbeddingIndex [chunked_safetensors | {mode_label}]: "
                f"{len(chunk_files)} chunks in {chunks_dir}  ({total_size_mb:.1f} MB total)"
            )
            sample_shape = None

            for chunk_path in chunk_files:
                sf = open_safetensors(chunk_path)
                chunk_path_str = str(chunk_path)
                # Single pass over sf.keys() to avoid scanning the header twice.
                # preload_all=True  → _index[id] = True   (tensor already in _flat_cache)
                # preload_all=False → _index[id] = path   (opened on demand via LRU)
                for audio_id in sf.keys():
                    if self._preload_all:
                        t = sf.get_tensor(audio_id)
                        t = t.squeeze(0) if t.dim() == 3 else t
                        self._flat_cache[audio_id] = t
                        self._index[audio_id] = True
                        if sample_shape is None:
                            sample_shape = list(t.shape)
                    else:
                        self._index[audio_id] = chunk_path_str
                        if sample_shape is None:
                            t = sf.get_tensor(audio_id)
                            sample_shape = list(t.shape)
                # Close the handle; with preload_all=True tensors are already cached;
                # with preload_all=False handles are re-opened on demand via the LRU cache.
                del sf

            logger.info("=" * 60)
            logger.info("[AUDIO EMBEDDINGS] chunked safetensors loaded:")
            logger.info(f"Format        : chunked safetensors")
            logger.info(f"Mode          : {mode_label}")
            logger.info(f"Chunk files   : {len(chunk_files)}")
            logger.info(f"Total audio   : {len(self._index)}")
            logger.info(f"Total size    : {total_size_mb:.1f} MB")
            logger.info(f"Tensor shape  : {sample_shape}  (dtype: float16)")
            logger.info(f"Chunks dir    : {chunks_dir}")
            if self._preload_all:
                logger.info(f"In RAM        : {len(self._flat_cache)} embeddings")
            else:
                logger.info(f"Handle LRU    : max {self._max_sf_handles} open simultaneously")
            logger.info("=" * 60)

        # ── LEGACY .pt ─────────────────────────────────────────────────────
        elif legacy_files:
            self._mode = "legacy_pt"
            logger.warning(
                f"LazyEmbeddingIndex: no audio_embeddings.safetensors found. "
                f"Using {len(legacy_files)} legacy .pt chunks. "
                f"Re-run preprocess_audio_embeddings_colab.py to migrate to safetensors."
            )
            total_size_bytes = sum(f.stat().st_size for f in legacy_files)
            total_size_mb    = total_size_bytes / (1024 ** 2)
            sample_shape     = None

            for chunk_file in legacy_files:
                chunk = _safe_torch_load(chunk_file)
                for audio_id, tensor in chunk.items():
                    self._index[audio_id] = chunk_file.name
                    if sample_shape is None:
                        sample_shape = list(tensor.shape)
                    if self._preload_all:
                        t = tensor.squeeze(0) if tensor.dim() == 3 else tensor
                        self._flat_cache[audio_id] = t
                if not self._preload_all:
                    del chunk

            logger.info("=" * 60)
            logger.info("[AUDIO EMBEDDINGS] legacy .pt chunks loaded:")
            logger.info(f"Format        : torch .pt  (legacy)")
            logger.info(f"Chunk files   : {len(legacy_files)}")
            logger.info(f"Total audio   : {len(self._index)}")
            logger.info(f"Total size    : {total_size_mb:.1f} MB")
            logger.info(f"Tensor shape  : {sample_shape}  (dtype: float16)")
            if self._preload_all:
                logger.info(f"In RAM        : {len(self._flat_cache)} embeddings")
            logger.info("=" * 60)

        else:
            raise FileNotFoundError(
                f"No audio_embeddings.safetensors, chunks/ dir, or audio_embeddings_*.pt "
                f"found in {self.embeddings_dir}. "
                "Run preprocess_audio_embeddings_colab.py first."
            )

    def __contains__(self, audio_id: str) -> bool:
        return audio_id in self._index

    def __len__(self) -> int:
        return len(self._index)

    def get(self, audio_id: str) -> torch.Tensor:
        if audio_id not in self._index:
            raise KeyError(
                f"Audio '{audio_id}' not found in embeddings. "
                "Re-run preprocess_audio_embeddings_colab.py."
            )

        # preload_all=True → serve directly from RAM (same for all modes)
        if self._preload_all:
            t = self._flat_cache[audio_id]
            return t if t.is_contiguous() else t.contiguous()

        # preload_all=False, single safetensors → lazy mmap read
        if self._mode == "safetensors":
            t = self._sf.get_tensor(audio_id)
            t = t.squeeze(0) if t.dim() == 3 else t
            return t if t.is_contiguous() else t.contiguous()

        # preload_all=False, chunked safetensors → LRU cache of file handles
        if self._mode == "chunked_safetensors":
            chunk_path_str = self._index[audio_id]
            if chunk_path_str not in self._sf_handles:
                # Evict the oldest handle if the cache is full.
                if len(self._sf_handles) >= self._max_sf_handles:
                    oldest = self._sf_lru.popleft()
                    del self._sf_handles[oldest]
                self._sf_handles[chunk_path_str] = open_safetensors(chunk_path_str)
                self._sf_lru.append(chunk_path_str)
            else:
                # Move to most-recently-used position.
                try:
                    self._sf_lru.remove(chunk_path_str)
                except ValueError:
                    pass
                self._sf_lru.append(chunk_path_str)
            t = self._sf_handles[chunk_path_str].get_tensor(audio_id)
            t = t.squeeze(0) if t.dim() == 3 else t
            return t if t.is_contiguous() else t.contiguous()

        # preload_all=False, legacy .pt → LRU cache of loaded chunk dicts
        chunk_name = self._index[audio_id]
        if chunk_name not in self._lru_cache:
            if len(self._lru_cache) >= self._max_cached_chunks:
                oldest = self._lru.popleft()
                del self._lru_cache[oldest]
            self._lru_cache[chunk_name] = _safe_torch_load(
                self.embeddings_dir / chunk_name
            )
            self._lru.append(chunk_name)
        else:
            try:
                self._lru.remove(chunk_name)
            except ValueError:
                pass
            self._lru.append(chunk_name)
        tensor = self._lru_cache[chunk_name][audio_id]
        tensor = tensor.squeeze(0) if tensor.dim() == 3 else tensor
        return tensor if tensor.is_contiguous() else tensor.contiguous()


class LazyLatentIndex:
    """
    Provides access to precomputed VAE image latents from disk.

    Mirrors the structure of LazyEmbeddingIndex but for image latents
    generated by preprocess_image_latents_colab.py.
    Each value is a [4, H/8, W/8] float16 tensor (e.g. [4, 64, 64] at 512px).

    Supports safetensors (preferred) and legacy .pt chunks.
    preload_all=True loads the entire index into RAM (~37 MB for 1500 images).
    preload_all=False uses lazy mmap via safe_open.
    """

    def __init__(self, latents_dir: str, preload_all: bool = True):
        self.latents_dir    = Path(latents_dir)
        self._preload_all   = preload_all
        self._index: dict   = {}
        self._flat_cache: dict = {}
        self._sf            = None
        self._lru_cache: dict          = {}
        self._lru: collections.deque   = collections.deque()
        self._max_cached_chunks: int   = 4
        self._mode: str                = "unknown"
        self._build_index()

    def _build_index(self):
        sf_file      = self.latents_dir / "image_latents.safetensors"
        legacy_files = sorted(self.latents_dir.glob("image_latents_*.pt"))

        # ── SAFETENSORS ────────────────────────────────────────────────────
        if sf_file.exists():
            self._mode    = "safetensors"
            total_size_mb = sf_file.stat().st_size / (1024 ** 2)
            mode_label    = "preload RAM" if self._preload_all else "lazy mmap"
            logger.info(
                f"LazyLatentIndex [safetensors | {mode_label}]: "
                f"{sf_file.name}  ({total_size_mb:.1f} MB)"
            )
            self._sf     = open_safetensors(sf_file)
            keys         = list(self._sf.keys())
            sample_shape = None

            if self._preload_all:
                for image_id in keys:
                    t = self._sf.get_tensor(image_id)
                    self._flat_cache[image_id] = t
                    self._index[image_id]      = True
                    if sample_shape is None:
                        sample_shape = list(t.shape)
                self._sf = None
            else:
                for image_id in keys:
                    self._index[image_id] = True
                if keys:
                    sample_shape = list(self._sf.get_tensor(keys[0]).shape)

            logger.info("=" * 60)
            logger.info("[IMAGE LATENTS] safetensors loaded successfully:")
            logger.info(f"Format        : safetensors")
            logger.info(f"Mode          : {mode_label}")
            logger.info(f"Total images  : {len(self._index)}")
            logger.info(f"File size     : {total_size_mb:.1f} MB")
            logger.info(f"Tensor shape  : {sample_shape}  (dtype: float16)")
            logger.info(f"File          : {sf_file}")
            if self._preload_all:
                logger.info(f"In RAM        : {len(self._flat_cache)} latents")
            logger.info("=" * 60)

        # ── LEGACY .pt ─────────────────────────────────────────────────────
        elif legacy_files:
            self._mode = "legacy_pt"
            logger.warning(
                f"LazyLatentIndex: no image_latents.safetensors found. "
                f"Using {len(legacy_files)} legacy .pt chunks. "
                f"Re-run preprocess_image_latents_colab.py to migrate to safetensors."
            )
            total_size_bytes = sum(f.stat().st_size for f in legacy_files)
            total_size_mb    = total_size_bytes / (1024 ** 2)
            sample_shape     = None

            for chunk_file in legacy_files:
                chunk = _safe_torch_load(chunk_file)
                for image_id, tensor in chunk.items():
                    self._index[image_id] = chunk_file.name
                    if sample_shape is None:
                        sample_shape = list(tensor.shape)
                    if self._preload_all:
                        self._flat_cache[image_id] = tensor
                if not self._preload_all:
                    del chunk

            logger.info("=" * 60)
            logger.info("[IMAGE LATENTS] legacy .pt chunks loaded:")
            logger.info(f"Format        : torch .pt  (legacy)")
            logger.info(f"Chunk files   : {len(legacy_files)}")
            logger.info(f"Total images  : {len(self._index)}")
            logger.info(f"Total size    : {total_size_mb:.1f} MB")
            logger.info(f"Tensor shape  : {sample_shape}  (dtype: float16)")
            if self._preload_all:
                logger.info(f"In RAM        : {len(self._flat_cache)} latents")
            logger.info("=" * 60)

        else:
            raise FileNotFoundError(
                f"No image_latents.safetensors or image_latents_*.pt "
                f"found in {self.latents_dir}. "
                "Run preprocess_image_latents_colab.py first."
            )

    def __contains__(self, image_id: str) -> bool:
        return image_id in self._index

    def __len__(self) -> int:
        return len(self._index)

    def get(self, image_id: str) -> torch.Tensor:
        if image_id not in self._index:
            raise KeyError(
                f"Image '{image_id}' not found in precomputed latents. "
                "Re-run preprocess_image_latents_colab.py."
            )

        # preload_all=True → serve from RAM
        if self._preload_all:
            t = self._flat_cache[image_id]
            return t if t.is_contiguous() else t.contiguous()

        # preload_all=False, safetensors → lazy mmap
        if self._mode == "safetensors":
            t = self._sf.get_tensor(image_id)
            return t if t.is_contiguous() else t.contiguous()

        # preload_all=False, legacy .pt → LRU cache of chunk dicts
        chunk_name = self._index[image_id]
        if chunk_name not in self._lru_cache:
            if len(self._lru_cache) >= self._max_cached_chunks:
                oldest = self._lru.popleft()
                del self._lru_cache[oldest]
            self._lru_cache[chunk_name] = _safe_torch_load(
                self.latents_dir / chunk_name
            )
            self._lru.append(chunk_name)
        else:
            try:
                self._lru.remove(chunk_name)
            except ValueError:
                pass
            self._lru.append(chunk_name)
        t = self._lru_cache[chunk_name][image_id]
        return t if t.is_contiguous() else t.contiguous()


class Museart(Dataset):
    """
    PyTorch Dataset for the Museart music-to-art task.

    Each sample pairs a WAV audio file with a WikiArt image that belongs to
    the same thematic class (e.g. "Impressionism", "Romanticism"). Audio and
    image paths are resolved from CSV manifests; embeddings and latents are
    served via LazyEmbeddingIndex / LazyLatentIndex to avoid runtime inference.
    """

    def __init__(
        self,
        args,
        tokenizer,
        logger,
        size=512,
        interpolation='bicubic',
        preloaded_embeddings=None,
    ):
        self.tokenizer = tokenizer
        self.size = size
        self.placeholder_token = args.placeholder_token
        self.data_set = args.data_set
        self.input_length = args.input_length

        self.image_root_dir = os.path.join(args.data_dir, 'images', self.data_set)
        self.audio_root_dir = os.path.join(args.data_dir, 'audio', self.data_set)

        # Accept CSV in either <data_dir>/images/images.csv or <data_dir>/images.csv
        csv_img_path = os.path.join(args.data_dir, 'images', 'images.csv')
        if not os.path.exists(csv_img_path):
            csv_img_path = os.path.join(args.data_dir, 'images.csv')
        csv_aud_path = os.path.join(args.data_dir, 'audio', 'audio.csv')
        if not os.path.exists(csv_aud_path):
            csv_aud_path = os.path.join(args.data_dir, 'audio.csv')

        self.df_image = pd.read_csv(csv_img_path)
        self.df_audio = pd.read_csv(csv_aud_path)

        # Ensure IDs are strings and the 'set' column is lowercase for safe comparison.
        for col in ['id']:
            if col in self.df_audio.columns:
                self.df_audio[col] = self.df_audio[col].astype(str)
            if col in self.df_image.columns:
                self.df_image[col] = self.df_image[col].astype(str)
        for df in [self.df_audio, self.df_image]:
            if 'set' in df.columns:
                df['set'] = df['set'].str.lower()

        self.image_path: list = []
        self.audio_path: list = []
        self.label: list = []

        self.center_crop = args.center_crop if self.data_set in ('train', 'validation') \
            else False

        # Walk the audio/image directories once at init with followlinks=True
        # for robust Colab/Kaggle symlink support.
        self.audio_id_to_path = {}
        for root, dirs, files in os.walk(self.audio_root_dir, followlinks=True):
            for f in files:
                p = Path(root) / f
                if p.suffix.lower() == '.wav':
                    self.audio_id_to_path[p.stem] = str(p)

        self.image_id_to_path = {}
        for root, dirs, files in os.walk(self.image_root_dir, followlinks=True):
            for f in files:
                p = Path(root) / f
                if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                    self.image_id_to_path[p.stem] = str(p)

        self.df_music  = self.df_audio[self.df_audio["set"] == self.data_set]
        self.df_images = self.df_image[self.df_image["set"] == self.data_set]
        self.prepare_dataset(set(self.audio_id_to_path.keys()))

        self.num_samples = len(self.audio_path)
        self._length = self.num_samples
        logger.info(f"{self.data_set}: {self.num_samples} samples")

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]
        self.templates = imagenet_templates_small

        # Pre-tokenise all templates once; txt_proc() just samples from this list.
        # Tokenisation costs ~0.5 ms/call; pre-computing saves ~80 s over 20 k steps.
        # Tensors are made contiguous here so there is zero overhead in the hot loop.
        _placeholder = args.placeholder_token
        self._cached_input_ids = [
            self.tokenizer(
                tmpl.format(_placeholder),
                padding="max_length", truncation=True,
                max_length=self.tokenizer.model_max_length, return_tensors="pt",
            ).input_ids[0].contiguous()
            for tmpl in self.templates
        ]
        # Cache len to avoid calling len() inside __getitem__.
        self._n_templates = len(self._cached_input_ids)

        if preloaded_embeddings is not None:
            self.audio_embeddings = preloaded_embeddings
            logger.info(f"Embeddings: {type(preloaded_embeddings).__name__} "
                        f"({len(self.audio_embeddings)} audio)")
        elif hasattr(args, 'embeddings_dir') and args.embeddings_dir:
            self.audio_embeddings = LazyEmbeddingIndex(args.embeddings_dir)
            logger.info(f"LazyEmbeddingIndex: {len(self.audio_embeddings)} audio")
        else:
            self.audio_embeddings = None
            logger.warning("No embeddings_dir specified.")

        # If image_latents_dir is given, bypass img_proc() and serve precomputed latents.
        image_latents_dir = getattr(args, 'image_latents_dir', None)
        if image_latents_dir:
            self.image_latents = LazyLatentIndex(image_latents_dir)
            logger.info(
                f"LazyLatentIndex: {len(self.image_latents)} image latents "
                f"(img_proc bypassed)"
            )
        else:
            self.image_latents = None
            logger.info("image_latents_dir not specified: using img_proc() at runtime.")

        # Pre-allocated constant bool tensors to avoid torch.tensor(True/False)
        # on every __getitem__ call (~320 k allocations saved over a 20 k-step run).
        self._TRUE  = torch.tensor(True)
        self._FALSE = torch.tensor(False)

    def __len__(self):
        return self._length

    def prepare_dataset(self, samples_audio):
        """
        Populate self.audio_path / image_path / label lists.

        Uses set_index + groupby for O(1) per-sample lookup instead of the
        naive O(n²) DataFrame scan inside the loop.
        """
        music_indexed = self.df_music.set_index('id')
        images_by_class = {
            cls: grp['id'].tolist()
            for cls, grp in self.df_images.groupby('class')
        }

        for aud_id in list(samples_audio):
            if aud_id not in music_indexed.index:
                continue
            label = music_indexed.loc[aud_id, 'class']
            if hasattr(label, '__iter__') and not isinstance(label, str):
                label = label.iloc[0]

            candidate_ids = images_by_class.get(label)
            if not candidate_ids:
                continue
            image_id = random.choice(candidate_ids)

            # Try exact id, then lowercased+underscored variant, then disk search.
            _img_id_str  = str(image_id)
            _img_id_norm = _img_id_str.lower().replace(" ", "_")
            img_path = (
                self.image_id_to_path.get(_img_id_str)
                or self.image_id_to_path.get(_img_id_norm)
            )
            if not img_path:
                # Direct disk search: try both underscore and dash filename variants.
                cls_folder = str(label).replace(" ", "_").replace("/", "-")
                cls_dir = Path(self.image_root_dir) / cls_folder
                _found = False
                _id_with_dashes = _img_id_str.replace("_", "-")
                for ext in ('.jpg', '.jpeg', '.png', '.webp', '.tiff'):
                    for _name in (_img_id_str, _id_with_dashes):
                        _candidate = cls_dir / f"{_name}{ext}"
                        if _candidate.exists():
                            img_path = str(_candidate)
                            self.image_id_to_path[_img_id_str] = img_path  # cache hit
                            _found = True
                            break
                    if _found:
                        break
                if not _found:
                    img_path = str(cls_dir / f"{_img_id_str}.jpg")

            self.audio_path.append(self.audio_id_to_path[aud_id])
            self.label.append(label)
            self.image_path.append(img_path)

    def img_proc(self, image_path: str) -> torch.Tensor:
        """Load, centre-crop, resize and normalise an image to [-1, 1]."""
        image_file = Image.open(image_path).convert("RGB")
        img = np.array(image_file, dtype=np.uint8)
        if self.center_crop:
            h, w = img.shape[0], img.shape[1]
            crop = min(h, w)
            img = img[(h - crop) // 2:(h + crop) // 2,
                      (w - crop) // 2:(w + crop) // 2]
        image = Image.fromarray(img).resize((self.size, self.size), resample=self.interpolation)
        image = (np.array(image, dtype=np.float32) / 127.5 - 1.0)
        return torch.from_numpy(image).permute(2, 0, 1)

    def aud_proc_beats(self, aud_path: str) -> torch.Tensor:
        """Retrieve a precomputed audio embedding for the given file path."""
        if self.audio_embeddings is None:
            raise ValueError("audio_embeddings not available. Specify --embeddings_dir.")
        audio_id = Path(aud_path).stem
        if isinstance(self.audio_embeddings, LazyEmbeddingIndex):
            t = self.audio_embeddings.get(audio_id)
            # Ensure contiguity once here in the worker thread so collate can use fast memcpy.
            return t if t.is_contiguous() else t.contiguous()
        feat = self.audio_embeddings.get(audio_id, None)
        if feat is None:
            raise KeyError(f"Audio '{audio_id}' not found in embeddings.")
        feat = feat.squeeze(0) if feat.dim() == 3 else feat
        return feat if feat.is_contiguous() else feat.contiguous()

    def txt_proc(self) -> torch.Tensor:
        """Return a random pre-tokenised template input_ids tensor."""
        # random.randrange avoids the internal len() call inside random.choice.
        return self._cached_input_ids[random.randrange(self._n_templates)]

    def __getitem__(self, idx):
        # Retry loop: if an image or latent is missing, try the next sample
        # instead of crashing the DataLoader worker.
        for _retry in range(self.num_samples):
            _idx = (idx + _retry) % self.num_samples
            aud_path   = self.audio_path[_idx]
            image_path = self.image_path[_idx]
            image_id   = Path(image_path).stem

            _img_id_norm = image_id.lower().replace(" ", "_")

            if self.image_latents is not None:
                if image_id in self.image_latents:
                    pixel_values = self.image_latents.get(image_id)
                    is_precomputed_latent = self._TRUE
                    break
                elif _img_id_norm in self.image_latents:
                    pixel_values = self.image_latents.get(_img_id_norm)
                    image_id = _img_id_norm
                    is_precomputed_latent = self._TRUE
                    break
                else:
                    # Keep the batch homogeneous [B, 4, H/8, W/8]: no fallback to
                    # img_proc() to avoid shape mismatches in the collate function.
                    if _retry == 0:
                        logger.warning(
                            f"[SKIP] missing latent for image_id='{image_id}' "
                            f"(path='{image_path}'). Trying next sample."
                        )
                    continue
            else:
                if os.path.isfile(image_path):
                    pixel_values = self.img_proc(image_path)
                    is_precomputed_latent = self._FALSE
                    break
                else:
                    if _retry == 0:
                        logger.warning(
                            f"[SKIP] image_path not found: '{image_path}'. "
                            "Trying next sample."
                        )
                    continue
        else:
            raise RuntimeError(
                f"No valid sample found starting from idx={idx}. "
                "Check that images and latents are present and paths are correct."
            )

        return {
            "input_ids":             self.txt_proc(),
            "aud_id":                Path(aud_path).stem,
            "image_id":              image_id,
            "label":                 self.label[_idx],
            "pixel_values":          pixel_values,
            "is_precomputed_latent": is_precomputed_latent,
            "audio_features":        self.aud_proc_beats(aud_path),
        }
