# @title dataloader_colab.py
"""
Dataset and lazy-loading indices for the Musipainter pipeline.

LazyEmbeddingIndex accepts a COMMA-SEPARATED STRING of directories so that
audio embeddings spread across multiple read-only Kaggle Datasets can be
treated as one unified index by the training loop.

Supported directory layouts (auto-detected, mixed layouts are fine):
  1. <dir>/audio_embeddings.safetensors   — single merged file
  2. <dir>/chunks/<class>_chunk_*.safetensors  — per-class chunks
  3. <dir>/chunks/chunk_*.safetensors          — generic chunks
  4. <dir>/audio_embeddings_*.pt               — legacy torch.save chunks
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

from modules.preprocess.utils import safe_torch_load as _safe_torch_load, open_safetensors
from modules.preprocess.argparse_multiembedding import LazyEmbeddingIndex

logger = logging.getLogger(__name__)

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear":   PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic":  PIL.Image.Resampling.BICUBIC,
        "lanczos":  PIL.Image.Resampling.LANCZOS,
        "nearest":  PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear":   PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic":  PIL.Image.BICUBIC,
        "lanczos":  PIL.Image.LANCZOS,
        "nearest":  PIL.Image.NEAREST,
    }

imagenet_templates_small = ["an art image of {}"]


# ─────────────────────────────────────────────────────────────────────────────
#  LazyEmbeddingIndex  (multi-dataset, multi-layout)
# ─────────────────────────────────────────────────────────────────────────────

class LazyEmbeddingIndex:
    """
    Unified lazy index over one or more directories of precomputed BEATs audio embeddings.

    The *embeddings_dirs* argument accepts a comma-separated string of directory paths
    so that embeddings spread across multiple read-only Kaggle Datasets can be used
    transparently. If the same audio ID appears in more than one directory, the last
    occurrence wins — a warning is logged.

    Access modes (controlled by *preload_all*):
      True  (default): load all tensors into RAM at init → zero I/O during training.
      False:           mmap / LRU-cached file handles → O(1) RAM regardless of dataset size.
    """

    def __init__(
        self,
        embeddings_dirs: str,
        preload_all:     bool = True,
        max_sf_handles:  int  = 8,
    ):
        raw_dirs = [d.strip() for d in str(embeddings_dirs).split(",") if d.strip()]
        self._dirs: list[Path] = [Path(d) for d in raw_dirs]

        self._preload_all    = preload_all
        self._max_sf_handles = max_sf_handles

        self._index:      dict = {}   # audio_id → True (preloaded) or str path (lazy)
        self._flat_cache: dict = {}   # audio_id → tensor  (preload_all=True)
        self._sf_handles: dict = {}   # path_str  → SafeOpen handle (lazy mode)
        self._sf_lru:     collections.deque = collections.deque()
        self._lru_cache:  dict = {}   # path_str  → loaded .pt chunk dict (legacy lazy)
        self._lru:        collections.deque = collections.deque()
        self._max_cached_chunks = 4

        self._total_ids     = 0
        self._total_size_mb = 0.0

        valid_dirs = [d for d in self._dirs if d.exists()]
        if not valid_dirs:
            raise FileNotFoundError(
                f"LazyEmbeddingIndex: none of the following directories exist:\n"
                + "\n".join(f"  {d}" for d in self._dirs)
            )
        missing = set(self._dirs) - set(valid_dirs)
        if missing:
            logger.warning(
                f"LazyEmbeddingIndex: {len(missing)} director(ies) not found and skipped:\n"
                + "\n".join(f"  {d}" for d in sorted(missing))
            )
        self._dirs = valid_dirs
        self._build_index()

    # ──────────────────────────────────────────────────────────────────────────
    def _build_index(self) -> None:
        mode_label = "preload RAM" if self._preload_all else "lazy mmap"
        logger.info(
            f"LazyEmbeddingIndex [{mode_label}] scanning "
            f"{len(self._dirs)} director(ies)..."
        )

        sample_shape: list | None = None
        total_files_scanned = 0

        for d in self._dirs:
            n_before   = len(self._index)
            sf_files   = self._collect_sf_files(d)

            if not sf_files:
                logger.warning(f"  No safetensors files found in {d} — skipped.")
                # Still try legacy .pt files
                self._index_legacy_pt(d, sample_shape)
                continue

            total_files_scanned += len(sf_files)

            for sf_path in sf_files:
                self._total_size_mb += sf_path.stat().st_size / (1024 ** 2)
                try:
                    sf = open_safetensors(sf_path)
                    chunk_path_str = str(sf_path)

                    for audio_id in sf.keys():
                        if audio_id in self._index:
                            logger.debug(
                                f"  Duplicate audio_id '{audio_id}' — "
                                f"overwriting with {sf_path.name}"
                            )
                        if self._preload_all:
                            t = sf.get_tensor(audio_id)
                            t = t.squeeze(0) if t.dim() == 3 else t
                            self._flat_cache[audio_id] = t
                            self._index[audio_id]      = True
                            if sample_shape is None:
                                sample_shape = list(t.shape)
                        else:
                            self._index[audio_id] = chunk_path_str
                            if sample_shape is None:
                                t = sf.get_tensor(audio_id)
                                sample_shape = list(t.shape)

                    if self._preload_all:
                        del sf   # handle no longer needed

                except Exception as exc:
                    logger.warning(f"  Cannot read {sf_path}: {exc} — skipped.")

            self._index_legacy_pt(d, sample_shape)

            n_added = len(self._index) - n_before
            logger.info(f"  {d}  →  {n_added} audio IDs")

        self._total_ids = len(self._index)

        logger.info("=" * 60)
        logger.info("[AUDIO EMBEDDINGS]  index built:")
        logger.info(f"  Directories      : {len(self._dirs)}")
        logger.info(f"  Files scanned    : {total_files_scanned}")
        logger.info(f"  Total audio IDs  : {self._total_ids}")
        logger.info(f"  Total size       : {self._total_size_mb:.1f} MB")
        logger.info(f"  Sample shape     : {sample_shape}  (float16)")
        logger.info(f"  Mode             : {mode_label}")
        if self._preload_all:
            logger.info(f"  In RAM           : {len(self._flat_cache)} tensors")
        else:
            logger.info(f"  LRU handles      : max {self._max_sf_handles}")
        logger.info("=" * 60)

    def _collect_sf_files(self, d: Path) -> list[Path]:
        """
        Collect all safetensors files in directory *d* following priority order:
          1. Single merged file at root.
          2. Per-class or generic chunks in chunks/ subdirectory.
          3. Any *.safetensors at root level.
        """
        files: list[Path] = []

        # Priority 1: single merged file
        root_sf = d / "audio_embeddings.safetensors"
        if root_sf.exists():
            files.append(root_sf)
            return files

        # Priority 2: chunks/ subdirectory (per-class or generic)
        chunks_dir = d / "chunks"
        if chunks_dir.exists():
            found = sorted(chunks_dir.glob("*.safetensors"))
            if found:
                files.extend(found)
                return files

        # Priority 3: any .safetensors at root
        root_sfs = sorted(d.glob("*.safetensors"))
        if root_sfs:
            files.extend(root_sfs)

        return files

    def _index_legacy_pt(self, d: Path, sample_shape: list | None) -> None:
        """Index legacy torch.save .pt chunks from *d* (backward compatibility)."""
        legacy_files = sorted(d.glob("audio_embeddings_*.pt"))
        if not legacy_files:
            return
        logger.warning(
            f"  [LEGACY] Found {len(legacy_files)} .pt chunks in {d}. "
            "Consider re-running preprocess to produce safetensors chunks."
        )
        for chunk_file in legacy_files:
            self._total_size_mb += chunk_file.stat().st_size / (1024 ** 2)
            try:
                chunk = _safe_torch_load(chunk_file)
                for audio_id, tensor in chunk.items():
                    if self._preload_all:
                        t = tensor.squeeze(0) if tensor.dim() == 3 else tensor
                        self._flat_cache[audio_id] = t
                        self._index[audio_id]      = True
                    else:
                        self._index[audio_id] = str(chunk_file)
                if not self._preload_all:
                    del chunk
            except Exception as exc:
                logger.warning(f"  Cannot read {chunk_file}: {exc} — skipped.")

    # ──────────────────────────────────────────────────────────────────────────
    def __contains__(self, audio_id: str) -> bool:
        return audio_id in self._index

    def __len__(self) -> int:
        return self._total_ids

    def get(self, audio_id: str) -> torch.Tensor:
        if audio_id not in self._index:
            raise KeyError(
                f"Audio '{audio_id}' not found in any of: "
                + ", ".join(str(d) for d in self._dirs)
                + ". Re-run preprocess_audio_embeddings_colab.py."
            )

        # preload_all=True → serve directly from RAM
        if self._preload_all:
            t = self._flat_cache[audio_id]
            return t if t.is_contiguous() else t.contiguous()

        source = self._index[audio_id]

        if source is True:
            raise RuntimeError(
                f"Internal error: index['{audio_id}'] is True but preload_all=False"
            )

        if source.endswith(".safetensors"):
            return self._get_from_sf(audio_id, source)
        else:
            return self._get_from_pt(audio_id, source)

    def _get_from_sf(self, audio_id: str, sf_path_str: str) -> torch.Tensor:
        """LRU-cached safe_open handle → single tensor read."""
        if sf_path_str not in self._sf_handles:
            if len(self._sf_handles) >= self._max_sf_handles:
                oldest = self._sf_lru.popleft()
                self._sf_handles.pop(oldest, None)
            self._sf_handles[sf_path_str] = open_safetensors(sf_path_str)
            self._sf_lru.append(sf_path_str)
        else:
            try:
                self._sf_lru.remove(sf_path_str)
            except ValueError:
                pass
            self._sf_lru.append(sf_path_str)

        t = self._sf_handles[sf_path_str].get_tensor(audio_id)
        t = t.squeeze(0) if t.dim() == 3 else t
        return t if t.is_contiguous() else t.contiguous()

    def _get_from_pt(self, audio_id: str, pt_path_str: str) -> torch.Tensor:
        """LRU-cached .pt chunk dict → single tensor access."""
        chunk_name = pt_path_str
        if chunk_name not in self._lru_cache:
            if len(self._lru_cache) >= self._max_cached_chunks:
                oldest = self._lru.popleft()
                self._lru_cache.pop(oldest, None)
            self._lru_cache[chunk_name] = _safe_torch_load(chunk_name)
            self._lru.append(chunk_name)
        else:
            try:
                self._sf_lru.remove(sf_path_str)
            except ValueError:
                pass
            self._lru.append(chunk_name)

        tensor = self._lru_cache[chunk_name][audio_id]
        tensor = tensor.squeeze(0) if tensor.dim() == 3 else tensor
        return tensor if tensor.is_contiguous() else tensor.contiguous()


# ─────────────────────────────────────────────────────────────────────────────
#  LazyLatentIndex
# ─────────────────────────────────────────────────────────────────────────────

class LazyLatentIndex:
    """
    Provides access to precomputed VAE image latents from disk.

    Each value is a [4, H/8, W/8] float16 tensor (e.g. [4, 64, 64] at 512px).
    Supports safetensors (preferred) and legacy .pt chunks.
    preload_all=True loads the entire index into RAM.
    preload_all=False uses lazy mmap via safe_open.
    """

    def __init__(self, latents_dir: str, preload_all: bool = True):
        self.latents_dir    = Path(latents_dir)
        self._preload_all   = preload_all
        self._index: dict   = {}
        self._flat_cache: dict = {}
        self._sf            = None
        self._lru_cache: dict         = {}
        self._lru: collections.deque  = collections.deque()
        self._max_cached_chunks: int  = 4
        self._mode: str               = "unknown"
        self._build_index()

    def _build_index(self):
        sf_file      = self.latents_dir / "image_latents.safetensors"
        legacy_files = sorted(self.latents_dir.glob("image_latents_*.pt"))

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
            logger.info("[IMAGE LATENTS] safetensors loaded:")
            logger.info(f"  Total images : {len(self._index)}")
            logger.info(f"  Size         : {total_size_mb:.1f} MB")
            logger.info(f"  Shape        : {sample_shape}  (float16)")
            logger.info(f"  Mode         : {mode_label}")
            logger.info("=" * 60)

        elif legacy_files:
            self._mode = "legacy_pt"
            logger.warning(
                f"LazyLatentIndex: no image_latents.safetensors found — "
                f"using {len(legacy_files)} legacy .pt chunks. "
                "Re-run preprocess_image_latents_colab.py to migrate."
            )
            for chunk_file in legacy_files:
                chunk = _safe_torch_load(chunk_file)
                for image_id, tensor in chunk.items():
                    self._index[image_id] = chunk_file.name
                    if self._preload_all:
                        self._flat_cache[image_id] = tensor
                if not self._preload_all:
                    del chunk

            logger.info("=" * 60)
            logger.info("[IMAGE LATENTS] legacy .pt chunks loaded:")
            logger.info(f"  Total images : {len(self._index)}")
            logger.info(f"  Size         : {total_size_mb:.1f} MB")
            logger.info(f"  Shape        : {sample_shape}  (float16)")
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

        if self._preload_all:
            t = self._flat_cache[image_id]
            return t if t.is_contiguous() else t.contiguous()

        if self._mode == "safetensors":
            t = self._sf.get_tensor(image_id)
            return t if t.is_contiguous() else t.contiguous()

        # legacy .pt — LRU cache of chunk dicts
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


# ─────────────────────────────────────────────────────────────────────────────
#  Museart Dataset
# ─────────────────────────────────────────────────────────────────────────────

class Museart(Dataset):
    """
    PyTorch Dataset for the Museart music-to-art task (Early Fusion branch).

    Each sample pairs a WAV audio file with a WikiArt image from the same class.
    Audio embeddings and image latents are served via LazyEmbeddingIndex / LazyLatentIndex.
    The LazyEmbeddingIndex accepts a comma-separated string of directories so multiple
    Kaggle Datasets can be used transparently.
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
        self.tokenizer         = tokenizer
        self.size              = size
        self.placeholder_token = args.placeholder_token
        self.data_set          = args.data_set
        self.input_length      = args.input_length

        self.image_root_dir = os.path.join(args.data_dir, 'images', self.data_set)
        self.audio_root_dir = os.path.join(args.data_dir, 'audio',  self.data_set)

        # CSV resolution: try canonical path, fall back to root
        csv_img_path = os.path.join(args.data_dir, 'images', 'images.csv')
        if not os.path.exists(csv_img_path):
            csv_img_path = os.path.join(args.data_dir, 'images.csv')
        csv_aud_path = os.path.join(args.data_dir, 'audio', 'audio.csv')
        if not os.path.exists(csv_aud_path):
            csv_aud_path = os.path.join(args.data_dir, 'audio.csv')

        self.df_image = pd.read_csv(csv_img_path)
        self.df_audio = pd.read_csv(csv_aud_path)

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
        self.label:      list = []

        self.center_crop = (
            args.center_crop if self.data_set in ('train', 'validation') else False
        )

        # Walk directories once at init with followlinks=True
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
        self._length     = self.num_samples
        logger.info(f"{self.data_set}: {self.num_samples} samples")

        self.interpolation = {
            "linear":   PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic":  PIL_INTERPOLATION["bicubic"],
            "lanczos":  PIL_INTERPOLATION["lanczos"],
        }[interpolation]
        self.templates = imagenet_templates_small

        _placeholder = args.placeholder_token
        self._cached_input_ids = [
            self.tokenizer(
                tmpl.format(_placeholder),
                padding="max_length", truncation=True,
                max_length=self.tokenizer.model_max_length, return_tensors="pt",
            ).input_ids[0].contiguous()
            for tmpl in self.templates
        ]
        self._n_templates = len(self._cached_input_ids)

        # ── Audio embeddings index ─────────────────────────────────────────
        if preloaded_embeddings is not None:
            self.audio_embeddings = preloaded_embeddings
            logger.info(
                f"Embeddings: {type(preloaded_embeddings).__name__} "
                f"({len(self.audio_embeddings)} audio)"
            )
        elif hasattr(args, 'embeddings_dir') and args.embeddings_dir:
            self.audio_embeddings = LazyEmbeddingIndex(
                args.embeddings_dir,
                preload_all    = getattr(args, 'embeddings_preload_all', True),
                max_sf_handles = getattr(args, 'embeddings_max_sf_handles', 8),
            )
            logger.info(
                f"LazyEmbeddingIndex: {len(self.audio_embeddings)} audio  "
                f"(dirs: {args.embeddings_dir})"
            )
        else:
            self.audio_embeddings = None
            logger.warning("No embeddings_dir specified — audio_embeddings unavailable.")

        # ── Image latents index ────────────────────────────────────────────
        image_latents_dir = getattr(args, 'image_latents_dir', None)
        if image_latents_dir:
            self.image_latents = LazyLatentIndex(image_latents_dir)
            logger.info(
                f"LazyLatentIndex: {len(self.image_latents)} latents "
                f"(img_proc bypassed)"
            )
        else:
            self.image_latents = None
            logger.info("image_latents_dir not set: using img_proc() at runtime.")

        # Pre-allocated constant bool tensors (avoids allocation in hot loop)
        self._TRUE  = torch.tensor(True)
        self._FALSE = torch.tensor(False)

    def __len__(self):
        return self._length

    def prepare_dataset(self, samples_audio):
        """
        Populate audio_path / image_path / label lists.
        Uses set_index + groupby for O(1) per-sample lookup.
        """
        music_indexed   = self.df_music.set_index('id')
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

            _img_id_str  = str(image_id)
            _img_id_norm = _img_id_str.lower().replace(" ", "_")
            img_path = (
                self.image_id_to_path.get(_img_id_str)
                or self.image_id_to_path.get(_img_id_norm)
            )
            if not img_path:
                cls_folder = str(label).replace(" ", "_").replace("/", "-")
                cls_dir    = Path(self.image_root_dir) / cls_folder
                _found     = False
                _id_dash   = _img_id_str.replace("_", "-")
                for ext in ('.jpg', '.jpeg', '.png', '.webp', '.tiff'):
                    for _name in (_img_id_str, _id_dash):
                        _cand = cls_dir / f"{_name}{ext}"
                        if _cand.exists():
                            img_path = str(_cand)
                            self.image_id_to_path[_img_id_str] = img_path
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
        image_file = Image.open(image_path).convert("RGB")
        img = np.array(image_file, dtype=np.uint8)
        if self.center_crop:
            h, w = img.shape[0], img.shape[1]
            crop = min(h, w)
            img  = img[(h - crop) // 2:(h + crop) // 2,
                       (w - crop) // 2:(w + crop) // 2]
        image = Image.fromarray(img).resize(
            (self.size, self.size), resample=self.interpolation
        )
        image = (np.array(image, dtype=np.float32) / 127.5 - 1.0)
        return torch.from_numpy(image).permute(2, 0, 1)

    def aud_proc_beats(self, aud_path: str) -> torch.Tensor:
        if self.audio_embeddings is None:
            raise ValueError(
                "audio_embeddings not available. Specify --embeddings_dir."
            )
        audio_id = Path(aud_path).stem
        if isinstance(self.audio_embeddings, LazyEmbeddingIndex):
            t = self.audio_embeddings.get(audio_id)
            return t if t.is_contiguous() else t.contiguous()
        # Fallback for dict-like preloaded embeddings
        feat = self.audio_embeddings.get(audio_id, None)
        if feat is None:
            raise KeyError(f"Audio '{audio_id}' not found in embeddings.")
        feat = feat.squeeze(0) if feat.dim() == 3 else feat
        return feat if feat.is_contiguous() else feat.contiguous()

    def txt_proc(self) -> torch.Tensor:
        """Return a random pre-tokenised template input_ids tensor."""
        return self._cached_input_ids[random.randrange(self._n_templates)]

    def __getitem__(self, idx):
        # Retry loop: if an image or latent is missing, try the next sample.
        for _retry in range(self.num_samples):
            _idx       = (idx + _retry) % self.num_samples
            aud_path   = self.audio_path[_idx]
            image_path = self.image_path[_idx]
            image_id   = Path(image_path).stem
            _img_id_norm = image_id.lower().replace(" ", "_")

            if self.image_latents is not None:
                if image_id in self.image_latents:
                    pixel_values          = self.image_latents.get(image_id)
                    is_precomputed_latent = self._TRUE
                    break
                elif _img_id_norm in self.image_latents:
                    pixel_values          = self.image_latents.get(_img_id_norm)
                    image_id              = _img_id_norm
                    is_precomputed_latent = self._TRUE
                    break
                else:
                    if _retry == 0:
                        logger.warning(
                            f"[SKIP] missing latent for image_id='{image_id}'. "
                            "Trying next sample."
                        )
                    continue
            else:
                if os.path.isfile(image_path):
                    pixel_values          = self.img_proc(image_path)
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
                "Check that images and latents are present."
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
