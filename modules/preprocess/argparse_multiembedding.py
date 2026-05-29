# modules/preprocess/argparse_multiembedding.py

import argparse
import logging
import collections
import torch
from pathlib import Path
from modules.preprocess.utils import safe_torch_load as _safe_torch_load, open_safetensors

logger = logging.getLogger(__name__)

def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', '1'):
        return True
    if v.lower() in ('no', 'false', '0'):
        return False
    raise argparse.ArgumentTypeError(f"Boolean expected, got: '{v}'")

def add_multiembedding_args(parser: argparse.ArgumentParser) -> None:
    try:
        parser.add_argument(
            "--embeddings_dir",
            type=str,
            default="./audio_embeddings/",
            help=(
                "Comma-separated paths to directories containing precomputed "
                "BEATs audio embeddings. Supports merged safetensors, per-class chunks, "
                "or legacy .pt files across multiple Kaggle Dataset mounts."
            ),
        )
    except argparse.ArgumentError:
        pass

    try:
        parser.add_argument(
            "--embeddings_preload_all",
            type=_str2bool,
            default=True,
            help="True: load all tensors into RAM at init. False: lazy mmap mode.",
        )
    except argparse.ArgumentError:
        pass

    try:
        parser.add_argument(
            "--embeddings_max_sf_handles",
            type=int,
            default=8,
            help="LRU cache size for open safetensors file handles (lazy mmap mode).",
        )
    except argparse.ArgumentError:
        pass


class LazyEmbeddingIndex:
    """
    Unified lazy index over one or more directories of precomputed BEATs audio embeddings.
    """
    def __init__(self, embeddings_dirs: str, preload_all: bool = True, max_sf_handles: int = 8):
        raw_dirs = [d.strip() for d in str(embeddings_dirs).split(",") if d.strip()]
        self._dirs: list[Path] = [Path(d) for d in raw_dirs]
        self._preload_all    = preload_all
        self._max_sf_handles = max_sf_handles

        self._index:      dict = {}   
        self._flat_cache: dict = {}   
        self._sf_handles: dict = {}   
        self._sf_lru:     collections.deque = collections.deque()
        self._lru_cache:  dict = {}   
        self._lru:        collections.deque = collections.deque()
        self._max_cached_chunks = 4

        self._total_ids     = 0
        self._total_size_mb = 0.0

        valid_dirs = [d for d in self._dirs if d.exists()]
        if not valid_dirs:
            raise FileNotFoundError("LazyEmbeddingIndex: none of the specified directories exist.")
        self._dirs = valid_dirs
        self._build_index()

    def _build_index(self) -> None:
        mode_label = "preload RAM" if self._preload_all else "lazy mmap"
        logger.info(f"LazyEmbeddingIndex [{mode_label}] scanning {len(self._dirs)} director(ies)...")

        sample_shape: list | None = None
        total_files_scanned = 0

        for d in self._dirs:
            n_before   = len(self._index)
            sf_files   = self._collect_sf_files(d)

            if not sf_files:
                logger.warning(f"  No safetensors files found in {d} — attempting legacy conversion.")
                self._index_legacy_pt(d, sample_shape)
                continue

            total_files_scanned += len(sf_files)

            for sf_path in sf_files:
                self._total_size_mb += sf_path.stat().st_size / (1024 ** 2)
                try:
                    sf = open_safetensors(sf_path)
                    chunk_path_str = str(sf_path)

                    for audio_id in sf.keys():
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
                        del sf
                except Exception as exc:
                    logger.warning(f"  Cannot read {sf_path}: {exc} — skipped.")

            self._index_legacy_pt(d, sample_shape)
            n_added = len(self._index) - n_before
            logger.info(f"  {d}  →  {n_added} audio IDs")

        self._total_ids = len(self._index)
        logger.info(f"[AUDIO EMBEDDINGS] Index built: {self._total_ids} IDs across {self._total_size_mb:.1f} MB.")

    def _collect_sf_files(self, d: Path) -> list[Path]:
        files: list[Path] = []
        root_sf = d / "audio_embeddings.safetensors"
        if root_sf.exists():
            files.append(root_sf)
            return files
        chunks_dir = d / "chunks"
        if chunks_dir.exists():
            found = sorted(chunks_dir.glob("*.safetensors"))
            if found:
                files.extend(found)
                return files
        root_sfs = sorted(d.glob("*.safetensors"))
        if root_sfs:
            files.extend(root_sfs)
        return files

    def _index_legacy_pt(self, d: Path, sample_shape: list | None) -> None:
        legacy_files = sorted(d.glob("audio_embeddings_*.pt"))
        if not legacy_files:
            return
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

    def __contains__(self, audio_id: str) -> bool:
        return audio_id in self._index

    def __len__(self) -> int:
        return self._total_ids

    def get(self, audio_id: str) -> torch.Tensor:
        if audio_id not in self._index:
            raise KeyError(f"Audio '{audio_id}' not found in index.")
        if self._preload_all:
            t = self._flat_cache[audio_id]
            return t if t.is_contiguous() else t.contiguous()

        source = self._index[audio_id]
        if source.endswith(".safetensors"):
            return self._get_from_sf(audio_id, source)
        else:
            return self._get_from_pt(audio_id, source)

    def _get_from_sf(self, audio_id: str, sf_path_str: str) -> torch.Tensor:
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
        chunk_name = pt_path_str
        if chunk_name not in self._lru_cache:
            if len(self._lru_cache) >= self._max_cached_chunks:
                oldest = self._lru.popleft()
                self._lru_cache.pop(oldest, None)
            self._lru_cache[chunk_name] = _safe_torch_load(chunk_name)
            self._lru.append(chunk_name)
        else:
            try:
                self._lru.remove(chunk_name)  # CORRETTO: rimosso riferimento a sf_path_str
            except ValueError:
                pass
            self._lru.append(chunk_name)

        tensor = self._lru_cache[chunk_name][audio_id]
        tensor = tensor.squeeze(0) if tensor.dim() == 3 else tensor
        return tensor if tensor.is_contiguous() else tensor.contiguous()


def build_embedding_index(args, log=None):
    _log = log or logger
    embeddings_dir = getattr(args, 'embeddings_dir', None)
    if not embeddings_dir:
        _log.warning("build_embedding_index: args.embeddings_dir is empty.")
        return None

    preload_all    = getattr(args, 'embeddings_preload_all', True)
    max_sf_handles = getattr(args, 'embeddings_max_sf_handles', 8)

    index = LazyEmbeddingIndex(
        embeddings_dirs = embeddings_dir,
        preload_all     = preload_all,
        max_sf_handles  = max_sf_handles,
    )
    return index