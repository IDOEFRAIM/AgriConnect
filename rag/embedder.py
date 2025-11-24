# embedder_milvus.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import threading
import hashlib
import json
import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, Future
import asyncio
from pathlib import Path

_logger = logging.getLogger("embedder_milvus")

# Lazy heavy deps placeholders
SentenceTransformer = None
AutoTokenizer = None
AutoModel = None
torch = None
np = None

# --- CONFIG ---
@dataclass
class RobustEmbedderConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    normalize: bool = True               # L2 normalize vectors for Milvus IP
    device: Optional[str] = None         # "cpu" | "cuda" | None (auto)
    use_8bit: bool = False               # bitsandbytes 8-bit if available
    dtype: str = "float32"               # "float32", "float16"
    max_length: Optional[int] = None
    cache_size: int = 10000
    cache_path: Optional[str] = None
    show_progress: bool = False
    thread_workers: int = 4              # Set to 1 for GPU to avoid CUDA contention
    metrics_hook: Optional[Callable[[Dict[str, Any]], None]] = None
    query_prefix: Optional[str] = None   # "query: " for asymmetric models (BGE, E5)
    doc_prefix: Optional[str] = None     # "passage: " for asymmetric models
    enable_warmup: bool = True           # Auto-warmup on first use
    cache_ttl: Optional[int] = None      # Cache TTL in seconds (None = no expiry)


# --- LRU CACHE WITH TTL ---
class _LRUCache:
    """Thread-safe LRU cache with optional TTL support"""
    
    def __init__(self, capacity: int, ttl: Optional[int] = None):
        self.capacity = max(0, capacity)
        self.ttl = ttl
        self.lock = threading.RLock()
        self.data: Dict[str, Tuple[List[float], float]] = {}  # key -> (vector, timestamp)
        self.order: List[str] = []

    def get(self, key: str) -> Optional[List[float]]:
        if self.capacity == 0:
            return None
        
        with self.lock:
            entry = self.data.get(key)
            if entry is None:
                return None
            
            vector, timestamp = entry
            
            # Check TTL expiry
            if self.ttl and (time.time() - timestamp) > self.ttl:
                self._evict(key)
                return None
            
            # Move to end (most recently used)
            try:
                self.order.remove(key)
            except ValueError:
                pass
            self.order.append(key)
            
            return vector

    def put(self, key: str, value: List[float]) -> None:
        if self.capacity == 0:
            return
        
        with self.lock:
            timestamp = time.time()
            
            if key in self.data:
                try:
                    self.order.remove(key)
                except ValueError:
                    pass
            
            self.data[key] = (value, timestamp)
            self.order.append(key)
            
            # Evict oldest if over capacity
            while len(self.order) > self.capacity:
                old_key = self.order.pop(0)
                self.data.pop(old_key, None)

    def _evict(self, key: str) -> None:
        """Internal eviction (must hold lock)"""
        self.data.pop(key, None)
        try:
            self.order.remove(key)
        except ValueError:
            pass

    def to_dict(self) -> Dict[str, List[float]]:
        """Export cache (vectors only, no timestamps)"""
        with self.lock:
            return {k: v[0] for k, v in self.data.items()}

    def load_dict(self, d: Dict[str, List[float]]) -> None:
        """Import cache from dict"""
        with self.lock:
            timestamp = time.time()
            self.data = {k: (v, timestamp) for k, v in d.items()}
            keys = list(self.data.keys())
            
            if self.capacity and len(keys) > self.capacity:
                keys = keys[-self.capacity:]
                self.data = {k: self.data[k] for k in keys}
            
            self.order = list(self.data.keys())

    def clear(self) -> None:
        """Clear all cached entries"""
        with self.lock:
            self.data.clear()
            self.order.clear()

    def stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        with self.lock:
            return {
                "size": len(self.data),
                "capacity": self.capacity,
                "utilization": int((len(self.data) / max(1, self.capacity)) * 100)
            }


# --- EMBEDDER CLASS ---
class RobustEmbedder:
    """
    Production-grade embedder optimized for Milvus RAG pipelines.
    
    Features:
    - Asymmetric encoding support (query vs document prefixes)
    - Batching with backpressure handling
    - LRU cache with optional TTL
    - Cache persistence (JSON)
    - 8-bit quantization support
    - Thread-safe async wrapper
    - Milvus upsert helper
    - Metrics hooks for observability
    """
    
    def __init__(self, cfg: RobustEmbedderConfig):
        self.cfg = cfg
        self._model = None
        self._tokenizer = None
        self._hf_model = None
        self._dim: Optional[int] = None
        self._cache = _LRUCache(cfg.cache_size, cfg.cache_ttl)
        self._model_lock = threading.RLock()
        self._warmed_up = False
        
        # Executor for async operations
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, cfg.thread_workers),
            thread_name_prefix="embedder"
        )
        
        # Initialize device
        self._init_device()
        
        # Load persisted cache if available
        if cfg.cache_path and Path(cfg.cache_path).exists():
            self._load_cache(cfg.cache_path)
        
        _logger.info(
            f"RobustEmbedder initialized: model={cfg.model_name}, "
            f"device={self.device}, cache_size={cfg.cache_size}"
        )

    # ==================== PUBLIC API ====================
    
    def encode(
        self, 
        texts: Sequence[str], 
        is_query: bool = False,
        show_progress: Optional[bool] = None
    ) -> List[List[float]]:
        """
        Encode texts into L2-normalized vectors.
        
        Args:
            texts: Input texts to encode
            is_query: If True, apply query_prefix; else apply doc_prefix
            show_progress: Override cfg.show_progress
            
        Returns:
            List of vector embeddings (normalized if cfg.normalize=True)
        """
        if not texts:
            return []
        
        # Auto-warmup on first encode
        if self.cfg.enable_warmup and not self._warmed_up:
            self.warmup()
        
        start_time = time.time()
        
        # Apply prefixes (critical for asymmetric models like BGE/E5)
        processed_texts = self._apply_prefixes(texts, is_query)
        
        # Check cache and collect uncached indices
        outputs: List[Optional[List[float]]] = [None] * len(texts)
        to_encode_texts: List[str] = []
        to_encode_indices: List[int] = []
        cache_hits = 0
        
        for i, text in enumerate(processed_texts):
            cache_key = self._cache_key(text)
            cached_vec = self._cache.get(cache_key)
            
            if cached_vec is not None:
                outputs[i] = cached_vec
                cache_hits += 1
            else:
                to_encode_texts.append(text)
                to_encode_indices.append(i)
        
        # Encode uncached texts in batches
        if to_encode_texts:
            self._ensure_model()
            batch_size = max(1, self.cfg.batch_size)
            progress = show_progress if show_progress is not None else self.cfg.show_progress
            
            for batch_start in range(0, len(to_encode_texts), batch_size):
                batch_end = min(batch_start + batch_size, len(to_encode_texts))
                batch_texts = to_encode_texts[batch_start:batch_end]
                batch_indices = to_encode_indices[batch_start:batch_end]
                
                try:
                    batch_vecs = self._encode_batch(batch_texts, progress)
                    
                    for j, vec in enumerate(batch_vecs):
                        output_idx = batch_indices[j]
                        vec_list = self._to_list(vec)
                        outputs[output_idx] = vec_list
                        
                        # Cache document embeddings (queries are highly variable)
                        if not is_query or self.cfg.cache_size > 5000:
                            cache_key = self._cache_key(batch_texts[j])
                            self._cache.put(cache_key, vec_list)
                
                except Exception as e:
                    _logger.error(f"Batch encoding failed: {e}", exc_info=True)
                    # Fallback: zero vectors
                    dim = self.dim()
                    for j in range(len(batch_texts)):
                        output_idx = batch_indices[j]
                        if outputs[output_idx] is None:
                            outputs[output_idx] = [0.0] * dim
        
        # Validate all outputs are filled
        dim = self.dim()
        for i, vec in enumerate(outputs):
            if vec is None:
                _logger.warning(f"Text {i} produced None vector; using zero vector")
                outputs[i] = [0.0] * dim
        
        # Emit metrics
        elapsed = time.time() - start_time
        if self.cfg.metrics_hook:
            try:
                self.cfg.metrics_hook({
                    "event": "encode",
                    "count": len(texts),
                    "cache_hits": cache_hits,
                    "cache_miss": len(to_encode_texts),
                    "is_query": is_query,
                    "elapsed_ms": int(elapsed * 1000),
                    "throughput": len(texts) / max(elapsed, 0.001)
                })
            except Exception as e:
                _logger.debug(f"Metrics hook failed: {e}")
        
        return outputs  # type: ignore

    async def encode_async(
        self, 
        texts: Sequence[str], 
        is_query: bool = False
    ) -> List[List[float]]:
        """Async wrapper for encode() using executor"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, 
            self.encode, 
            texts, 
            is_query
        )

    def dim(self) -> int:
        """Get embedding dimension (lazy init)"""
        if self._dim is None:
            probe_vec = self.encode(["__dimension_probe__"], is_query=False)
            self._dim = len(probe_vec[0])
            _logger.info(f"Embedding dimension: {self._dim}")
        return self._dim

    def warmup(self, samples: Optional[Sequence[str]] = None) -> None:
        """Warmup model by encoding sample texts"""
        if self._warmed_up:
            return
        
        _logger.info("Warming up embedder...")
        self._ensure_model()
        
        if samples:
            warmup_texts = list(samples)[:8]
        else:
            warmup_texts = [
                "This is a warmup sentence.",
                "Another example for GPU memory allocation."
            ]
        
        try:
            self.encode(warmup_texts, is_query=False)
            self._warmed_up = True
            _logger.info("Warmup complete")
        except Exception as e:
            _logger.error(f"Warmup failed: {e}", exc_info=True)

    def save_cache(self, path: Optional[str] = None) -> None:
        """Persist cache to disk (JSON format)"""
        save_path = path or self.cfg.cache_path
        if not save_path:
            _logger.debug("No cache_path configured; skipping save")
            return
        
        try:
            cache_dict = self._cache.to_dict()
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(cache_dict, f, ensure_ascii=False, indent=2)
            
            _logger.info(f"Cache saved: {len(cache_dict)} entries -> {save_path}")
        except Exception as e:
            _logger.error(f"Cache save failed: {e}", exc_info=True)

    def _load_cache(self, path: str) -> None:
        """Load persisted cache from disk"""
        try:
            if not Path(path).exists():
                _logger.debug(f"Cache file not found: {path}")
                return
            
            with open(path, "r", encoding="utf-8") as f:
                cache_dict = json.load(f)
            
            self._cache.load_dict(cache_dict)
            _logger.info(f"Cache loaded: {len(cache_dict)} entries from {path}")
        except Exception as e:
            _logger.error(f"Cache load failed: {e}", exc_info=True)

    def cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self._cache.stats()

    def clear_cache(self) -> None:
        """Clear all cached embeddings"""
        self._cache.clear()
        _logger.info("Cache cleared")

    # ==================== MILVUS INTEGRATION ====================
    
    def upsert_to_milvus(
        self,
        collection,
        docs: Sequence[Dict[str, Any]],
        id_field: str = "chunk_id",
        text_field: str = "text",
        vector_field: str = "vector",
        meta_fields: Optional[Sequence[str]] = None,
        batch_size: int = 256,
        timeout: Optional[float] = None,
        skip_on_error: bool = True,
    ) -> Dict[str, Any]:
        """
        Upsert documents into Milvus collection.
        
        Args:
            collection: pymilvus Collection instance
            docs: List of document dicts with at minimum {id_field, text_field}
            id_field: Primary key field name
            text_field: Field containing text to embed
            vector_field: Field name for embedding vector
            meta_fields: Additional scalar fields to include
            batch_size: Insert batch size
            timeout: Insert timeout in seconds
            skip_on_error: Continue on encoding errors vs raise
            
        Returns:
            Dict with stats: {inserted, failed, errors, elapsed_ms}
        """
        try:
            from pymilvus import Collection
        except ImportError:
            raise RuntimeError(
                "pymilvus not installed. Install with: pip install pymilvus"
            )
        
        start_time = time.time()
        inserted = 0
        failed = 0
        errors: List[str] = []
        meta_fields = list(meta_fields or [])
        
        # Prepare records for insertion
        records_to_insert: List[Dict[str, Any]] = []
        
        for idx, doc in enumerate(docs):
            try:
                doc_id = doc.get(id_field)
                if doc_id is None:
                    raise ValueError(f"Missing {id_field}")
                
                text = doc.get(text_field, "")
                if not text.strip():
                    _logger.warning(f"Empty text for doc {doc_id}")
                    if not skip_on_error:
                        raise ValueError("Empty text")
                    continue
                
                # Encode as document (not query)
                vec = self.encode([text], is_query=False)[0]
                
                # Build record with metadata
                record = {
                    id_field: doc_id,
                    vector_field: vec,
                    text_field: text,
                }
                
                # Add scalar metadata
                for field in meta_fields:
                    if field in doc:
                        record[field] = doc[field]
                
                records_to_insert.append(record)
                
            except Exception as e:
                failed += 1
                error_msg = f"Doc {doc.get(id_field, idx)} prep failed: {str(e)}"
                errors.append(error_msg)
                _logger.error(error_msg, exc_info=True)
                
                if not skip_on_error:
                    raise
        
        # Insert in batches
        for batch_start in range(0, len(records_to_insert), batch_size):
            batch = records_to_insert[batch_start:batch_start + batch_size]
            
            try:
                collection.insert(batch, timeout=timeout)
                inserted += len(batch)
                _logger.debug(f"Inserted batch {batch_start//batch_size + 1}: {len(batch)} records")
            except Exception as e:
                failed += len(batch)
                error_msg = f"Batch {batch_start//batch_size + 1} insert failed: {str(e)}"
                errors.append(error_msg)
                _logger.error(error_msg, exc_info=True)
                
                if not skip_on_error:
                    raise
        
        # Flush to ensure persistence (optional but recommended)
        try:
            collection.flush(timeout=timeout)
        except Exception as e:
            _logger.warning(f"Flush failed: {e}")
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        result = {
            "inserted": inserted,
            "failed": failed,
            "total": len(docs),
            "errors": errors[:10],  # Limit error list
            "elapsed_ms": elapsed_ms,
        }
        
        _logger.info(
            f"Milvus upsert complete: {inserted} inserted, {failed} failed, {elapsed_ms}ms"
        )
        
        return result

    # ==================== INTERNAL METHODS ====================
    
    def _init_device(self) -> None:
        """Auto-detect or set device (CPU/CUDA)"""
        if self.cfg.device:
            self.device = self.cfg.device
            return
        
        try:
            import torch as torch_module
            global torch
            torch = torch_module
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            self.device = "cpu"
        
        _logger.info(f"Device set to: {self.device}")

    def _ensure_model(self) -> None:
        """Lazy-load embedding model (sentence-transformers or HF)"""
        with self._model_lock:
            if self._model is not None or self._hf_model is not None:
                return
            
            # Try sentence-transformers first (simpler API)
            if self._try_load_sentence_transformers():
                return
            
            # Fallback to HuggingFace transformers
            if self._try_load_hf_transformers():
                return
            
            raise RuntimeError(
                "No embedding backend available. Install sentence-transformers or transformers+torch."
            )

    def _try_load_sentence_transformers(self) -> bool:
        """Attempt to load sentence-transformers model"""
        try:
            from sentence_transformers import SentenceTransformer as ST
            global SentenceTransformer
            SentenceTransformer = ST
            
            kwargs = {"device": self.device}
            self._model = SentenceTransformer(self.cfg.model_name, **kwargs)
            
            try:
                self._tokenizer = self._model.tokenizer
            except AttributeError:
                self._tokenizer = None
            
            _logger.info(f"Loaded sentence-transformers: {self.cfg.model_name}")
            return True
            
        except Exception as e:
            _logger.debug(f"sentence-transformers load failed: {e}")
            return False

    def _try_load_hf_transformers(self) -> bool:
        """Attempt to load HuggingFace transformers model"""
        try:
            from transformers import AutoTokenizer as AT, AutoModel as AM
            import torch as torch_module
            import numpy as numpy_module
            
            global AutoTokenizer, AutoModel, torch, np
            AutoTokenizer = AT
            AutoModel = AM
            torch = torch_module
            np = numpy_module
            
            # Handle 8-bit quantization
            if self.cfg.use_8bit:
                try:
                    from transformers import BitsAndBytesConfig
                    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                    self._hf_model = AutoModel.from_pretrained(
                        self.cfg.model_name, 
                        quantization_config=bnb_config
                    )
                    _logger.info("Loaded model with 8-bit quantization")
                except Exception as e:
                    _logger.warning(f"8-bit quantization failed: {e}, loading full model")
                    self._hf_model = AutoModel.from_pretrained(self.cfg.model_name)
            else:
                self._hf_model = AutoModel.from_pretrained(self.cfg.model_name)
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.model_name, 
                use_fast=True
            )
            self._hf_model.to(self.device)
            self._hf_model.eval()
            
            _logger.info(f"Loaded HF transformers: {self.cfg.model_name}")
            return True
            
        except Exception as e:
            _logger.error(f"HF transformers load failed: {e}", exc_info=True)
            return False

    def _apply_prefixes(self, texts: Sequence[str], is_query: bool) -> List[str]:
        """Apply query or document prefix if configured"""
        if is_query and self.cfg.query_prefix:
            return [f"{self.cfg.query_prefix}{text}" for text in texts]
        elif not is_query and self.cfg.doc_prefix:
            return [f"{self.cfg.doc_prefix}{text}" for text in texts]
        else:
            return list(texts)

    def _encode_batch(self, texts: Sequence[str], show_progress: bool = False):
        """Encode a batch using the loaded model"""
        # Sentence-transformers path
        if self._model is not None:
            try:
                arr = self._model.encode(
                    list(texts),
                    batch_size=self.cfg.batch_size,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True,
                    normalize_embeddings=False,  # We handle normalization ourselves
                )
                
                if self.cfg.normalize:
                    arr = self._l2_normalize(arr)
                
                if self.cfg.dtype == "float16":
                    arr = arr.astype("float16")
                
                return [arr[i] for i in range(arr.shape[0])]
            
            except Exception as e:
                _logger.error(f"Sentence-transformers encode failed: {e}", exc_info=True)
                # Try HF fallback if available
                if self._hf_model:
                    return self._hf_encode_mean_pool(texts)
                raise
        
        # HuggingFace transformers path
        if self._hf_model is not None:
            return self._hf_encode_mean_pool(texts)
        
        raise RuntimeError("No model available for encoding")

    def _hf_encode_mean_pool(self, texts: Sequence[str]):
        """Encode using HF model with mean pooling"""
        if self._tokenizer is None or self._hf_model is None:
            raise RuntimeError("HF model/tokenizer not initialized")
        
        # Tokenize
        tok_kwargs = {
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
        }
        if self.cfg.max_length:
            tok_kwargs["max_length"] = self.cfg.max_length
        
        encoded = self._tokenizer(list(texts), **tok_kwargs)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self._hf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Mean pooling
            token_embeddings = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
            sum_embeddings = (token_embeddings * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            
            # To numpy
            arr = mean_pooled.cpu().numpy()
            
            if self.cfg.normalize:
                arr = self._l2_normalize(arr)
            
            if self.cfg.dtype == "float16":
                arr = arr.astype("float16")
            
            return [arr[i] for i in range(arr.shape[0])]

    @staticmethod
    def _l2_normalize(arr):
        """L2 normalize vectors (critical for Milvus IP metric)"""
        try:
            import numpy as _np
            arr = arr.astype('float32') if arr.dtype != 'float32' else arr
            norms = _np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1e-12
            return arr / norms
        except Exception:
            # Pure Python fallback
            result = []
            for row in arr:
                norm = sum(x * x for x in row) ** 0.5
                if norm == 0:
                    result.append(row)
                else:
                    result.append([x / norm for x in row])
            return _np.array(result, dtype='float32')

    @staticmethod
    def _to_list(vec) -> List[float]:
        """Convert numpy/torch array to Python list"""
        try:
            return vec.tolist()
        except AttributeError:
            return [float(x) for x in vec]

    @staticmethod
    def _cache_key(text: str) -> str:
        """Generate cache key from text (SHA256 hash)"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def __del__(self):
        """Cleanup: save cache and shutdown executor"""
        try:
            self.save_cache()
        except Exception:
            pass
        
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass



        'i'