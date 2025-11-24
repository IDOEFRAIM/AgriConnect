from __future__ import annotations
import asyncio
import time
import logging
import threading
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

_logger = logging.getLogger("cross_encoder")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False
    _logger.warning("NumPy not available")

# Lazy imports for heavy deps
AutoTokenizer = None
AutoModelForSequenceClassification = None
torch = None


def _lazy_import_transformers():
    """Import transformers and torch only when needed"""
    global AutoTokenizer, AutoModelForSequenceClassification, torch
    if AutoTokenizer is None:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
        except ImportError as e:
            raise ImportError(
                "transformers and torch required. Install: pip install transformers torch"
            ) from e


# ==================== CONFIGURATION ====================

@dataclass
class CrossEncoderConfig:
    """Configuration for cross-encoder reranker"""
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: Optional[str] = None  # "cpu" | "cuda" | None (auto)
    batch_size: int = 16
    max_length: int = 512
    use_8bit: bool = False
    dtype: str = "float32"  # "float32" | "float16"
    load_in_4bit: bool = False  # GPTQ/4bit quantization
    score_normalization: str = "sigmoid"  # "sigmoid" | "softmax" | "none"
    timeout_s: Optional[float] = 10.0
    max_workers: int = 4
    cache_enabled: bool = True
    cache_size: int = 10000
    warmup_on_init: bool = True
    metrics_hook: Optional[Any] = None


# ==================== CACHE ====================

class _CrossEncoderCache:
    """Thread-safe LRU cache for cross-encoder scores"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = max(1, capacity)
        self.lock = threading.RLock()
        self.cache: OrderedDict[str, float] = OrderedDict()

    def get(self, key: str) -> Optional[float]:
        """Get cached score"""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None
    
    def put(self, key: str, score: float) -> None:
        """Put score in cache"""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.cache[key] = score
            else:
                if len(self.cache) >= self.capacity:
                    # Remove the least recently used item (first in OrderedDict)
                    self.cache.popitem(last=False)
                self.cache[key] = score
    
    def clear(self) -> int:
        """Clear cache"""
        with self.lock:
            count = len(self.cache)
            self.cache.clear()
            return count
    
    def size(self) -> int:
        """Get cache size"""
        with self.lock:
            return len(self.cache)
    
    @staticmethod
    def make_key(query: str, doc_text: str) -> str:
        """Generate cache key using MD5 hash"""
        combined = f"{query}||{doc_text}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()


# ==================== CROSS-ENCODER ====================

class CrossEncoder:
    """Cross-encoder reranker using transformer models (PyTorch/Hugging Face)"""
    
    def __init__(self, cfg: CrossEncoderConfig):
        self.cfg = cfg
        self._model = None
        self._tokenizer = None
        self._device = None
        # Thread pool for synchronous execution in async context
        self._executor = ThreadPoolExecutor(max_workers=cfg.max_workers)
        self._cache = _CrossEncoderCache(cfg.cache_size) if cfg.cache_enabled else None
        self._cache_hits = 0
        self._cache_misses = 0
        self._lock = threading.Lock()
        
        self._init_model()
        
        if cfg.warmup_on_init:
            self._warmup()
    
    def _init_model(self) -> None:
        """Initialize model and tokenizer, handle device placement and quantization"""
        try:
            _lazy_import_transformers()
            
            _logger.info(f"Loading cross-encoder: {self.cfg.model_name}")
            
            # Determine device
            if self.cfg.device:
                self._device = self.cfg.device
            elif torch.cuda.is_available():
                self._device = "cuda"
            else:
                self._device = "cpu"
            
            _logger.info(f"Using device: {self._device}")
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
            
            # Load model with quantization options
            model_kwargs = {}
            if self.cfg.load_in_4bit:
                model_kwargs["load_in_4bit"] = True
                model_kwargs["device_map"] = "auto"
            elif self.cfg.use_8bit:
                model_kwargs["load_in_8bit"] = True
                model_kwargs["device_map"] = "auto"
            
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.cfg.model_name,
                **model_kwargs
            )
            
            # Move to device if not handled by quantization (device_map="auto")
            if not (self.cfg.load_in_4bit or self.cfg.use_8bit):
                self._model = self._model.to(self._device)
            
            # Set dtype (e.g., half precision for CUDA)
            if self.cfg.dtype == "float16" and self._device == "cuda":
                self._model = self._model.half()
            
            self._model.eval()
            
            _logger.info("Model loaded successfully")
            
        except Exception as e:
            _logger.error(f"Failed to load model: {e}", exc_info=True)
            raise
    
    def _warmup(self) -> None:
        """Run a dummy inference batch to compile kernels and load necessary layers"""
        try:
            _logger.info("Warming up model...")
            dummy_pairs = [("warmup query", "warmup document")] * min(4, self.cfg.batch_size)
            # The result is intentionally ignored
            self._score_batch(dummy_pairs)
            _logger.info("Warmup complete")
        except Exception as e:
            _logger.warning(f"Warmup failed: {e}")
    
    def _score_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Score batch of query-document pairs (core inference logic)"""
        if not pairs:
            return []
        
        try:
            # Tokenize pairs: [Query, Document]
            features = self._tokenizer(
                [p[0] for p in pairs],
                [p[1] for p in pairs],
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
                return_tensors="pt"
            )
            
            # Move inputs to device
            features = {k: v.to(self._device) for k, v in features.items()}
            
            with torch.no_grad():
                outputs = self._model(**features)
                logits = outputs.logits
                
                # Apply score normalization as configured
                if self.cfg.score_normalization == "sigmoid":
                    scores = torch.sigmoid(logits).squeeze(-1)
                elif self.cfg.score_normalization == "softmax":
                    # Assuming binary classification where class 1 is positive/relevant
                    scores = torch.softmax(logits, dim=-1)[:, 1]
                else:
                    # Raw logits
                    scores = logits.squeeze(-1)
                
                # Convert scores to standard Python list
                scores_list = scores.cpu().tolist()
                
                # Handle single item batch edge case
                if isinstance(scores_list, float):
                    return [scores_list]
                
                return scores_list
        
        except Exception as e:
            _logger.error(f"Batch scoring failed: {e}", exc_info=True)
            return [0.0] * len(pairs)
    
    def _score_with_cache(self, query: str, docs: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], float]]:
        """Score documents, prioritizing cache hits and grouping misses for batch inference"""
        results = []
        pairs_to_score = []
        doc_indices = []
        
        # 1. Check cache for all documents
        for idx, doc in enumerate(docs):
            doc_text = doc.get("text", "")
            
            if self._cache:
                cache_key = _CrossEncoderCache.make_key(query, doc_text)
                cached_score = self._cache.get(cache_key)
                
                if cached_score is not None:
                    self._cache_hits += 1
                    # Store (doc, score, original_index)
                    results.append((doc, cached_score, idx))
                    continue
                else:
                    self._cache_misses += 1
            
            # If not cached, prepare for batch scoring
            pairs_to_score.append((query, doc_text))
            doc_indices.append(idx) # Keep track of original position
        
        # 2. Score uncached pairs in batches
        if pairs_to_score:
            batch_size = self.cfg.batch_size
            for i in range(0, len(pairs_to_score), batch_size):
                batch = pairs_to_score[i:i + batch_size]
                batch_indices = doc_indices[i:i + batch_size]
                
                scores = self._score_batch(batch)
                
                for (q, d), score, idx in zip(batch, scores, batch_indices):
                    # Store new score in cache
                    if self._cache:
                        cache_key = _CrossEncoderCache.make_key(q, d)
                        self._cache.put(cache_key, float(score))
                    
                    # Store result
                    results.append((docs[idx], float(score), idx))
        
        # 3. Restore original document order (important for non-reranking scenarios)
        results.sort(key=lambda x: x[2])
        return [(doc, score) for doc, score, _ in results]
    
    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Rerank documents synchronously based on relevance to the query"""
        if not docs:
            return []
        
        start_time = time.time()
        
        try:
            # Score documents (using cache if enabled)
            scored_docs = self._score_with_cache(query, docs)
            
            reranked = []
            for doc, score in scored_docs:
                doc_copy = doc.copy()
                # Store the score in the document metadata
                doc_copy["cross_encoder_score"] = float(score)
                doc_copy["score"] = float(score) # Use "score" for consistent sorting
                reranked.append(doc_copy)
            
            # Sort documents by the new relevance score (highest first)
            reranked.sort(key=lambda x: x["score"], reverse=True)
            
            # Apply top_k cut-off
            if top_k:
                reranked = reranked[:top_k]
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            # Optional metrics hook
            if self.cfg.metrics_hook:
                try:
                    self.cfg.metrics_hook({
                        "event": "cross_encoder_rerank",
                        "model_name": self.cfg.model_name,
                        "num_docs_in": len(docs),
                        "num_returned": len(reranked),
                        "latency_ms": elapsed_ms,
                        "cache_hits": self._cache_hits,
                        "cache_misses": self._cache_misses
                    })
                except Exception:
                    # Ignore errors in metrics logging
                    pass
            
            return reranked
        
        except Exception as e:
            _logger.error(f"Reranking failed: {e}", exc_info=True)
            # Fallback: return original documents, cut to top_k if needed
            return docs[:top_k] if top_k else docs
    
    async def rerank_async(self, query: str, docs: List[Dict[str, Any]], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Rerank documents asynchronously, running sync method in executor with timeout"""
        if not docs:
            return []
        
        try:
            loop = asyncio.get_running_loop()
            
            # Run the synchronous rerank method in the thread pool
            future = loop.run_in_executor(
                self._executor,
                lambda: self.rerank(query, docs, top_k)
            )
            
            # Apply timeout if configured
            if self.cfg.timeout_s:
                return await asyncio.wait_for(future, timeout=self.cfg.timeout_s)
            else:
                return await future
        
        except asyncio.TimeoutError:
            _logger.warning(f"Rerank timeout after {self.cfg.timeout_s}s. Returning original docs.")
            # Fallback: return original documents, cut to top_k if needed
            return docs[:top_k] if top_k else docs
        except Exception as e:
            _logger.error(f"Async rerank failed: {e}", exc_info=True)
            return docs[:top_k] if top_k else docs
    
    def clear_cache(self) -> int:
        """Clear cache and reset hit/miss counters"""
        if self._cache:
            count = self._cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            return count
        return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics (size, hits, miss rate)"""
        if not self._cache:
            return {"enabled": False}
        
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        
        return {
            "enabled": True,
            "size": self._cache.size(),
            "capacity": self.cfg.cache_size,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": round(hit_rate, 3)
        }
    
    def close(self) -> None:
        """Clean up resources: shutdown thread pool and clear CUDA cache"""
        if self._executor:
            self._executor.shutdown(wait=True)
        
        if self._device == "cuda" and torch:
            try:
                # Attempt to free up GPU memory
                torch.cuda.empty_cache()
            except Exception:
                pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ==================== FACTORY ====================

def create_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", **kwargs) -> CrossEncoder:
    """Convenience function to create a cross-encoder instance from common parameters"""
    cfg = CrossEncoderConfig(model_name=model_name, **kwargs)
    return CrossEncoder(cfg)