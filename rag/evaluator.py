# evaluator.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import json
import csv
import time
import threading
import hashlib
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False


# ==================== CONFIG ====================

@dataclass
class EvaluatorConfig:
    """Configuration for RAG pipeline evaluation"""
    batch_size: int = 256
    semantic_threshold: float = 0.75
    save_intermediate: bool = False
    verbose: bool = False
    cache_embeddings: bool = True
    embedding_cache_size: int = 10000
    top_k_for_retrieval_metrics: int = 10
    max_workers: int = 8
    persist_cache_path: Optional[str] = None
    enable_perplexity: bool = False  # Requires additional model
    enable_bleu: bool = False  # Requires nltk


# ==================== UTILITIES ====================

def _safe_mean(xs: Sequence[float]) -> float:
    """Compute mean with zero fallback"""
    if not xs:
        return 0.0
    return float(sum(xs) / len(xs))


def _safe_percentile(xs: Sequence[float], p: float) -> float:
    """Compute percentile without numpy"""
    if not xs:
        return 0.0
    sorted_xs = sorted(xs)
    idx = int(len(sorted_xs) * p / 100.0)
    idx = min(idx, len(sorted_xs) - 1)
    return float(sorted_xs[idx])


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors"""
    try:
        import math
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        dot_product = sum(x * y for x, y in zip(a, b))
        return float(dot_product / (norm_a * norm_b))
    except Exception:
        return 0.0


def _jaccard_tokens(a: str, b: str) -> float:
    """Jaccard similarity at token level"""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a and not tokens_b:
        return 1.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return float(intersection / max(1, union))


def _rouge_l_simple(hypothesis: str, reference: str) -> float:
    """
    Simplified ROUGE-L (Longest Common Subsequence F1).
    No external dependencies.
    """
    hyp_tokens = hypothesis.split()
    ref_tokens = reference.split()
    n, m = len(hyp_tokens), len(ref_tokens)
    
    if n == 0 or m == 0:
        return 0.0
    
    # LCS via dynamic programming
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if hyp_tokens[i] == ref_tokens[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    
    lcs_length = dp[0][0]
    precision = lcs_length / n
    recall = lcs_length / m
    
    if precision + recall == 0:
        return 0.0
    
    beta = 1.2  # ROUGE-L typically uses beta=1.2
    f1 = ((1 + beta ** 2) * precision * recall) / (recall + beta ** 2 * precision)
    return float(f1)


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Levenshtein edit distance"""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


# ==================== EMBEDDING CACHE ====================

class _EmbeddingCache:
    """Thread-safe LRU cache for embeddings"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = max(0, capacity)
        self.lock = threading.RLock()
        self.store: Dict[str, List[float]] = {}
        self.order: deque = deque()

    def get(self, key: str) -> Optional[List[float]]:
        """Get cached embedding"""
        if self.capacity == 0:
            return None
        
        with self.lock:
            vec = self.store.get(key)
            if vec is None:
                return None
            
            # Move to end (most recently used)
            try:
                self.order.remove(key)
            except ValueError:
                pass
            self.order.append(key)
            
            return vec

    def put(self, key: str, vec: List[float]) -> None:
        """Cache embedding"""
        if self.capacity == 0:
            return
        
        with self.lock:
            if key in self.store:
                try:
                    self.order.remove(key)
                except ValueError:
                    pass
            
            self.store[key] = vec
            self.order.append(key)
            
            # Evict oldest if over capacity
            while len(self.order) > self.capacity:
                old_key = self.order.popleft()
                self.store.pop(old_key, None)

    def to_dict(self) -> Dict[str, List[float]]:
        """Export cache"""
        with self.lock:
            return dict(self.store)

    def load_dict(self, d: Dict[str, List[float]]) -> None:
        """Import cache"""
        with self.lock:
            self.store = dict(d)
            keys = list(d.keys())
            if self.capacity and len(keys) > self.capacity:
                keys = keys[-self.capacity:]
                self.store = {k: self.store[k] for k in keys}
            self.order = deque(keys)

    def clear(self) -> None:
        """Clear all cache"""
        with self.lock:
            self.store.clear()
            self.order.clear()

    def size(self) -> int:
        """Get cache size"""
        with self.lock:
            return len(self.store)


# ==================== EVALUATOR ====================

class Evaluator:
    """
    Modular evaluator for RAG pipelines.
    
    Features:
    - Lexical metrics: Jaccard, ROUGE-L, Levenshtein
    - Semantic metrics: cosine similarity (requires embedder)
    - Retrieval metrics: Recall@k, MRR, NDCG
    - Generation metrics: length, perplexity (optional)
    - Human evaluation ingestion
    - Batch and streaming evaluation
    - Embedding cache with persistence
    - Parallel processing
    """
    
    def __init__(
        self, 
        cfg: EvaluatorConfig, 
        embedder: Optional[Any] = None,
        tokenizer: Optional[Any] = None
    ):
        self.cfg = cfg
        self.embedder = embedder
        self.tokenizer = tokenizer
        
        # Metric registry
        self._metrics: Dict[str, Callable[[Dict[str, Any]], float]] = {}
        self._register_default_metrics()
        
        # Embedding cache
        self._emb_cache = (
            _EmbeddingCache(cfg.embedding_cache_size) 
            if cfg.cache_embeddings 
            else None
        )
        
        # Thread pool for parallel processing
        self._executor = ThreadPoolExecutor(max_workers=max(1, cfg.max_workers))
        
        # Load persisted cache if available
        if cfg.persist_cache_path and Path(cfg.persist_cache_path).exists():
            self._load_cache(cfg.persist_cache_path)

    # ==================== METRIC REGISTRATION ====================
    
    def register_metric(self, name: str, fn: Callable[[Dict[str, Any]], float]) -> None:
        """
        Register a custom metric function.
        
        Args:
            name: Metric name
            fn: Function taking item dict and returning float score
        """
        self._metrics[name] = fn

    def _register_default_metrics(self) -> None:
        """Register built-in metrics"""
        # Lexical metrics
        self.register_metric("length", self._metric_length)
        self.register_metric("jaccard_ref", self._metric_jaccard_ref)
        self.register_metric("rouge_l", self._metric_rouge_l)
        self.register_metric("edit_distance", self._metric_edit_distance)
        
        # Semantic metrics
        self.register_metric("semantic_sim", self._metric_semantic_sim)
        
        # Retrieval metrics
        self.register_metric("recall_at_k", self._metric_recall_at_k)
        self.register_metric("mrr", self._metric_mrr)
        self.register_metric("ndcg", self._metric_ndcg)
        self.register_metric("precision_at_k", self._metric_precision_at_k)

    # ==================== METRIC IMPLEMENTATIONS ====================
    
    def _metric_length(self, item: Dict[str, Any]) -> float:
        """Response length in tokens"""
        response = item.get("response") or ""
        return float(len(response.split()))

    def _metric_jaccard_ref(self, item: Dict[str, Any]) -> float:
        """Jaccard similarity with best reference"""
        refs = item.get("references") or []
        resp = item.get("response") or ""
        
        if not refs:
            return 0.0
        
        best_score = 0.0
        for ref in refs:
            score = _jaccard_tokens(resp, ref)
            best_score = max(best_score, score)
        
        return float(best_score)

    def _metric_rouge_l(self, item: Dict[str, Any]) -> float:
        """ROUGE-L F1 with best reference"""
        refs = item.get("references") or []
        resp = item.get("response") or ""
        
        if not refs:
            return 0.0
        
        best_score = 0.0
        for ref in refs:
            score = _rouge_l_simple(resp, ref)
            best_score = max(best_score, score)
        
        return float(best_score)

    def _metric_edit_distance(self, item: Dict[str, Any]) -> float:
        """Normalized Levenshtein distance (lower is better, so we return 1-normalized)"""
        refs = item.get("references") or []
        resp = item.get("response") or ""
        
        if not refs:
            return 0.0
        
        min_distance = float('inf')
        for ref in refs:
            distance = _levenshtein_distance(resp, ref)
            min_distance = min(min_distance, distance)
        
        # Normalize by max length
        max_len = max(len(resp), max((len(r) for r in refs), default=1))
        normalized = 1.0 - (min_distance / max(max_len, 1))
        return float(max(0.0, normalized))

    def _metric_semantic_sim(self, item: Dict[str, Any]) -> float:
        """Semantic similarity using embedder"""
        if not self.embedder:
            return 0.0
        
        resp = item.get("response") or ""
        refs = item.get("references") or []
        
        # Compare with references if available
        if refs:
            texts = [resp] + refs
            try:
                vecs = self._encode_cached(texts)
                query_vec = vecs[0]
                
                best_sim = 0.0
                for ref_vec in vecs[1:]:
                    sim = _cosine_similarity(query_vec, ref_vec)
                    best_sim = max(best_sim, sim)
                
                return float(best_sim)
            except Exception:
                return 0.0
        
        # Fallback: compare with retrieved documents
        docs = item.get("docs") or []
        if not docs:
            return 0.0
        
        doc_text = " ".join(d.get("text", "") for d in docs)
        if not doc_text:
            return 0.0
        
        try:
            vecs = self._encode_cached([resp, doc_text])
            return float(_cosine_similarity(vecs[0], vecs[1]))
        except Exception:
            return 0.0

    def _metric_recall_at_k(self, item: Dict[str, Any]) -> float:
        """Recall@k for retrieval"""
        k = self.cfg.top_k_for_retrieval_metrics
        retrieved = (item.get("retrieved") or [])[:k]
        relevant = set(item.get("relevant_ids") or [])
        
        if not relevant:
            return 0.0
        
        hits = sum(1 for doc_id in retrieved if doc_id in relevant)
        return float(hits / len(relevant))

    def _metric_mrr(self, item: Dict[str, Any]) -> float:
        """Mean Reciprocal Rank"""
        retrieved = item.get("retrieved") or []
        relevant = set(item.get("relevant_ids") or [])
        
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                return float(1.0 / rank)
        
        return 0.0

    def _metric_ndcg(self, item: Dict[str, Any]) -> float:
        """Normalized Discounted Cumulative Gain"""
        k = self.cfg.top_k_for_retrieval_metrics
        retrieved = (item.get("retrieved") or [])[:k]
        relevant = set(item.get("relevant_ids") or [])
        
        if not relevant:
            return 0.0
        
        # DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                dcg += 1.0 / (1.0 + float(i - 1) / (2.0 ** 0.5))  # log2(i+1) approximation
        
        # IDCG (ideal)
        idcg = 0.0
        for i in range(1, min(len(relevant), k) + 1):
            idcg += 1.0 / (1.0 + float(i - 1) / (2.0 ** 0.5))
        
        return float(dcg / idcg) if idcg > 0 else 0.0

    def _metric_precision_at_k(self, item: Dict[str, Any]) -> float:
        """Precision@k for retrieval"""
        k = self.cfg.top_k_for_retrieval_metrics
        retrieved = (item.get("retrieved") or [])[:k]
        relevant = set(item.get("relevant_ids") or [])
        
        if not retrieved:
            return 0.0
        
        hits = sum(1 for doc_id in retrieved if doc_id in relevant)
        return float(hits / len(retrieved))

    # ==================== ENCODING WITH CACHE ====================
    
    def _encode_cached(self, texts: Sequence[str]) -> List[List[float]]:
        """Encode texts with caching"""
        if not self.embedder:
            raise RuntimeError("Embedder not configured")
        
        outputs: List[Optional[List[float]]] = [None] * len(texts)
        to_encode: List[str] = []
        to_encode_indices: List[int] = []
        
        # Check cache
        for i, text in enumerate(texts):
            cache_key = self._cache_key(text)
            cached_vec = self._emb_cache.get(cache_key) if self._emb_cache else None
            
            if cached_vec is not None:
                outputs[i] = cached_vec
            else:
                to_encode.append(text)
                to_encode_indices.append(i)
        
        # Encode uncached texts
        if to_encode:
            try:
                vecs = self.embedder.encode(list(to_encode), is_query=False)
                
                for j, vec in enumerate(vecs):
                    idx = to_encode_indices[j]
                    outputs[idx] = vec
                    
                    # Cache the embedding
                    if self._emb_cache:
                        cache_key = self._cache_key(to_encode[j])
                        self._emb_cache.put(cache_key, vec)
            except Exception as e:
                raise RuntimeError(f"Encoding failed: {e}")
        
        # Validate all outputs
        for i, vec in enumerate(outputs):
            if vec is None:
                raise RuntimeError(f"Failed to encode text {i}")
        
        return outputs  # type: ignore

    @staticmethod
    def _cache_key(text: str) -> str:
        """Generate cache key from text"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # ==================== EVALUATION CORE ====================
    
    def evaluate_batch(
        self, 
        items: List[Dict[str, Any]], 
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of items.
        
        Args:
            items: List of evaluation items with keys:
                - query: str
                - response: str
                - references: List[str] (optional)
                - docs: List[Dict] (optional, for retrieval)
                - retrieved: List[str] (optional, doc IDs)
                - relevant_ids: List[str] (optional, ground truth)
            metrics: Metric names to compute (default: all registered)
            
        Returns:
            Report dict with aggregated metrics and per-item results
        """
        metrics = metrics or list(self._metrics.keys())
        results: List[Dict[str, Any]] = []
        aggregated: Dict[str, List[float]] = defaultdict(list)
        
        start_time = time.time()
        
        # Process in batches
        for batch_start in range(0, len(items), self.cfg.batch_size):
            batch = items[batch_start:batch_start + self.cfg.batch_size]
            
            # Parallel metric computation
            futures = []
            for item in batch:
                future = self._executor.submit(self._compute_item_metrics, item, metrics)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    item_result = future.result()
                    results.append(item_result)
                    
                    # Aggregate metrics
                    for metric_name, score in item_result.get("metrics", {}).items():
                        aggregated[metric_name].append(score)
                except Exception as e:
                    if self.cfg.verbose:
                        print(f"Item evaluation failed: {e}")
        
        elapsed = time.time() - start_time
        
        # Build report
        report = {
            "count": len(items),
            "duration_s": elapsed,
            "metrics": {}
        }
        
        # Compute statistics
        for metric_name, values in aggregated.items():
            if NUMPY_AVAILABLE and np is not None:
                arr = np.array(values)
                report["metrics"][metric_name] = {
                    "mean": float(arr.mean()),
                    "median": float(np.median(arr)),
                    "std": float(arr.std()),
                    "p90": float(np.percentile(arr, 90)),
                    "p95": float(np.percentile(arr, 95)),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                }
            else:
                report["metrics"][metric_name] = {
                    "mean": _safe_mean(values),
                    "median": _safe_percentile(values, 50),
                    "p90": _safe_percentile(values, 90),
                    "p95": _safe_percentile(values, 95),
                    "min": min(values) if values else 0.0,
                    "max": max(values) if values else 0.0,
                }
        
        report["items"] = results
        
        return report

    def _compute_item_metrics(
        self, 
        item: Dict[str, Any], 
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Compute all metrics for a single item"""
        item_result = {
            "query": item.get("query"),
            "response": item.get("response"),
            "metrics": {}
        }
        
        for metric_name in metrics:
            try:
                metric_fn = self._metrics.get(metric_name)
                if metric_fn is None:
                    item_result["metrics"][metric_name] = 0.0
                    continue
                
                score = float(metric_fn(item))
                item_result["metrics"][metric_name] = score
            except Exception as e:
                if self.cfg.verbose:
                    print(f"Metric {metric_name} failed for item: {e}")
                item_result["metrics"][metric_name] = 0.0
        
        return item_result

    def evaluate_single(
        self, 
        item: Dict[str, Any], 
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate a single item"""
        return self.evaluate_batch([item], metrics)

    # ==================== STREAMING EVALUATION ====================
    
    def evaluate_stream(
        self,
        iterator: Iterable[Dict[str, Any]],
        callback: Callable[[Dict[str, Any]], None],
        metrics: Optional[List[str]] = None
    ) -> None:
        """
        Evaluate items from an iterator with streaming callback.
        
        Args:
            iterator: Iterator yielding evaluation items
            callback: Function called with each batch report
            metrics: Metric names to compute
        """
        batch = []
        
        for item in iterator:
            batch.append(item)
            
            if len(batch) >= self.cfg.batch_size:
                report = self.evaluate_batch(batch, metrics)
                try:
                    callback(report)
                except Exception as e:
                    if self.cfg.verbose:
                        print(f"Callback failed: {e}")
                batch = []
        
        # Process remaining items
        if batch:
            report = self.evaluate_batch(batch, metrics)
            try:
                callback(report)
            except Exception as e:
                if self.cfg.verbose:
                    print(f"Callback failed: {e}")

    # ==================== PERSISTENCE ====================
    
    def save_report(
        self, 
        report: Dict[str, Any], 
        path: str, 
        fmt: str = "json"
    ) -> None:
        """
        Save evaluation report to file.
        
        Args:
            report: Report dict from evaluate_batch
            path: Output file path
            fmt: Format ("json" or "csv")
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        if fmt == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        
        elif fmt == "csv":
            items = report.get("items", [])
            if not items:
                with open(path, "w", encoding="utf-8") as f:
                    f.write("")
                return
            
            metric_keys = list(items[0].get("metrics", {}).keys())
            headers = ["query", "response"] + metric_keys
            
            with open(path, "w", newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                
                for item in items:
                    row = [
                        item.get("query", ""),
                        item.get("response", "")
                    ] + [
                        item["metrics"].get(k, "") for k in metric_keys
                    ]
                    writer.writerow(row)
        
        else:
            raise ValueError(f"Unsupported format: {fmt}")

    def _load_cache(self, path: str) -> None:
        """Load persisted embedding cache"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                cache_dict = json.load(f)
            
            if self._emb_cache:
                self._emb_cache.load_dict(cache_dict)
                print(f"Loaded {len(cache_dict)} cached embeddings from {path}")
        except Exception as e:
            if self.cfg.verbose:
                print(f"Failed to load cache: {e}")

    def save_cache(self, path: Optional[str] = None) -> None:
        """Save embedding cache to disk"""
        save_path = path or self.cfg.persist_cache_path
        if not save_path or not self._emb_cache:
            return
        
        try:
            cache_dict = self._emb_cache.to_dict()
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(cache_dict, f, ensure_ascii=False, indent=2)
            
            if self.cfg.verbose:
                print(f"Saved {len(cache_dict)} embeddings to {save_path}")
        except Exception as e:
            if self.cfg.verbose:
                print(f"Failed to save cache: {e}")

    # ==================== HUMAN EVALUATION ====================
    
    def ingest_human_labels(
        self, 
        labeled_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Ingest human labels and compute classification metrics.
        
        Args:
            labeled_items: List with keys:
                - label: bool (ground truth)
                - predicted: bool (model prediction)
                
        Returns:
            Dict with precision, recall, F1, confusion matrix
        """
        tp = fp = tn = fn = 0
        
        for item in labeled_items:
            label = bool(item.get("label"))
            predicted = bool(item.get("predicted", True))
            
            if predicted and label:
                tp += 1
            elif predicted and not label:
                fp += 1
            elif not predicted and not label:
                tn += 1
            elif not predicted and label:
                fn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "confusion_matrix": {
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn
            }
        }

    # ==================== UTILITIES ====================
    
    def cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self._emb_cache:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "size": self._emb_cache.size(),
            "capacity": self.cfg.embedding_cache_size,
            "utilization": self._emb_cache.size() / max(1, self.cfg.embedding_cache_size)
        }

    def clear_cache(self) -> None:
        """Clear embedding cache"""
        if self._emb_cache:
            self._emb_cache.clear()

    def shutdown(self) -> None:
        """Shutdown evaluator and save cache"""
        try:
            # Save cache if configured
            if self._emb_cache and self.cfg.persist_cache_path:
                self.save_cache()
        finally:
            # Shutdown executor
            try:
                self._executor.shutdown(wait=False)
            except Exception:
                pass

    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.shutdown()
        except Exception:
            pass


        'i'