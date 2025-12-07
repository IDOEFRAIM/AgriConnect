# rag/reranker.py
from __future__ import annotations

import asyncio
import time
import hashlib
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable, Sequence

_logger = logging.getLogger("rag.reranker")
_logger.addHandler(logging.NullHandler())

# --- Types ---
Doc = Dict[str, Any]
RankedDoc = Dict[str, Any]

# --- Config ---
@dataclass
class RerankConfig:
    cross_weight: float = 0.6      # weight for cross-encoder score
    vector_weight: float = 0.4     # weight for vector/dot score
    score_normalize: bool = True   # normalize scores before combining
    top_n: int = 20                # default number to return
    batch_size: int = 32
    timeout_s: float = 2.0
    cache_size: int = 10000

# --- Embedding utilities ---
def l2_normalize(v: Sequence[float]) -> List[float]:
    """Return L2-normalized vector (list)."""
    if not v:
        return []
    s = sum(float(x) ** 2 for x in v)
    norm = math.sqrt(s) if s > 0 else 0.0
    if norm == 0.0:
        return [0.0 for _ in v]
    return [float(x) / norm for x in v]

def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two sequences (assumes same length)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(float(x) * float(y) for x, y in zip(a, b))

# --- Reranker class ---
class Reranker:
    """
    Combines vector scores (from retriever) with cross-encoder reranking.
    
    Expected doc format from retriever:
    {
        "id": str,
        "text": str,
        "score": float,  # retrieval score
        "embedding": List[float],  # optional
        "meta": dict
    }
    """

    def __init__(self, cross_encoder: Any, cfg: Optional[RerankConfig] = None):
        self.cross_encoder = cross_encoder
        self.cfg = cfg or RerankConfig()
        self._lock = asyncio.Lock()

    def _ensure_vector_scores(
        self, 
        candidates: List[Doc], 
        query_emb: Optional[Sequence[float]] = None
    ) -> None:
        """
        Ensure each candidate has a vector_score.
        If query_emb provided, compute from embeddings; otherwise use retrieval score.
        """
        if query_emb is not None:
            qn = l2_normalize(query_emb)
            for c in candidates:
                emb = c.get("embedding") or c.get("emb") or c.get("vector")
                if emb and isinstance(emb, (list, tuple)) and len(emb) == len(qn):
                    cn = l2_normalize(emb)
                    c["vector_score"] = cosine_similarity(qn, cn)
                else:
                    # Fallback to retrieval score
                    c["vector_score"] = float(c.get("score", 0.0))
        else:
            # Use retrieval score as vector score
            for c in candidates:
                if "vector_score" not in c:
                    c["vector_score"] = float(c.get("score", 0.0))

    def _normalize_scores(self, docs: List[Doc], key: str) -> None:
        """Normalize scores to [0,1] using min-max"""
        vals = [float(d.get(key, 0.0)) for d in docs]
        if not vals:
            return
        
        mn = min(vals)
        mx = max(vals)
        
        if mx - mn <= 1e-12:
            # All same value
            for d in docs:
                d[key] = 1.0 if mx > 0 else 0.0
            return
        
        # Normalize
        for d in docs:
            d[key] = (float(d.get(key, 0.0)) - mn) / (mx - mn)

    def rerank(
        self, 
        query: str, 
        candidates: List[Doc], 
        top_n: Optional[int] = None, 
        query_emb: Optional[Sequence[float]] = None
    ) -> List[RankedDoc]:
        """
        Synchronous rerank.
        
        Args:
            query: query string
            candidates: list of docs from retriever
            top_n: number of results to return
            query_emb: optional query embedding for better vector scoring
            
        Returns:
            Ranked list of docs with combined_score, cross_score, vector_score
        """
        top_n = top_n or self.cfg.top_n
        start = time.perf_counter()

        if not candidates:
            return []

        # Ensure vector scores
        self._ensure_vector_scores(candidates, query_emb=query_emb)

        # Normalize vector scores if enabled
        if self.cfg.score_normalize:
            self._normalize_scores(candidates, key="vector_score")

        # Compute cross-encoder scores
        pairs = [(query, c.get("text", "")) for c in candidates]
        
        try:
            cross_scores = self.cross_encoder.predict_batch(pairs)
        except Exception:
            _logger.exception("cross-encoder predict_batch failed, using zeros")
            cross_scores = [0.0] * len(pairs)

        for c, s in zip(candidates, cross_scores):
            c["cross_score"] = float(s)

        # Normalize cross scores if enabled
        if self.cfg.score_normalize:
            self._normalize_scores(candidates, key="cross_score")

        # Combine scores
        for c in candidates:
            c["combined_score"] = (
                self.cfg.cross_weight * c.get("cross_score", 0.0)
                + self.cfg.vector_weight * c.get("vector_score", 0.0)
            )

        # Sort by combined score
        ranked = sorted(
            candidates, 
            key=lambda d: d.get("combined_score", 0.0), 
            reverse=True
        )
        
        dt = time.perf_counter() - start
        _logger.info(
            "rerank (sync) completed in %.3fs, returned %d/%d items",
            dt, min(top_n, len(ranked)), len(candidates)
        )
        
        # Log top results for debugging
        for i, doc in enumerate(ranked[:3]):
            _logger.debug(
                " RANK %d: id=%s combined=%.3f vector=%.3f cross=%.3f text_len=%d",
                i,
                doc.get("id"),
                doc.get("combined_score", 0.0),
                doc.get("vector_score", 0.0),
                doc.get("cross_score", 0.0),
                len(doc.get("text", ""))
            )
            if doc.get("text"):
                _logger.debug("  preview: %s", doc["text"][:100])
        
        return ranked[:top_n]

    async def rerank_async(
        self, 
        query: str, 
        candidates: List[Doc], 
        top_n: Optional[int] = None, 
        timeout: Optional[float] = None, 
        query_emb: Optional[Sequence[float]] = None
    ) -> List[RankedDoc]:
        """
        Asynchronous rerank.
        
        Args:
            query: query string
            candidates: list of docs from retriever
            top_n: number of results to return
            timeout: timeout in seconds
            query_emb: optional query embedding
            
        Returns:
            Ranked list of docs
        """
        top_n = top_n or self.cfg.top_n
        timeout = timeout or self.cfg.timeout_s
        start = time.perf_counter()

        if not candidates:
            return []

        # Ensure vector scores
        self._ensure_vector_scores(candidates, query_emb=query_emb)

        if self.cfg.score_normalize:
            self._normalize_scores(candidates, key="vector_score")

        # Compute cross-encoder scores
        cross_scores: List[float] = []
        
        try:
            # Try async rerank if available
            if hasattr(self.cross_encoder, "rerank_async"):
                docs_copy = [
                    {"id": c.get("id"), "text": c.get("text", "")} 
                    for c in candidates
                ]
                ranked = await asyncio.wait_for(
                    self.cross_encoder.rerank_async(
                        query, docs_copy, 
                        top_n=len(docs_copy), 
                        timeout=timeout
                    ),
                    timeout=timeout
                )
                # Extract scores by id
                id_to_score = {
                    d.get("id"): d.get("cross_score", 0.0) 
                    for d in ranked
                }
                cross_scores = [
                    float(id_to_score.get(c.get("id"), 0.0)) 
                    for c in candidates
                ]
            else:
                # Fallback to sync predict_batch in executor
                loop = asyncio.get_event_loop()
                pairs = [(query, c.get("text", "")) for c in candidates]
                cross_scores = await asyncio.wait_for(
                    loop.run_in_executor(
                        None, 
                        self.cross_encoder.predict_batch, 
                        pairs
                    ),
                    timeout=timeout
                )
        except asyncio.TimeoutError:
            _logger.warning("Cross-encoder timed out, using zeros")
            cross_scores = [0.0] * len(candidates)
        except Exception:
            _logger.exception("Cross-encoder failed, using zeros")
            cross_scores = [0.0] * len(candidates)

        for c, s in zip(candidates, cross_scores):
            c["cross_score"] = float(s)

        if self.cfg.score_normalize:
            self._normalize_scores(candidates, key="cross_score")

        # Combine scores
        for c in candidates:
            c["combined_score"] = (
                self.cfg.cross_weight * c.get("cross_score", 0.0)
                + self.cfg.vector_weight * c.get("vector_score", 0.0)
            )

        # Sort by combined score
        ranked = sorted(
            candidates, 
            key=lambda d: d.get("combined_score", 0.0), 
            reverse=True
        )
        
        dt = time.perf_counter() - start
        _logger.info(
            "rerank_async completed in %.3fs, returned %d/%d items",
            dt, min(top_n, len(ranked)), len(candidates)
        )
        
        # Debug log
        for i, doc in enumerate(ranked[:3]):
            _logger.debug(
                " RANK %d: id=%s combined=%.3f vector=%.3f cross=%.3f text_len=%d",
                i,
                doc.get("id"),
                doc.get("combined_score", 0.0),
                doc.get("vector_score", 0.0),
                doc.get("cross_score", 0.0),
                len(doc.get("text", ""))
            )
            if doc.get("text"):
                _logger.debug("  preview: %s", doc["text"][:100])
        
        return ranked[:top_n]


# --- Utility: diversity reranking ---
def rerank_with_diversity(
    ranked: List[RankedDoc], 
    diversity_lambda: float = 0.7, 
    max_items: int = 10, 
    cluster_key: Optional[str] = None
) -> List[RankedDoc]:
    """
    Simple greedy re-ranking to promote diversity.
    
    Args:
        ranked: already ranked documents
        diversity_lambda: 0..1, higher favors score over diversity
        max_items: max results to return
        cluster_key: metadata key for clustering (e.g., 'source')
        
    Returns:
        Diversified list
    """
    if not ranked:
        return []
    
    selected: List[RankedDoc] = []
    remaining = ranked.copy()
    
    while remaining and len(selected) < max_items:
        best_idx = 0
        best_score = -1e9
        
        for i, doc in enumerate(remaining):
            base = doc.get("combined_score", doc.get("score", 0.0))
            penalty = 0.0
            
            if cluster_key:
                cluster = (
                    doc.get("meta", {}).get(cluster_key) 
                    or doc.get(cluster_key)
                )
                if cluster:
                    same = sum(
                        1 for s in selected 
                        if (s.get("meta", {}).get(cluster_key) 
                            or s.get(cluster_key)) == cluster
                    )
                    penalty = same
            
            # Higher diversity_lambda favors base score
            adjusted = diversity_lambda * base - (1.0 - diversity_lambda) * penalty
            
            if adjusted > best_score:
                best_score = adjusted
                best_idx = i
        
        selected.append(remaining.pop(best_idx))
    
    return selected


# --- Adapter for pipeline ---
class RerankAdapter:
    """Convenience adapter for using reranker in pipeline"""
    
    def __init__(
        self, 
        reranker: Reranker, 
        diversity: Optional[Dict[str, Any]] = None
    ):
        self.reranker = reranker
        self.diversity = diversity or {}

    async def run(
        self, 
        query: str, 
        candidates: List[Doc], 
        top_k: int = 10, 
        query_emb: Optional[Sequence[float]] = None
    ) -> List[Doc]:
        """
        Run reranking pipeline with optional diversity.
        
        Args:
            query: query string
            candidates: docs from retriever
            top_k: number to return
            query_emb: optional query embedding
            
        Returns:
            Reranked and optionally diversified docs
        """
        # Step 1: Rerank
        ranked = await self.reranker.rerank_async(
            query, 
            candidates, 
            top_n=max(len(candidates), top_k), 
            query_emb=query_emb
        )
        
        # Step 2: Optional diversity
        if self.diversity:
            ranked = rerank_with_diversity(
                ranked, 
                diversity_lambda=self.diversity.get("lambda", 0.7), 
                max_items=top_k, 
                cluster_key=self.diversity.get("cluster_key")
            )
        else:
            ranked = ranked[:top_k]
        
        return ranked