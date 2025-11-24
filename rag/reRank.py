"""
reranker.py - Multi-stage Reranker for RAG
Améliorations:
- Pipeline de reranking en 3 étapes (vector, meta-fusion, cross-encoder)
- Cache LRU optimisé avec statistiques
- Normalisation de scores robuste
- Support de multiples stratégies de fusion
- Batch processing optimisé
- Métriques détaillées
- Gestion d'erreurs complète
"""
from __future__ import annotations
import asyncio
import time
import logging
import threading
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from enum import Enum

_logger = logging.getLogger("reranker")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False
    _logger.warning("NumPy not available, using fallback computations")


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class ScoreFusionStrategy(Enum):
    """Stratégies de fusion des scores"""
    WEIGHTED_AVERAGE = "weighted_avg"  # Moyenne pondérée
    MAX = "max"  # Score maximum
    MIN = "min"  # Score minimum
    HARMONIC_MEAN = "harmonic"  # Moyenne harmonique
    GEOMETRIC_MEAN = "geometric"  # Moyenne géométrique


@dataclass
class RerankerConfig:
    """Configuration pour le reranker"""
    # Candidates
    top_k: int = 50  # Nombre initial de candidats
    rerank_top_n: int = 20  # Nombre final après reranking
    
    # Cross-encoder
    cross_encoder_batch: int = 16
    use_cross_encoder: bool = False
    cross_encoder_top_k: int = 30  # Candidats à envoyer au cross-encoder
    
    # Scoring
    normalize_vectors: bool = True  # Vecteurs normalisés (pour IP)
    score_alpha: float = 0.7  # Poids: alpha*vector + (1-alpha)*meta
    fusion_strategy: ScoreFusionStrategy = ScoreFusionStrategy.WEIGHTED_AVERAGE
    min_score_threshold: float = 0.0  # Score minimum pour être retenu
    
    # Performance
    max_workers: int = 8
    cache_enabled: bool = True
    cache_size: int = 10000
    batch_encode_size: int = 32
    
    # Monitoring
    metrics_hook: Optional[Callable[[Dict[str, Any]], None]] = None
    log_operations: bool = True


# ==============================================================================
# LRU CACHE
# ==============================================================================

class _RerankerCache:
    """Cache LRU thread-safe pour vecteurs"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = max(1, capacity)
        self.lock = threading.RLock()
        self.cache: OrderedDict[str, List[float]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[List[float]]:
        """Récupère un vecteur du cache"""
        with self.lock:
            if key in self.cache:
                self.hits += 1
                # Move to end (most recent)
                self.cache.move_to_end(key)
                return self.cache[key]
            self.misses += 1
            return None

    def put(self, key: str, value: List[float]) -> None:
        """Ajoute un vecteur au cache"""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.cache[key] = value
            else:
                if len(self.cache) >= self.capacity:
                    # Remove oldest
                    self.cache.popitem(last=False)
                self.cache[key] = value

    def clear(self) -> int:
        """Vide le cache"""
        with self.lock:
            count = len(self.cache)
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            return count

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques"""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0.0
            return {
                "size": len(self.cache),
                "capacity": self.capacity,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(hit_rate, 3)
            }


# ==============================================================================
# RERANKER
# ==============================================================================

class Reranker:
    """
    Reranker multi-étapes pour RAG.
    
    Pipeline:
    1. Vector scoring (dot-product/cosine)
    2. Meta-score fusion (retriever + vector)
    3. Cross-encoder reranking (optionnel)
    
    Features:
    - Cache LRU pour vecteurs
    - Multiple stratégies de fusion
    - Support async complet
    - Batch processing
    - Métriques détaillées
    """

    def __init__(
        self,
        cfg: RerankerConfig,
        embedder: Any,
        cross_encoder: Optional[Any] = None,
    ):
        self.cfg = cfg
        self.embedder = embedder
        self.cross_encoder = cross_encoder
        
        # Threading
        self._executor = ThreadPoolExecutor(max_workers=max(1, cfg.max_workers))
        self._lock = threading.RLock()
        
        # Cache
        self._cache = _RerankerCache(cfg.cache_size) if cfg.cache_enabled else None
        
        # Métriques
        self._metrics = {
            "reranks": 0,
            "candidates_in": 0,
            "candidates_out": 0,
            "cross_encoder_calls": 0,
            "errors": 0,
            "total_time_s": 0.0
        }

    # ==================== PUBLIC API ====================

    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Reranking async des candidats.
        
        Args:
            query: Requête de recherche
            candidates: Liste de candidats à reranker
            top_n: Nombre de résultats finaux (défaut: config)
        
        Returns:
            Liste ordonnée de candidats (score desc)
        """
        if not candidates:
            return []
        
        top_n = top_n or self.cfg.rerank_top_n
        start_time = time.time()
        
        try:
            self._metrics["reranks"] += 1
            self._metrics["candidates_in"] += len(candidates)
            
            # 1. Encode query
            q_vec = await self._encode_query(query)
            
            # 2. Ensure candidate vectors (avec cache)
            candidates = await self._ensure_candidate_vectors(candidates, q_vec)
            
            # 3. Vector scoring
            scored = self._score_by_vector(q_vec, candidates)
            
            # 4. Fusion avec meta-scores
            fused = self._fuse_scores(scored)
            
            # 5. Filter by threshold
            fused = [c for c in fused if c.get("score", 0.0) >= self.cfg.min_score_threshold]
            
            # 6. Sort et trim pour cross-encoder
            fused.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            top_for_ce = fused[:self.cfg.cross_encoder_top_k]
            
            # 7. Optional cross-encoder reranking
            if self.cfg.use_cross_encoder and self.cross_encoder and top_for_ce:
                try:
                    top_for_ce = await self._cross_rerank(query, top_for_ce)
                    self._metrics["cross_encoder_calls"] += 1
                except Exception as e:
                    _logger.error(f"Cross-encoder failed, using vector scores: {e}")
            
            # 8. Final sort et trim
            result = sorted(
                top_for_ce,
                key=lambda x: x.get("score", 0.0),
                reverse=True
            )[:top_n]
            
            # Métriques
            elapsed = time.time() - start_time
            self._metrics["candidates_out"] += len(result)
            self._metrics["total_time_s"] += elapsed
            
            if self.cfg.log_operations:
                _logger.info(
                    f"Rerank: {len(candidates)} → {len(result)} candidates in {elapsed:.3f}s"
                )
            
            if self.cfg.metrics_hook:
                try:
                    self.cfg.metrics_hook({
                        "event": "rerank",
                        "query_len": len(query),
                        "candidates_in": len(candidates),
                        "candidates_out": len(result),
                        "duration_s": elapsed
                    })
                except Exception:
                    pass
            
            return result
            
        except Exception as e:
            self._metrics["errors"] += 1
            _logger.error(f"Rerank failed: {e}", exc_info=True)
            # Fallback: retourner les candidats originaux triés
            return sorted(
                candidates,
                key=lambda x: x.get("score", 0.0),
                reverse=True
            )[:top_n]

    def rerank_sync(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Version synchrone du reranking"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.rerank(query, candidates, top_n))

    # ==================== ENCODING ====================

    async def _encode_query(self, query: str) -> List[float]:
        """Encode la query en vecteur"""
        try:
            # Try async encoding
            if hasattr(self.embedder, "encode_async"):
                if asyncio.iscoroutinefunction(self.embedder.encode_async):
                    vecs = await self.embedder.encode_async([query])
                    return vecs[0] if vecs else []
            
            # Fallback to sync in executor
            loop = asyncio.get_running_loop()
            vecs = await loop.run_in_executor(
                self._executor,
                self.embedder.encode,
                [query]
            )
            return vecs[0] if vecs else []
            
        except Exception as e:
            _logger.error(f"Query encoding failed: {e}")
            # Fallback: zero vector
            dim = getattr(self.embedder, "dim", 384)
            if callable(dim):
                dim = dim()
            return [0.0] * dim

    async def _ensure_candidate_vectors(
        self,
        candidates: List[Dict[str, Any]],
        q_vec: Sequence[float]
    ) -> List[Dict[str, Any]]:
        """S'assure que tous les candidats ont un vecteur"""
        missing_indices = []
        texts_to_encode = []
        
        for i, cand in enumerate(candidates):
            # Check si vecteur déjà présent
            vec = cand.get("vector") or cand.get("vec") or cand.get("embedding")
            if vec and len(vec) == len(q_vec):
                cand["vector"] = vec
                continue
            
            # Check cache
            cache_key = self._make_cache_key(cand)
            if self._cache:
                cached_vec = self._cache.get(cache_key)
                if cached_vec and len(cached_vec) == len(q_vec):
                    cand["vector"] = cached_vec
                    continue
            
            # Besoin d'encoder
            missing_indices.append(i)
            texts_to_encode.append(cand.get("text", ""))
        
        # Encode missing vectors
        if missing_indices:
            try:
                encoded = await self._encode_texts(texts_to_encode)
                
                for j, idx in enumerate(missing_indices):
                    if j < len(encoded):
                        vec = encoded[j]
                        candidates[idx]["vector"] = vec
                        
                        # Cache le vecteur
                        if self._cache:
                            cache_key = self._make_cache_key(candidates[idx])
                            self._cache.put(cache_key, vec)
                            
            except Exception as e:
                _logger.error(f"Candidate encoding failed: {e}")
                # Fallback: zero vectors
                zero_vec = [0.0] * len(q_vec)
                for idx in missing_indices:
                    candidates[idx]["vector"] = zero_vec
        
        return candidates

    async def _encode_texts(self, texts: List[str]) -> List[List[float]]:
        """Encode une liste de textes"""
        if not texts:
            return []
        
        try:
            # Try async
            if hasattr(self.embedder, "encode_async"):
                if asyncio.iscoroutinefunction(self.embedder.encode_async):
                    return await self.embedder.encode_async(texts)
            
            # Fallback sync in executor
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor,
                self.embedder.encode,
                texts
            )
        except Exception as e:
            _logger.error(f"Text encoding failed: {e}")
            return []

    # ==================== VECTOR SCORING ====================

    def _score_by_vector(
        self,
        q_vec: Sequence[float],
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Score les candidats par similarité vectorielle"""
        for cand in candidates:
            vec = cand.get("vector", [])
            
            if not vec or len(vec) != len(q_vec):
                cand["_vector_score"] = 0.0
                continue
            
            try:
                if self.cfg.normalize_vectors:
                    # Dot product (pour vecteurs normalisés)
                    score = self._dot_product(q_vec, vec)
                else:
                    # Cosine similarity
                    score = self._cosine_similarity(q_vec, vec)
                
                cand["_vector_score"] = float(score)
            except Exception as e:
                _logger.debug(f"Vector scoring failed: {e}")
                cand["_vector_score"] = 0.0
        
        return candidates

    def _dot_product(self, a: Sequence[float], b: Sequence[float]) -> float:
        """Calcule le dot product"""
        if NUMPY_AVAILABLE:
            return float(np.dot(a, b))
        return sum(x * y for x, y in zip(a, b))

    def _cosine_similarity(self, a: Sequence[float], b: Sequence[float]) -> float:
        """Calcule la similarité cosine"""
        if NUMPY_AVAILABLE:
            a_arr = np.array(a, dtype=float)
            b_arr = np.array(b, dtype=float)
            norm_a = np.linalg.norm(a_arr)
            norm_b = np.linalg.norm(b_arr)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))
        
        # Fallback sans numpy
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot / (norm_a * norm_b)

    # ==================== SCORE FUSION ====================

    def _fuse_scores(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fusionne les scores vectoriels et meta-scores"""
        # Extraction des scores
        vector_scores = [c.get("_vector_score", 0.0) for c in candidates]
        meta_scores = [c.get("score", 0.0) for c in candidates]
        
        # Normalisation des meta-scores
        norm_meta = self._normalize_scores(meta_scores)
        
        # Fusion selon stratégie
        strategy = self.cfg.fusion_strategy
        
        for i, cand in enumerate(candidates):
            vec_score = vector_scores[i]
            meta_score = norm_meta[i]
            
            if strategy == ScoreFusionStrategy.WEIGHTED_AVERAGE:
                final_score = (
                    self.cfg.score_alpha * vec_score +
                    (1.0 - self.cfg.score_alpha) * meta_score
                )
            elif strategy == ScoreFusionStrategy.MAX:
                final_score = max(vec_score, meta_score)
            elif strategy == ScoreFusionStrategy.MIN:
                final_score = min(vec_score, meta_score)
            elif strategy == ScoreFusionStrategy.HARMONIC_MEAN:
                if vec_score > 0 and meta_score > 0:
                    final_score = 2 * (vec_score * meta_score) / (vec_score + meta_score)
                else:
                    final_score = 0.0
            elif strategy == ScoreFusionStrategy.GEOMETRIC_MEAN:
                final_score = (vec_score * meta_score) ** 0.5
            else:
                final_score = vec_score
            
            cand["score"] = float(final_score)
            cand["_meta_score"] = meta_score
        
        return candidates

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalise les scores dans [0, 1]"""
        if not scores:
            return []
        
        max_score = max(scores)
        min_score = min(scores)
        score_range = max_score - min_score
        
        if score_range == 0:
            return [0.5] * len(scores)  # Tous identiques
        
        return [(s - min_score) / score_range for s in scores]

    # ==================== CROSS-ENCODER ====================

    async def _cross_rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Reranking avec cross-encoder"""
        if not self.cross_encoder or not candidates:
            return candidates
        
        try:
            # Check si méthode async
            rerank_method = getattr(self.cross_encoder, "rerank", None)
            if rerank_method is None:
                rerank_method = getattr(self.cross_encoder, "rerank_async", None)
            
            if rerank_method is None:
                _logger.warning("Cross-encoder has no rerank method")
                return candidates
            
            if asyncio.iscoroutinefunction(rerank_method):
                # Async
                ranked = await rerank_method(query, candidates, len(candidates))
            else:
                # Sync dans executor
                loop = asyncio.get_running_loop()
                ranked = await loop.run_in_executor(
                    self._executor,
                    rerank_method,
                    query,
                    candidates,
                    len(candidates)
                )
            
            return ranked if ranked else candidates
            
        except Exception as e:
            _logger.error(f"Cross-encoder rerank failed: {e}", exc_info=True)
            return candidates

    # ==================== UTILITIES ====================

    def _make_cache_key(self, candidate: Dict[str, Any]) -> str:
        """Génère une clé de cache pour un candidat"""
        # Préfère l'ID si disponible et unique
        doc_id = candidate.get("id") or candidate.get("doc_id")
        if doc_id:
            return f"id:{doc_id}"
        
        # Sinon hash du texte
        text = candidate.get("text", "")
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        return f"text:{text_hash}"

    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques"""
        metrics = self._metrics.copy()
        
        # Ajout des stats de cache
        if self._cache:
            metrics["cache"] = self._cache.get_stats()
        
        # Calculs dérivés
        if metrics["reranks"] > 0:
            metrics["avg_candidates_in"] = round(
                metrics["candidates_in"] / metrics["reranks"], 2
            )
            metrics["avg_candidates_out"] = round(
                metrics["candidates_out"] / metrics["reranks"], 2
            )
            metrics["avg_time_s"] = round(
                metrics["total_time_s"] / metrics["reranks"], 3
            )
        
        return metrics

    def reset_metrics(self) -> None:
        """Reset les métriques"""
        self._metrics = {
            "reranks": 0,
            "candidates_in": 0,
            "candidates_out": 0,
            "cross_encoder_calls": 0,
            "errors": 0,
            "total_time_s": 0.0
        }

    def clear_cache(self) -> int:
        """Vide le cache"""
        if self._cache:
            return self._cache.clear()
        return 0

    def close(self) -> None:
        """Ferme les ressources"""
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ==============================================================================
# CROSS-ENCODER PLUGIN INTERFACE
# ==============================================================================

class CrossEncoderPlugin:
    """
    Interface pour plugins cross-encoder.
    
    Les implémentations doivent fournir:
    - rerank(query, candidates) -> List[Dict] (sync ou async)
    """

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank synchrone.
        Retourne candidats avec score mis à jour.
        """
        raise NotImplementedError

    async def rerank_async(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank asynchrone (optionnel).
        """
        raise NotImplementedError


# ==============================================================================
# FACTORY
# ==============================================================================

def create_reranker(
    embedder: Any,
    cross_encoder: Optional[Any] = None,
    **kwargs
) -> Reranker:
    """Factory pour créer un reranker"""
    cfg = RerankerConfig(**kwargs)
    return Reranker(cfg, embedder, cross_encoder)

'i'