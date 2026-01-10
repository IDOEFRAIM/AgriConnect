# rag/augmentation_pipeline.py - CORRIGÉ pour nouveau format search()
from __future__ import annotations
import time 
import json
from pathlib import Path
import asyncio
import uuid
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from rag.utils.cache_utils import normalize_cache_key, compute_cache_stats
from rag.utils.metricsTracker import MetricsTracker

_logger = logging.getLogger("rag.augmentation_pipeline")
_logger.addHandler(logging.NullHandler())

# -------------------------
# Configuration
# -------------------------
@dataclass
class AugmentationConfig:
    # Retrieval
    top_k: int = 50
    rerank_top_n: int = 10
    candidate_pool: int = 200
    
    # Snippet extraction
    max_snippets_per_doc: int = 3
    snippet_max_tokens: int = 200
    snippet_overlap: int = 50
    min_snippet_score: float = 0.1
    
    # Scoring weights
    retrieval_weight: float = 0.3
    rerank_weight: float = 0.5
    semantic_weight: float = 0.2
    support_threshold: float = 0.75
    diversity_penalty: float = 0.1
    
    # Timeouts
    timeout_s: float = 8.0
    retrieval_timeout_s: float = 3.0
    rerank_timeout_s: float = 2.0
    encode_timeout_s: float = 2.0
    retry_attempts: int = 2
    retry_backoff_s: float = 0.5
    
    # Performance
    concurrency: int = 8
    batch_size: int = 32
    enable_caching: bool = True
    cache_ttl_s: float = 3600.0
    warm_cache: bool = True
    
    # Cross-encoder reranking
    use_cross_encoder: bool = True
    cross_encoder_batch: int = 16
    
    # Quality controls
    enable_deduplication: bool = True
    dedup_threshold: float = 0.85
    enable_quality_filter: bool = True
    min_text_length: int = 20
    
    # Monitoring
    enable_metrics: bool = True
    log_slow_queries: bool = True
    slow_query_threshold_s: float = 2.0

# -------------------------
# Response Models
# -------------------------
@dataclass
class SnippetInfo:
    text: str
    doc_id: str
    citation: str
    retrieval_score: float
    rerank_score: float
    has_embedding: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "doc_id": self.doc_id,
            "citation": self.citation,
            "retrieval_score": round(self.retrieval_score, 2),
            "rerank_score": round(self.rerank_score, 2),
            "has_embedding": self.has_embedding
        }

@dataclass
class ContextInfo:
    query: str
    snippets: List[SnippetInfo]
    total_tokens: int
    retrieval_time_s: float
    processing_time_s: float
    is_timeout: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "snippets": [s.to_dict() for s in self.snippets],
            "total_tokens": self.total_tokens,
            "retrieval_time_s": round(self.retrieval_time_s, 3),
            "processing_time_s": round(self.processing_time_s, 3),
            "is_timeout": self.is_timeout
        }

@dataclass
class DiagnosticsInfo:
    steps: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    cache: Dict[str, Any] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": {k: {kk: round(vv, 3) if isinstance(vv, float) else vv 
                         for kk, vv in v.items()} 
                     for k, v in self.steps.items()},
            "cache": self.cache,
            "timings": {k: round(v, 3) for k, v in self.timings.items()},
            "summary": self.summary
        }

@dataclass
class AugmentedResponse:
    request_id: str
    query: str
    context: ContextInfo
    diagnostics: DiagnosticsInfo
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "query": self.query,
            "context": self.context.to_dict(),
            "diagnostics": self.diagnostics.to_dict()
        }

# -------------------------
# Main Pipeline
# -------------------------
class AugmentationPipeline:
    """
    Pipeline complète pour l'augmentation de contexte RAG.
    
    CORRECTION: Gère le nouveau format de search() qui retourne:
    {
        "ok": bool,
        "results": List[List[Dict]],
        "num_queries": int,
        "total_hits": int,
        "time_s": float,
        "error": str (optionnel)
    }
    """
    
    def __init__(
        self,
        retriever: Any,
        encoder: Any,
        reranker: Optional[Any] = None,
        cfg: Optional[AugmentationConfig] = None
    ):
        self.retriever = retriever
        self.encoder = encoder
        self.reranker = reranker
        self.cfg = cfg or AugmentationConfig()
        
        # Métriques tracker
        self._metrics = MetricsTracker(max_history=1000)

        # Cache d'embeddings
        self._embedding_cache: Dict[str, List[float]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_file = Path("embedding_cache.json")
        
        # Semaphore for concurrent operations
        self._sem = asyncio.Semaphore(self.cfg.concurrency)

        # Charger cache au démarrage
        if self.cfg.enable_caching:
            self._load_cache()
    
    def _load_cache(self):
        """Charge le cache depuis le disque avec clés normalisées."""
        if not self._cache_file.exists():
            _logger.info("No cache found, starting fresh")
            return
        
        try:
            with open(self._cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                raw_embeddings = data.get('embeddings', {})
                self._embedding_cache = {
                    normalize_cache_key(k): v 
                    for k, v in raw_embeddings.items()
                }
                
                stats = data.get('stats', {})
                self._cache_hits = stats.get('total_hits', 0)
                self._cache_misses = stats.get('total_misses', 0)
                
                _logger.info(
                    "✓ Cache loaded: %d entries, %d hits",
                    len(self._embedding_cache), self._cache_hits
                )
        
        except Exception as e:
            _logger.warning("Failed to load cache: %s", e)
            self._embedding_cache = {}

    def _save_cache(self):
        """Sauvegarde atomique du cache."""
        if not self.cfg.enable_caching:
            return
        
        try:
            tmp_file = self._cache_file.with_suffix('.tmp')
            
            data = {
                'embeddings': self._embedding_cache,
                'stats': {
                    'total_hits': self._cache_hits,
                    'total_misses': self._cache_misses,
                    'cache_size': len(self._embedding_cache),
                    'last_updated': time.perf_counter()
                },
                'metadata': {
                    'version': '1.0',
                    'normalized_keys': True
                }
            }
            
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            tmp_file.replace(self._cache_file)
        
        except Exception as e:
            _logger.warning("Failed to save cache: %s", e)

    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques de performance."""
        base_metrics = self._metrics.get_metrics()
        cache_stats = compute_cache_stats(self._cache_hits, self._cache_misses)
        
        return {
            **base_metrics,
            'cache': {
                **cache_stats,
                'cache_size': len(self._embedding_cache)
            }
        }

    async def augment(
        self, 
        query: str,
        request_id: Optional[str] = None,
        cfg_override: Optional[AugmentationConfig] = None
    ) -> AugmentedResponse:
        """Point d'entrée principal : augmente le contexte pour une requête."""
        start_time = time.perf_counter()
        success = False
        cfg = cfg_override or self.cfg
        request_id = request_id or f"aug_{int(time.perf_counter() * 1000)}"
        
        diagnostics = DiagnosticsInfo()
        
        try:
            # Step 1: Retrieve documents
            retrieve_start = time.perf_counter()
            docs = await self._retrieve_documents(query, cfg, diagnostics)
            retrieve_time = time.perf_counter() - retrieve_start
            
            if not docs:
                return self._empty_response(
                    request_id, query, retrieve_time, 
                    time.perf_counter() - start_time, diagnostics, False
                )
            
            # Step 2: Rerank documents (optionnel)
            if self.reranker and cfg.use_cross_encoder:
                await self._rerank_documents(query, docs, cfg, diagnostics)
            
            # Step 3: Extract and encode snippets
            snippets = await self._extract_snippets(query, docs, cfg, diagnostics)
            
            if not snippets:
                return self._empty_response(
                    request_id, query, retrieve_time,
                    time.perf_counter() - start_time, diagnostics, False
                )
            
            # Step 4: Format final response
            format_start = time.perf_counter()
            context = self._build_context(
                query, snippets, retrieve_time, 
                time.perf_counter() - start_time
            )
            format_time = time.perf_counter() - format_start
            
            # Step 5: Complete diagnostics
            diagnostics.steps["format"] = {
                "count": len(snippets),
                "time_s": format_time
            }
            diagnostics.cache = {
                "cache_size": len(self._embedding_cache),
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses
            }
            diagnostics.timings = {
                "total_time_s": time.perf_counter() - start_time,
                "augment_time_s": time.perf_counter() - start_time - format_time,
                "format_time_s": format_time
            }
            diagnostics.summary = {
                "snippets_returned": len(snippets),
                "total_tokens": context.total_tokens,
                "is_timeout": False,
                "query": query
            }
            success = True
            return AugmentedResponse(
                request_id=request_id,
                query=query,
                context=context,
                diagnostics=diagnostics
            )
            
        except asyncio.TimeoutError:
            success = False
            _logger.warning("Augmentation timed out for query: %s", query)
            return self._empty_response(
                request_id, query, 0.0, 
                time.perf_counter() - start_time, diagnostics, True
            )
        except Exception as e:
            success = False
            _logger.exception("Augmentation failed for query: %s", query)
            return self._empty_response(
                request_id, query, 0.0,
                time.perf_counter() - start_time, diagnostics, False
            )
        finally:
            latency = time.perf_counter() - start_time
            self._metrics.record_query(latency, success=success)

    async def _retrieve_documents(
        self,
        query: str,
        cfg: AugmentationConfig,
        diagnostics: DiagnosticsInfo
    ) -> List[Dict[str, Any]]:
        """
        Step 1: Retrieve documents - CORRIGÉ pour nouveau format.
        
        Le retriever.retrieve() retourne maintenant une liste de dicts:
        [
            {
                "id": str,
                "score": float,
                "distance": float,
                "text": str,
                "meta": dict
            },
            ...
        ]
        """
        timeout = cfg.retrieval_timeout_s or cfg.timeout_s or 8.0

        try:
            _logger.info("Starting retrieval for query: %s", query[:50])
            retrieve_coro = self.retriever.retrieve(query=query, top_k=cfg.top_k)
            _logger.debug("retrieve coro: %s", retrieve_coro)
        except Exception as e:
            _logger.exception("Failed to create retrieve coroutine: %s", e)
            diagnostics.steps["retrieve"] = {
                "count": 0, 
                "time_s": 0.0, 
                "error": str(e)
            }
            return []

        task = asyncio.create_task(retrieve_coro) if asyncio.iscoroutine(retrieve_coro) else retrieve_coro

        start = time.perf_counter()
        try:
            print('starting retrievin docs')
            docs = await asyncio.wait_for(task, timeout=timeout)
            print('docs:',docs)
            elapsed = time.perf_counter() - start

            # Validation des résultats
            if docs is None:
                docs = []
            elif not isinstance(docs, list):
                try:
                    docs = list(docs)
                except Exception:
                    docs = []

            _logger.info("RETRIEVER HITS COUNT: %d", len(docs))
            
            # Debug logging avec vérification des champs
            for i, h in enumerate(docs[:5]):
                hit_id = h.get('id', 'NO_ID')
                score = h.get('score', 0.0)
                text = h.get('text', '') or ''
                meta = h.get('meta', {}) or {}
                
                _logger.debug(
                    "HIT %d: id=%s score=%.3f text_len=%d meta_keys=%s",
                    i, hit_id, score, len(text), list(meta.keys())
                )
                
                # Preview du texte
                if text:
                    preview = text[:200].replace("\n", " ")
                    _logger.debug("  preview: %s", preview)
                else:
                    _logger.warning("  ⚠️ NO TEXT in hit %d", i)

            diagnostics.steps["retrieve"] = {
                "count": len(docs),
                "time_s": round(elapsed, 6),
                "timeout": False,
            }

            return docs

        except asyncio.TimeoutError:
            elapsed = time.perf_counter() - start
            _logger.warning("Retrieval timed out after %.3fs", elapsed)

            try:
                task.cancel()
                await task
            except asyncio.CancelledError:
                _logger.debug("Retrieve task cancelled")
            except Exception:
                _logger.debug("Exception while cancelling task", exc_info=True)

            diagnostics.steps["retrieve"] = {
                "count": 0,
                "time_s": round(elapsed, 6),
                "timeout": True,
            }
            return []

        except Exception as e:
            elapsed = time.perf_counter() - start
            _logger.exception("Retrieval failed after %.3fs: %s", elapsed, e)
            diagnostics.steps["retrieve"] = {
                "count": 0,
                "time_s": round(elapsed, 6),
                "error": str(e),
            }
            
            try:
                if not task.done():
                    task.cancel()
                    await task
            except Exception:
                _logger.debug("Exception cancelling task after failure", exc_info=True)
            
            return []
    
    async def _rerank_documents(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        cfg: AugmentationConfig,
        diagnostics: DiagnosticsInfo
    ) -> None:
        """Step 2: Rerank documents using cross-encoder"""
        rerank_start = time.perf_counter()
        try:
            # Prepare docs for reranking
            rerank_docs = [{
                "id": d.get("id") or d.get("doc_id", "unknown"),
                "text": d.get("text", ""),
                "score": d.get("score", 0.0)
            } for d in docs]
            
            # Call reranker
            if hasattr(self.reranker, "rerank_async"):
                reranked = await asyncio.wait_for(
                    self.reranker.rerank_async(
                        query, rerank_docs, top_n=cfg.rerank_top_n
                    ),
                    timeout=cfg.rerank_timeout_s
                )
            else:
                reranked = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.reranker.rerank,
                        query, rerank_docs, cfg.rerank_top_n
                    ),
                    timeout=cfg.rerank_timeout_s
                )
            
            # Debug logging
            for i, r in enumerate(reranked[:5]):
                _logger.debug(
                    "RANK %d: id=%s combined=%.3f vector=%.3f cross=%.3f",
                    i,
                    r.get('id'),
                    r.get('combined_score', 0.0),
                    r.get('vector_score', 0.0),
                    r.get('cross_score', 0.0)
                )

            # Update original docs with rerank scores
            id_to_rerank = {
                d.get("id"): d.get("combined_score", d.get("cross_score", 0.0))
                for d in reranked
            }
            
            for doc in docs:
                doc_id = doc.get("id") or doc.get("doc_id")
                doc["rerank_score"] = id_to_rerank.get(doc_id, doc.get("score", 0.0))
            
            # Sort by rerank score
            docs.sort(key=lambda d: d.get("rerank_score", 0.0), reverse=True)
            
            diagnostics.steps["rerank"] = {
                "count": len(reranked),
                "time_s": time.perf_counter() - rerank_start
            }
            
        except asyncio.TimeoutError:
            _logger.warning("Reranking timed out, using retrieval scores")
            for doc in docs:
                doc["rerank_score"] = doc.get("score", 0.0)
            diagnostics.steps["rerank"] = {
                "count": 0,
                "time_s": cfg.rerank_timeout_s,
                "timeout": True
            }
        except Exception as e:
            _logger.exception("Reranking failed, using retrieval scores")
            for doc in docs:
                doc["rerank_score"] = doc.get("score", 0.0)
            diagnostics.steps["rerank"] = {
                "count": 0,
                "time_s": time.perf_counter() - rerank_start,
                "error": str(e)
            }
    
    async def _extract_snippets(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        cfg: AugmentationConfig,
        diagnostics: DiagnosticsInfo
    ) -> List[SnippetInfo]:
        """Step 3: Extract and encode snippets from documents"""
        encode_start = time.perf_counter()
        snippets: List[SnippetInfo] = []
        texts_to_encode: List[str] = []
        snippet_map: List[int] = []
        
        try:
            # Extract snippets from top documents
            for doc in docs[:cfg.rerank_top_n]:
                doc_snippets = self._extract_doc_snippets(doc, cfg)
                for snip in doc_snippets[:cfg.max_snippets_per_doc]:
                    snippets.append(snip)
                    
                    # Check cache
                    cache_key = normalize_cache_key(snip.text)
                    if cfg.enable_caching and cache_key in self._embedding_cache:
                        snip.has_embedding = True
                        self._cache_hits += 1
                    else:
                        snippet_map.append(len(snippets) - 1)
                        texts_to_encode.append(snip.text)
                        self._cache_misses += 1
            
            # Encode uncached snippets
            if texts_to_encode:
                embeddings = await self._encode_texts(texts_to_encode, "snippet")
                
                # Update snippets with embeddings
                for idx, emb in zip(snippet_map, embeddings):
                    if emb and len(emb) > 0:
                        snippets[idx].has_embedding = True
                        if cfg.enable_caching:
                            cache_key = normalize_cache_key(snippets[idx].text)
                            self._embedding_cache[cache_key] = emb
            
            # Filter by score
            snippets = [
                s for s in snippets 
                if s.rerank_score >= cfg.min_snippet_score
            ]
            
            # Sort by rerank score
            snippets.sort(key=lambda s: s.rerank_score, reverse=True)
            
            # Apply diversity filter if needed
            if cfg.diversity_penalty > 0 and len(snippets) > cfg.rerank_top_n:
                snippets = self._apply_diversity_filter(
                    snippets, cfg.rerank_top_n, cfg.diversity_penalty
                )
            
            diagnostics.steps["extract_snippets"] = {
                "count": len(snippets),
                "time_s": time.perf_counter() - encode_start
            }
            
            return snippets[:cfg.rerank_top_n]
            
        except Exception as e:
            _logger.exception("Snippet extraction failed")
            diagnostics.steps["encode"] = {
                "texts_encoded": 0,
                "time_s": time.perf_counter() - encode_start,
                "error": str(e)
            }
            return []
    
    def _extract_doc_snippets(
        self, 
        doc: Dict[str, Any],
        cfg: AugmentationConfig
    ) -> List[SnippetInfo]:
        """Extract snippets from a single document"""
        snippets = []
        
        # Get text from doc
        text = doc.get("text", "").strip()
        doc_id = doc.get("id") or doc.get("doc_id", "unknown")
        
        # Get citation from meta or default
        meta = doc.get("meta", {}) or {}
        citation = meta.get("source", doc.get("source", "unknown"))
        
        retrieval_score = float(doc.get("score", 0.0))
        rerank_score = float(doc.get("rerank_score", retrieval_score))
        
        if not text:
            _logger.warning("Empty text for doc_id=%s", doc_id)
            return []
        
        # For now, use full text as single snippet
        # TODO: Implement proper chunking if needed
        snippets.append(SnippetInfo(
            text=text,
            doc_id=doc_id,
            citation=citation,
            retrieval_score=retrieval_score,
            rerank_score=rerank_score,
            has_embedding=False
        ))
        
        return snippets
    
    async def _encode_texts(
        self,
        texts: List[str],
        operation: str = "query"
    ) -> List[List[float]]:
        """Encode avec cache normalisé et métriques."""
        if not texts:
            return []
        
        normalized_texts = [normalize_cache_key(t) for t in texts]
        
        results = []
        to_encode = []
        to_encode_indices = []
        
        # Check cache
        for i, (original, normalized) in enumerate(zip(texts, normalized_texts)):
            if self.cfg.enable_caching and normalized in self._embedding_cache:
                results.append(self._embedding_cache[normalized])
                self._cache_hits += 1
            else:
                results.append(None)
                to_encode.append(original)
                to_encode_indices.append(i)
                self._cache_misses += 1
        
        # Encoder ce qui manque
        if to_encode:
            try:
                if hasattr(self.encoder, "encode_async"):
                    encoded = await asyncio.wait_for(
                        self.encoder.encode_async(to_encode),
                        timeout=self.cfg.encode_timeout_s
                    )
                else:
                    encoded = await asyncio.wait_for(
                        asyncio.to_thread(self.encoder.encode, to_encode),
                        timeout=self.cfg.encode_timeout_s
                    )
                
                # Stocker dans cache avec clés NORMALISÉES
                for idx, emb in zip(to_encode_indices, encoded):
                    results[idx] = emb
                    if self.cfg.enable_caching:
                        norm_key = normalized_texts[idx]
                        self._embedding_cache[norm_key] = emb
                
                # Sauvegarder périodiquement
                if self._cache_misses % 10 == 0:
                    self._save_cache()
            
            except asyncio.TimeoutError:
                raise TimeoutError(f"Encoding timeout après {self.cfg.encode_timeout_s}s")
            except Exception as e:
                _logger.exception("Encoding failed: %s", e)
                raise
        
        return results

    def _apply_diversity_filter(
        self,
        snippets: List[SnippetInfo],
        max_items: int,
        penalty: float
    ) -> List[SnippetInfo]:
        """Apply simple diversity filter based on doc_id"""
        selected: List[SnippetInfo] = []
        doc_counts: Dict[str, int] = {}
        
        for snip in snippets:
            if len(selected) >= max_items:
                break
            
            count = doc_counts.get(snip.doc_id, 0)
            adjusted_score = snip.rerank_score - (penalty * count)
            
            if adjusted_score >= 0 or len(selected) < max_items // 2:
                selected.append(snip)
                doc_counts[snip.doc_id] = count + 1
        
        return selected
    
    def _build_context(
        self,
        query: str,
        snippets: List[SnippetInfo],
        retrieval_time: float,
        processing_time: float
    ) -> ContextInfo:
        """Build final context info"""
        total_tokens = sum(
            len(s.text.split()) for s in snippets
        )
        
        return ContextInfo(
            query=query,
            snippets=snippets,
            total_tokens=total_tokens,
            retrieval_time_s=retrieval_time,
            processing_time_s=processing_time,
            is_timeout=False
        )
    
    def _empty_response(
        self,
        request_id: str,
        query: str,
        retrieval_time: float,
        processing_time: float,
        diagnostics: DiagnosticsInfo,
        is_timeout: bool
    ) -> AugmentedResponse:
        """Create empty response for failures"""
        context = ContextInfo(
            query=query,
            snippets=[],
            total_tokens=0,
            retrieval_time_s=retrieval_time,
            processing_time_s=processing_time,
            is_timeout=is_timeout
        )
        
        diagnostics.timings = {
            "total_time_s": processing_time,
            "augment_time_s": processing_time,
            "format_time_s": 0.0
        }
        diagnostics.summary = {
            "snippets_returned": 0,
            "total_tokens": 0,
            "is_timeout": is_timeout,
            "query": query
        }
        diagnostics.cache = {
            "cache_size": len(self._embedding_cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses
        }
        
        return AugmentedResponse(
            request_id=request_id,
            query=query,
            context=context,
            diagnostics=diagnostics
        )