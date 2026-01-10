# rag/augmentation_pipeline.py
from __future__ import annotations
import asyncio
import time
import uuid
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

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
    candidate_pool: int = 200  # Récupérer plus pour avoir de la marge
    
    # Snippet extraction
    max_snippets_per_doc: int = 3
    snippet_max_tokens: int = 200
    snippet_overlap: int = 50  # Chevauchement entre snippets
    min_snippet_score: float = 0.1
    
    # Scoring weights
    retrieval_weight: float = 0.3
    rerank_weight: float = 0.5
    semantic_weight: float = 0.2
    support_threshold: float = 0.75
    diversity_penalty: float = 0.1
    
    # Timeouts (avec retry)
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
    cache_ttl_s: float = 3600.0  # 1h TTL pour le cache
    warm_cache: bool = True  # Pré-chauffer le cache
    
    # Cross-encoder reranking
    use_cross_encoder: bool = True
    cross_encoder_batch: int = 16
    
    # Quality controls
    enable_deduplication: bool = True
    dedup_threshold: float = 0.85  # Similarité cosine
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
    
    Usage:
        pipeline = AugmentationPipeline(
            retriever=retriever,
            encoder=encoder,
            reranker=reranker,
            cfg=AugmentationConfig()
        )
        
        response = await pipeline.augment("Ma question ?")
        print(response.to_dict())
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
        
        # Cache for embeddings
        self._cache: Dict[str, List[float]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Semaphore for concurrent operations
        self._sem = asyncio.Semaphore(self.cfg.concurrency)
    
    async def augment(
        self, 
        query: str,
        request_id: Optional[str] = None,
        cfg_override: Optional[AugmentationConfig] = None
    ) -> AugmentedResponse:
        """
        Point d'entrée principal : augmente le contexte pour une requête.
        
        Args:
            query: La question/requête utilisateur
            request_id: ID optionnel pour tracer la requête
            cfg_override: Configuration optionnelle qui override self.cfg
            
        Returns:
            AugmentedResponse avec contexte enrichi et diagnostics
        """
        cfg = cfg_override or self.cfg
        request_id = request_id or f"aug_{int(time.time() * 1000)}"
        
        start_time = time.time()
        diagnostics = DiagnosticsInfo()
        
        try:
            # Step 1: Retrieve documents
            retrieve_start = time.time()
            docs = await self._retrieve_documents(query, cfg, diagnostics)
            retrieve_time = time.time() - retrieve_start
            
            if not docs:
                return self._empty_response(
                    request_id, query, retrieve_time, 
                    time.time() - start_time, diagnostics, False
                )
            
            # Step 2: Rerank documents (optionnel)
            if self.reranker and cfg.use_cross_encoder:
                await self._rerank_documents(query, docs, cfg, diagnostics)
            
            # Step 3: Extract and encode snippets
            snippets = await self._extract_snippets(query, docs, cfg, diagnostics)
            
            if not snippets:
                return self._empty_response(
                    request_id, query, retrieve_time,
                    time.time() - start_time, diagnostics, False
                )
            
            # Step 4: Format final response
            format_start = time.time()
            context = self._build_context(
                query, snippets, retrieve_time, 
                time.time() - start_time
            )
            format_time = time.time() - format_start
            
            # Step 5: Complete diagnostics
            diagnostics.steps["format"] = {
                "count": len(snippets),
                "time_s": format_time
            }
            diagnostics.cache = {
                "cache_size": len(self._cache),
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses
            }
            diagnostics.timings = {
                "total_time_s": time.time() - start_time,
                "augment_time_s": time.time() - start_time - format_time,
                "format_time_s": format_time
            }
            diagnostics.summary = {
                "snippets_returned": len(snippets),
                "total_tokens": context.total_tokens,
                "is_timeout": False,
                "query": query
            }
            
            return AugmentedResponse(
                request_id=request_id,
                query=query,
                context=context,
                diagnostics=diagnostics
            )
            
        except asyncio.TimeoutError:
            _logger.warning(f"Augmentation timed out for query: {query}")
            return self._empty_response(
                request_id, query, 0.0, 
                time.time() - start_time, diagnostics, True
            )
        except Exception as e:
            _logger.exception(f"Augmentation failed for query: {query}")
            return self._empty_response(
                request_id, query, 0.0,
                time.time() - start_time, diagnostics, False
            )
    
    async def _retrieve_documents(
        self, 
        query: str, 
        cfg: AugmentationConfig,
        diagnostics: DiagnosticsInfo
    ) -> List[Dict[str, Any]]:
        """Step 1: Retrieve documents from index"""
        try:
            retrieve_coro = self.retriever.retrieve(
                query=query,
                top_k=cfg.top_k
            )
            docs = await asyncio.wait_for(
                retrieve_coro,
                timeout=cfg.retrieval_timeout_s
            )
            
            diagnostics.steps["retrieve"] = {
                "count": len(docs),
                "time_s": 0.0  # Will be set by caller
            }
            
            return docs
            
        except asyncio.TimeoutError:
            _logger.warning("Retrieval timed out")
            diagnostics.steps["retrieve"] = {
                "count": 0,
                "time_s": cfg.retrieval_timeout_s,
                "timeout": True
            }
            return []
        except Exception as e:
            _logger.exception("Retrieval failed")
            diagnostics.steps["retrieve"] = {
                "count": 0,
                "time_s": 0.0,
                "error": str(e)
            }
            return []
    
    async def _rerank_documents(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        cfg: AugmentationConfig,
        diagnostics: DiagnosticsInfo
    ) -> None:
        """Step 2: Rerank documents using cross-encoder"""
        rerank_start = time.time()
        try:
            # Prepare docs for reranking
            rerank_docs = [{
                "id": d.get("id") or d.get("doc_id"),
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
                "time_s": time.time() - rerank_start
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
                "time_s": time.time() - rerank_start,
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
        encode_start = time.time()
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
                    if cfg.enable_caching and snip.text in self._cache:
                        snip.has_embedding = True
                        self._cache_hits += 1
                    else:
                        snippet_map.append(len(snippets) - 1)
                        texts_to_encode.append(snip.text)
                        self._cache_misses += 1
            
            # Encode uncached snippets
            if texts_to_encode:
                embeddings = await self._encode_texts(
                    texts_to_encode, cfg, diagnostics
                )
                
                # Update snippets with embeddings
                for idx, emb in zip(snippet_map, embeddings):
                    snippets[idx].has_embedding = len(emb) > 0
                    if cfg.enable_caching and len(emb) > 0:
                        self._cache[snippets[idx].text] = emb
            
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
            
            return snippets[:cfg.rerank_top_n]
            
        except Exception as e:
            _logger.exception("Snippet extraction failed")
            diagnostics.steps["encode"] = {
                "texts_encoded": 0,
                "time_s": time.time() - encode_start,
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
        
        # If doc has pre-chunked snippets
        if "chunks" in doc and doc["chunks"]:
            for chunk in doc["chunks"]:
                text = chunk.get("text", "").strip()
                if text:
                    snippets.append(SnippetInfo(
                        text=text,
                        doc_id=doc.get("id") or doc.get("doc_id", "unknown"),
                        citation=doc.get("source") or chunk.get("source", "unknown"),
                        retrieval_score=float(doc.get("score", 0.0)),
                        rerank_score=float(doc.get("rerank_score", doc.get("score", 0.0))),
                        has_embedding=False
                    ))
        else:
            # Use full text as single snippet
            text = doc.get("text", "").strip()
            if text:
                snippets.append(SnippetInfo(
                    text=text,
                    doc_id=doc.get("id") or doc.get("doc_id", "unknown"),
                    citation=doc.get("source", "unknown"),
                    retrieval_score=float(doc.get("score", 0.0)),
                    rerank_score=float(doc.get("rerank_score", doc.get("score", 0.0))),
                    has_embedding=False
                ))
        
        return snippets
    
    async def _encode_texts(
        self,
        texts: List[str],
        cfg: AugmentationConfig,
        diagnostics: DiagnosticsInfo
    ) -> List[List[float]]:
        """Encode texts using the encoder"""
        encode_start = time.time()
        try:
            if hasattr(self.encoder, "encode_async"):
                embeddings = await asyncio.wait_for(
                    self.encoder.encode_async(texts),
                    timeout=cfg.encode_timeout_s
                )
            else:
                embeddings = await asyncio.wait_for(
                    asyncio.to_thread(self.encoder.encode, texts),
                    timeout=cfg.encode_timeout_s
                )
            
            diagnostics.steps["encode"] = {
                "texts_encoded": len(texts),
                "time_s": time.time() - encode_start
            }
            
            return embeddings
            
        except asyncio.TimeoutError:
            _logger.warning("Encoding timed out")
            diagnostics.steps["encode"] = {
                "texts_encoded": 0,
                "time_s": cfg.encode_timeout_s,
                "timeout": True
            }
            return [[] for _ in texts]
        except Exception as e:
            _logger.exception("Encoding failed")
            diagnostics.steps["encode"] = {
                "texts_encoded": 0,
                "time_s": time.time() - encode_start,
                "error": str(e)
            }
            return [[] for _ in texts]
    
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
            
            # Count how many snippets from same doc
            count = doc_counts.get(snip.doc_id, 0)
            
            # Apply penalty
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
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses
        }
        
        return AugmentedResponse(
            request_id=request_id,
            query=query,
            context=context,
            diagnostics=diagnostics
        )


# -------------------------
# Example Usage
# -------------------------
async def example_usage():
    """Exemple d'utilisation de la pipeline"""
    from rag.utils.embedder import Embedder, EmbedderConfig
    from rag.utils.retriever import Retriever, RetrieverConfig
    from rag.utils.reRank import Reranker, RerankConfig
    from rag.utils.crossEncoder import CrossEncoder
    
    # Setup components
    embedder = Embedder(cfg=EmbedderConfig())
    retriever = Retriever(
        embedder=embedder,
        cfg=RetrieverConfig(top_k=50)
    )
    cross_encoder = CrossEncoder()
    reranker = Reranker(
        cross_encoder=cross_encoder,
        cfg=RerankConfig()
    )
    
    # Create pipeline
    pipeline = AugmentationPipeline(
        retriever=retriever,
        encoder=embedder,
        reranker=reranker,
        cfg=AugmentationConfig(
            top_k=50,
            rerank_top_n=10,
            use_cross_encoder=True
        )
    )
    
    # Run augmentation
    query = "Quelle est la tendance pluviométrique pour la période 11-20 septembre 2025 ?"
    response = await pipeline.augment(query)
    
    # Get formatted output
    output = response.to_dict()
    
    import json
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(example_usage())