# run_meteo.py
from ragPipeline import RAGPipeline
from retriever import Retriever
from embedder import Embedder
from reRank import ReRanker
from promptBuilder import PromptBuilder
from generator import UniversalGenerator
from evaluator import Evaluator
from logBuilder import Logger

from ingestionAdapter import IngestionRouter
from documentLoader import DocumentLoader
from indexer import GenericIndexer
from buildText import build_text, build_meta

# === Configuration ===
COLLECTION_NAME = "meteo_fanfar"
DATA_PATH = "bulletins_json/"  # dossier contenant tes fichiers
PERSIST_DIR = "chroma_persist"

# === Ingestion et normalisation (pré-indexation explicite) ===
router = IngestionRouter()
loader = DocumentLoader(DATA_PATH)

# Charger les documents (avec noms de fichiers)
docs = loader.load_all(with_filenames=True)

# Router + normaliser -> records
records = []
for doc, fname in docs:
    recs = router.route(doc, fname)
    if recs:
        records.extend(recs)

print(f"Records produits pour indexation: {len(records)}")

# === Indexation ===
embedder = Embedder()  # stub ou implémentation réelle
indexer = GenericIndexer(collection_name=COLLECTION_NAME, persist_dir=PERSIST_DIR)

indexer.index_records(records, text_builder=build_text, metadata_builder=build_meta)

# === Composants RAG (après indexation) ===
retriever = Retriever(persist_dir=PERSIST_DIR, collection_name=COLLECTION_NAME)
reranker = ReRanker()
prompt_builder = PromptBuilder()
generator = UniversalGenerator()
evaluator = Evaluator()
logger = Logger("logs/meteo_rag.jsonl")

# === Pipeline complet ===
pipeline = RAGPipeline(
    retriever=retriever,
    generator=generator,
    reranker=reranker,
    prompt_builder=prompt_builder,
    evaluator=evaluator,
    logger=logger
)

# === Requête utilisateur ===
query = "Quel est le climat typique à Bobo-Dioulasso en août ?"
result = pipeline.answer(query)

# === Affichage ===
print("🧠 Réponse :", result.get("response"))
print("📚 Sources :", [s.get("excerpt")[:120] for s in result.get("sources", [])])
print("📊 Scores :", result.get("scores"))
"""
augmenter.py - Enhanced Context Augmentation Pipeline for RAG
Improvements:
- Complete implementation of all methods
- Better error handling and recovery
- Enhanced logging and observability
- Type safety improvements
- Performance optimizations
"""
from __future__ import annotations
import asyncio
import time
import logging
import uuid
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from collections import defaultdict

_logger = logging.getLogger("augmenter")

# Type aliases
Doc = Dict[str, Any]
Vec = Sequence[float]


@dataclass
class AugmenterConfig:
    """Configuration for context augmentation pipeline"""
    # Retrieval
    top_k: int = 50
    rerank_top_n: int = 20
    
    # Snippet extraction
    snippet_max_tokens: int = 200
    snippet_overlap: int = 50
    min_snippet_score: float = 0.1
    max_snippets_per_doc: int = 3
    
    # Scoring
    support_threshold: float = 0.75
    diversity_penalty: float = 0.1
    
    # Performance
    timeout_s: float = 5.0
    concurrency: int = 8
    enable_caching: bool = True
    cache_max_size: int = 10000
    
    # Cross-encoder
    use_cross_encoder: bool = False
    cross_encoder_batch: int = 16
    cross_encoder_top_k: int = 50
    
    # Metrics
    metrics_hook: Optional[Callable[[Dict[str, Any]], None]] = None


@dataclass
class AugmentedResponse:
    """Response containing augmented context"""
    request_id: str
    query: str
    docs: List[Doc] = field(default_factory=list)
    snippets: List[Dict[str, Any]] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    support: bool = False


class Augmenter:
    """
    Context augmentation for RAG pipelines.
    
    Pipeline:
    1. Retrieve candidates (vector search)
    2. Rerank with dot-product similarity
    3. Optional cross-encoder rerank
    4. Extract snippets with sliding window
    5. Assemble citations
    6. Compute confidence score
    """
    
    def __init__(
        self,
        cfg: AugmenterConfig,
        retriever: Any,
        embedder: Any,
        cross_encoder: Optional[Any] = None,
    ):
        self.cfg = cfg
        self.retriever = retriever
        self.embedder = embedder
        self.cross_encoder = cross_encoder
        self._semaphore = asyncio.Semaphore(cfg.concurrency)
        self._snippet_cache: Dict[str, List[float]] = {} if cfg.enable_caching else None
        self._cache_hits = 0
        self._cache_misses = 0

    async def get_augmented(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        ctx: Optional[Dict[str, Any]] = None
    ) -> AugmentedResponse:
        """Main entrypoint for context augmentation"""
        request_id = (ctx or {}).get("request_id") or str(uuid.uuid4())
        start_time = time.time()
        
        resp = AugmentedResponse(request_id=request_id, query=query)
        diagnostics: Dict[str, Any] = {"stages": []}
        
        try:
            async with asyncio.timeout(self.cfg.timeout_s):
                # 1. Encode query
                t0 = time.time()
                q_vecs = await self._encode([query], is_query=True)
                diagnostics["stages"].append({
                    "stage": "encode_query",
                    "latency_ms": int((time.time() - t0) * 1000)
                })
                
                if not q_vecs:
                    _logger.warning(f"Query encoding failed: {request_id}")
                    resp.diagnostics = diagnostics
                    return resp
                
                q_vec = q_vecs[0]
                
                # 2. Retrieve candidates
                t0 = time.time()
                expr = self._filters_to_expr(filters)
                raw_hits = await self.retriever.search(
                    [q_vec],
                    top_k=self.cfg.top_k,
                    expr=expr
                )
                candidates = raw_hits[0] if raw_hits and raw_hits[0] else []
                diagnostics["stages"].append({
                    "stage": "retrieve",
                    "latency_ms": int((time.time() - t0) * 1000),
                    "candidates": len(candidates)
                })
                
                if not candidates:
                    _logger.info(f"No candidates found: {request_id}")
                    resp.diagnostics = diagnostics
                    return resp
                
                # 3. Lightweight rerank (dot-product)
                t0 = time.time()
                reranked = await self._rerank_dot(query, q_vec, candidates)
                diagnostics["stages"].append({
                    "stage": "rerank_dot",
                    "latency_ms": int((time.time() - t0) * 1000),
                    "candidates_after": len(reranked)
                })
                
                # 4. Optional cross-encoder rerank
                reranked_top = reranked[:self.cfg.cross_encoder_top_k]
                
                if self.cfg.use_cross_encoder and self.cross_encoder:
                    t0 = time.time()
                    reranked_final = await self._cross_rerank(query, reranked_top)
                    diagnostics["stages"].append({
                        "stage": "cross_rerank",
                        "latency_ms": int((time.time() - t0) * 1000),
                        "candidates_after": len(reranked_final)
                    })
                else:
                    reranked_final = reranked_top[:self.cfg.rerank_top_n]
                
                # 5. Snippet extraction
                t0 = time.time()
                snippets = await self._extract_snippets(query, q_vec, reranked_final)
                diagnostics["stages"].append({
                    "stage": "snippet_extraction",
                    "latency_ms": int((time.time() - t0) * 1000),
                    "snippets": len(snippets),
                    "cache_hits": self._cache_hits,
                    "cache_misses": self._cache_misses
                })
                
                # 6. Assemble citations
                t0 = time.time()
                citations = self._assemble_citations(reranked_final, snippets)
                diagnostics["stages"].append({
                    "stage": "citation_assembly",
                    "latency_ms": int((time.time() - t0) * 1000),
                    "citations": len(citations)
                })
                
                # 7. Confidence scoring
                t0 = time.time()
                confidence = self._confidence_score(query, reranked_final, snippets)
                support = confidence >= self.cfg.support_threshold
                diagnostics["stages"].append({
                    "stage": "confidence",
                    "latency_ms": int((time.time() - t0) * 1000),
                    "confidence": round(confidence, 3),
                    "support": support
                })
                
                # Fill response
                resp.docs = reranked_final
                resp.snippets = snippets
                resp.citations = citations
                resp.confidence = float(confidence)
                resp.support = bool(support)
                resp.diagnostics = diagnostics
                
                # Metrics hook
                if self.cfg.metrics_hook:
                    try:
                        self.cfg.metrics_hook({
                            "event": "get_augmented",
                            "request_id": request_id,
                            "candidates": len(candidates),
                            "snippets": len(snippets),
                            "confidence": confidence,
                            "latency_ms": int((time.time() - start_time) * 1000)
                        })
                    except Exception as e:
                        _logger.error(f"Metrics hook failed: {e}")
                
                return resp
        
        except asyncio.TimeoutError:
            _logger.warning(f"Request timeout: {request_id}")
            resp.diagnostics = diagnostics
            resp.diagnostics["warning"] = "timeout"
            return resp
        except Exception as e:
            _logger.error(f"Request failed: {request_id}", exc_info=True)
            resp.diagnostics = diagnostics
            resp.diagnostics["error"] = str(e)
            return resp
        finally:
            elapsed_ms = int((time.time() - start_time) * 1000)
            resp.diagnostics["total_latency_ms"] = elapsed_ms

    # ==================== ENCODING ====================
    
    async def _encode(self, texts: Sequence[str], is_query: bool = False) -> List[List[float]]:
        """Encode texts using embedder with proper async handling"""
        if not texts:
            return []
        
        try:
            # Try async first
            if hasattr(self.embedder, "encode_async"):
                return await self.embedder.encode_async(list(texts), is_query=is_query)
            
            # Fallback to executor
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.embedder.encode(list(texts), is_query=is_query)
            )
        except Exception as e:
            _logger.error(f"Encoding failed: {e}", exc_info=True)
            return []

    # ==================== FILTERING ====================
    
    def _filters_to_expr(self, filters: Optional[Dict[str, Any]]) -> Optional[str]:
        """Convert filters dict to Milvus expression with validation"""
        if not filters:
            return None
        if isinstance(filters, str):
            return filters
        
        parts = []
        for key, value in filters.items():
            if value is None:
                continue
            
            # Sanitize key
            key = str(key).replace("'", "").replace('"', "")
            
            if isinstance(value, (list, tuple)) and len(value) == 2:
                # Range: key >= v[0] && key <= v[1]
                parts.append(f"{key} >= {value[0]} && {key} <= {value[1]}")
            elif isinstance(value, str):
                # Escape quotes
                value = value.replace("'", "\\'")
                parts.append(f"{key} == '{value}'")
            elif isinstance(value, (int, float)):
                parts.append(f"{key} == {value}")
            elif isinstance(value, list):
                # IN operator
                values_str = ", ".join(
                    f"'{str(v).replace(\"'\", \"\\\\'\") }'" if isinstance(v, str) else str(v) 
                    for v in value
                )
                parts.append(f"{key} in [{values_str}]")
        
        return " && ".join(parts) if parts else None

    # ==================== RERANKING ====================
    
    async def _rerank_dot(self, query: str, q_vec: Vec, candidates: List[Doc]) -> List[Doc]:
        """Rerank using dot-product similarity with proper vector handling"""
        if not candidates:
            return []
        
        # Extract or encode vectors
        vecs = []
        missing_indices = []
        texts_to_encode = []
        
        for i, cand in enumerate(candidates):
            vec = cand.get("vector") or cand.get("vec") or cand.get("embedding")
            if vec and isinstance(vec, (list, tuple)):
                vecs.append(list(vec))
            else:
                vecs.append(None)
                missing_indices.append(i)
                texts_to_encode.append(cand.get("text", ""))
        
        # Encode missing vectors
        if missing_indices:
            try:
                encoded = await self._encode(texts_to_encode, is_query=False)
                for j, idx in enumerate(missing_indices):
                    if j < len(encoded):
                        vecs[idx] = encoded[j]
            except Exception as e:
                _logger.warning(f"Vector encoding failed during rerank: {e}")
                # Fallback to zero vectors
                dim = len(q_vec) if q_vec else 384
                for idx in missing_indices:
                    if vecs[idx] is None:
                        vecs[idx] = [0.0] * dim
        
        # Compute dot-product scores
        scores = []
        for vec in vecs:
            if vec and len(vec) == len(q_vec):
                try:
                    score = float(sum(a * b for a, b in zip(q_vec, vec)))
                except (TypeError, ValueError):
                    score = 0.0
            else:
                score = 0.0
            scores.append(score)
        
        # Attach scores and sort
        for cand, score in zip(candidates, scores):
            cand["score"] = float(score)
        
        return sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)

    async def _cross_rerank(self, query: str, candidates: List[Doc]) -> List[Doc]:
        """Optional cross-encoder rerank with fallback"""
        if not self.cross_encoder or not candidates:
            return candidates[:self.cfg.rerank_top_n]
        
        try:
            # Try async rerank
            if hasattr(self.cross_encoder, "rerank_async"):
                result = await self.cross_encoder.rerank_async(
                    query, candidates, self.cfg.rerank_top_n
                )
                return result if result else candidates[:self.cfg.rerank_top_n]
            
            # Fallback to sync
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.cross_encoder.rerank(query, candidates, self.cfg.rerank_top_n)
            )
            return result if result else candidates[:self.cfg.rerank_top_n]
        except Exception as e:
            _logger.error(f"Cross-rerank failed, using dot-product results: {e}")
            return candidates[:self.cfg.rerank_top_n]

    # ==================== SNIPPET EXTRACTION ====================
    
    def _cache_key(self, text: str) -> str:
        """Generate cache key for snippet text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    async def _extract_snippets(
        self,
        query: str,
        q_vec: Vec,
        candidates: List[Doc]
    ) -> List[Dict[str, Any]]:
        """Extract top snippets using sliding window with caching"""
        if not candidates:
            return []
        
        # Prepare windows
        windows_metadata: List[Dict[str, Any]] = []
        texts_to_encode: List[str] = []
        
        for doc_idx, doc in enumerate(candidates):
            text = doc.get("text") or ""
            tokens = text.split()
            
            if not tokens:
                continue
            
            win_size = max(1, self.cfg.snippet_max_tokens)
            step = max(1, win_size - self.cfg.snippet_overlap)
            
            for start_idx in range(0, len(tokens), step):
                end_idx = min(len(tokens), start_idx + win_size)
                span_text = " ".join(tokens[start_idx:end_idx])
                
                if not span_text.strip():
                    continue
                
                # Check cache
                cache_key = None
                cached_vec = None
                
                if self._snippet_cache is not None:
                    cache_key = self._cache_key(span_text)
                    cached_vec = self._snippet_cache.get(cache_key)
                    
                    if cached_vec:
                        self._cache_hits += 1
                        windows_metadata.append({
                            "doc_idx": doc_idx,
                            "doc_id": doc.get("id"),
                            "start": start_idx,
                            "end": end_idx,
                            "text": span_text,
                            "doc_score": float(doc.get("score", 0.0)),
                            "meta": doc.get("meta") or {},
                            "vec": cached_vec,
                            "cache_key": cache_key
                        })
                        if end_idx >= len(tokens):
                            break
                        continue
                    else:
                        self._cache_misses += 1
                
                windows_metadata.append({
                    "doc_idx": doc_idx,
                    "doc_id": doc.get("id"),
                    "start": start_idx,
                    "end": end_idx,
                    "text": span_text,
                    "doc_score": float(doc.get("score", 0.0)),
                    "meta": doc.get("meta") or {},
                    "vec": None,
                    "cache_key": cache_key
                })
                texts_to_encode.append(span_text)
                
                if end_idx >= len(tokens):
                    break
        
        # Encode windows
        if texts_to_encode:
            try:
                window_vecs = await self._encode(texts_to_encode, is_query=False)
                vec_idx = 0
                for metadata in windows_metadata:
                    if metadata["vec"] is None and vec_idx < len(window_vecs):
                        metadata["vec"] = window_vecs[vec_idx]
                        # Cache with size limit
                        if self._snippet_cache is not None and metadata["cache_key"]:
                            if len(self._snippet_cache) >= self.cfg.cache_max_size:
                                # Remove oldest entry
                                self._snippet_cache.pop(next(iter(self._snippet_cache)), None)
                            self._snippet_cache[metadata["cache_key"]] = window_vecs[vec_idx]
                        vec_idx += 1
            except Exception as e:
                _logger.error(f"Snippet encoding failed: {e}", exc_info=True)
                return []
        
        # Score windows
        per_doc_snippets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        
        for metadata in windows_metadata:
            vec = metadata["vec"]
            if not vec or len(vec) != len(q_vec):
                continue
            
            try:
                score = float(sum(a * b for a, b in zip(q_vec, vec)))
            except (TypeError, ValueError):
                score = 0.0
            
            if score < self.cfg.min_snippet_score:
                continue
            
            snippet = {
                "doc_id": metadata["doc_id"],
                "doc_score": metadata["doc_score"],
                "span": (metadata["start"], metadata["end"]),
                "text": metadata["text"],
                "score": score,
                "meta": metadata["meta"]
            }
            per_doc_snippets[metadata["doc_idx"]].append(snippet)
        
        # Select top snippets per doc
        all_snippets = []
        for doc_idx, snippets in per_doc_snippets.items():
            # Sort by score
            snippets_sorted = sorted(snippets, key=lambda x: x["score"], reverse=True)
            # Take top N per doc
            top_snippets = snippets_sorted[:self.cfg.max_snippets_per_doc]
            all_snippets.extend(top_snippets)
        
        # Apply diversity penalty and sort
        all_snippets = self._apply_diversity_penalty(all_snippets)
        all_snippets.sort(key=lambda x: x["score"], reverse=True)
        
        return all_snippets

    def _apply_diversity_penalty(self, snippets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply penalty for multiple snippets from same document"""
        doc_counts: Dict[Any, int] = defaultdict(int)
        
        for snippet in snippets:
            doc_id = snippet["doc_id"]
            count = doc_counts[doc_id]
            penalty = 1.0 - (count * self.cfg.diversity_penalty)
            snippet["score"] = snippet["score"] * max(0.0, min(1.0, penalty))
            doc_counts[doc_id] += 1
        
        return snippets

    # ==================== CITATIONS ====================
    
    def _assemble_citations(
        self,
        docs: List[Doc],
        snippets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Assemble citations from snippets with metadata"""
        citations = []
        
        for idx, snippet in enumerate(snippets):
            citation = {
                "index": idx,
                "doc_id": snippet["doc_id"],
                "text": snippet["text"],
                "score": round(snippet["score"], 4),
                "span": snippet.get("span"),
                "meta": snippet.get("meta", {})
            }
            citations.append(citation)
        
        return citations

    # ==================== CONFIDENCE SCORING ====================
    
    def _confidence_score(
        self,
        query: str,
        docs: List[Doc],
        snippets: List[Dict[str, Any]]
    ) -> float:
        """
        Compute confidence score based on:
        - Number and quality of snippets
        - Top document scores
        - Score distribution
        """
        if not snippets and not docs:
            return 0.0
        
        # Component 1: Snippet quality (0-0.4)
        snippet_score = 0.0
        if snippets:
            top_snippet_scores = [s["score"] for s in snippets[:5]]
            avg_top_score = sum(top_snippet_scores) / len(top_snippet_scores)
            snippet_score = min(0.4, avg_top_score * 0.4)
        
        # Component 2: Document scores (0-0.3)
        doc_score = 0.0
        if docs:
            top_doc_scores = [d.get("score", 0.0) for d in docs[:3]]
            avg_doc_score = sum(top_doc_scores) / len(top_doc_scores) if top_doc_scores else 0.0
            doc_score = min(0.3, avg_doc_score * 0.3)
        
        # Component 3: Coverage (0-0.2)
        coverage_score = 0.0
        if snippets:
            num_unique_docs = len(set(s["doc_id"] for s in snippets))
            coverage_score = min(0.2, (num_unique_docs / 5.0) * 0.2)
        
        # Component 4: Consistency (0-0.1)
        consistency_score = 0.0
        if len(snippets) >= 2:
            scores = [s["score"] for s in snippets[:10]]
            score_variance = sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)
            # Lower variance = higher consistency
            consistency_score = max(0.0, 0.1 - score_variance * 0.1)
        
        total_confidence = snippet_score + doc_score + coverage_score + consistency_score
        return min(1.0, max(0.0, total_confidence))
    
    def clear_cache(self) -> int:
        """Clear snippet cache and return number of entries cleared"""
        if self._snippet_cache is None:
            return 0
        count = len(self._snippet_cache)
        self._snippet_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        return count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if self._snippet_cache is None:
            return {"enabled": False}
        
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "enabled": True,
            "size": len(self._snippet_cache),
            "max_size": self.cfg.cache_max_size,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": round(hit_rate, 3)
        }