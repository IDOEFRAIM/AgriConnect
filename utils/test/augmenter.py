# rag/augmenter.py
from __future__ import annotations
import asyncio
import time
import re
import json
import uuid
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

_logger = logging.getLogger("rag.augmenter")
_logger.addHandler(logging.NullHandler())

Vec = List[float]
Doc = Dict[str, Any]

# -------------------------
# Configuration
# -------------------------
@dataclass
class AugmenterConfig:
    top_k: int = 50
    rerank_top_n: int = 20
    max_snippets_per_doc: int = 3
    snippet_max_tokens: int = 200
    snippet_overlap: int = 50
    min_snippet_score: float = 0.1
    support_threshold: float = 0.75
    diversity_penalty: float = 0.1
    timeout_s: float = 8.0
    concurrency: int = 8
    enable_caching: bool = True
    use_cross_encoder: bool = False
    cross_encoder_batch: int = 16
    cross_encoder_top_k: int = 50

# -------------------------
# Interfaces expected
# -------------------------
class AbstractRetriever:
    async def retrieve(self, query: str, top_k: int, **kwargs) -> List[Doc]:
        raise NotImplementedError

class AbstractEncoder:
    async def encode(self, texts: Sequence[str]) -> Sequence[Vec]:
        raise NotImplementedError

class AbstractReranker:
    async def rerank(self, query: str, docs: List[Doc], top_n: int, **kwargs) -> List[Doc]:
        raise NotImplementedError

# -------------------------
# Data structures
# -------------------------
@dataclass
class Snippet:
    text: str
    doc_id: str
    source: str
    retrieval_score: float = 0.0
    rerank_score: float = 0.0
    embedding: Optional[Vec] = None

@dataclass
class AugmentedContext:
    query: str
    snippets: List[Snippet]
    total_tokens: int
    retrieval_time_s: float
    processing_time_s: float
    is_timeout: bool = False

# -------------------------
# Utilities
# -------------------------
def _cosine_similarity(a: Vec, b: Vec) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def _token_estimate(text: str) -> int:
    return max(1, len(text.split()))

def _split_sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[.?!…])\s+', text)
    return [p.strip() for p in parts if p.strip()]

# -------------------------
# Robust JSON helpers (for upstream data extraction)
# -------------------------
def load_json_path(path: str, max_bytes: int = 10_000_000) -> List[Dict[str, Any]]:
    p = PathLike(path)
    size = p.stat().st_size
    if size > max_bytes:
        raise ValueError(f"File too large: {size} bytes (limit {max_bytes})")
    raw = p.read_bytes()
    try:
        data = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        data = json.loads(raw.decode("latin-1"))
    return normalize_raw(data, source=str(p))

def normalize_raw(data: Any, source: Optional[str] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        d = dict(data)
        if source:
            d.setdefault("source", source)
        out.append(d)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                d = dict(item)
                if source:
                    d.setdefault("source", source)
                out.append(d)
    else:
        raise ValueError("JSON must be an object or an array of objects")
    return out

def extract_text(doc: Dict[str, Any]) -> str:
    candidates = [
        doc.get("text"),
        (doc.get("meta") or {}).get("text"),
        doc.get("content"),
        doc.get("body"),
        doc.get("title"),
        doc.get("summary"),
    ]
    for c in candidates:
        if isinstance(c, str) and c.strip():
            return c.strip()
    # fallback: join stringifiable fields
    joined = " ".join(str(doc.get(k, "")) for k in ("title", "summary", "body", "content") if doc.get(k))
    return joined.strip()

# -------------------------
# Safe wrappers for external calls
# -------------------------
async def safe_encode(encoder: AbstractEncoder, texts: List[str], timeout: float, retries: int = 2) -> List[Vec]:
    last_exc = None
    for attempt in range(retries):
        try:
            coro = encoder.encode(texts)
            if asyncio.iscoroutine(coro):
                return list(await asyncio.wait_for(coro, timeout=timeout))
            else:
                return list(coro)
        except asyncio.TimeoutError as e:
            last_exc = e
            if attempt < retries - 1:
                await asyncio.sleep(0.05)
                continue
            raise
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                await asyncio.sleep(0.05)
                continue
            raise last_exc
    raise last_exc

# -------------------------
# PathLike helper (small wrapper to avoid extra imports)
# -------------------------
import os
from pathlib import Path
def PathLike(p: str) -> Path:
    return Path(p)

# -------------------------
# ContextAugmenter (no TextBuilder)
# -------------------------
class ContextAugmenter:
    def __init__(
        self,
        retriever: AbstractRetriever,
        encoder: AbstractEncoder,
        reranker: Optional[AbstractReranker] = None,
        cfg: Optional[AugmenterConfig] = None,
    ):
        if retriever is None or encoder is None:
            raise ValueError("retriever and encoder are required")
        self.retriever = retriever
        self.encoder = encoder
        self.reranker = reranker
        self.cfg = cfg or AugmenterConfig()
        self._cache: Dict[str, Vec] = {}
        self._encode_semaphore = asyncio.Semaphore(self.cfg.concurrency)

    def _extract_snippets_from_doc(self, doc: Doc) -> List[Snippet]:
        snippets: List[Snippet] = []
        # If doc already contains 'chunks' (pre-chunked), use them
        if isinstance(doc.get("chunks"), list) and doc["chunks"]:
            for c in doc["chunks"][: self.cfg.max_snippets_per_doc]:
                text = c.get("text", "")
                if not isinstance(text, str) or not text.strip():
                    continue
                snippets.append(Snippet(
                    text=text.strip(),
                    doc_id=doc.get("id") or str(uuid.uuid4()),
                    source=doc.get("source") or doc.get("meta", {}).get("source", "unknown"),
                    retrieval_score=float(doc.get("score", 0.0))
                ))
            return snippets
        # Otherwise use doc text as a single chunk
        text = extract_text(doc)
        if text:
            snippets.append(Snippet(
                text=text,
                doc_id=doc.get("id") or str(uuid.uuid4()),
                source=doc.get("source") or doc.get("meta", {}).get("source", "unknown"),
                retrieval_score=float(doc.get("score", 0.0))
            ))
        return snippets

    async def _encode_texts(self, texts: List[str]) -> List[Vec]:
        async with self._encode_semaphore:
            return await safe_encode(self.encoder, texts, timeout=max(0.1, self.cfg.timeout_s * 0.5))

    def _diversity_filter(self, snippets: List[Snippet], max_items: int) -> List[Snippet]:
        if not snippets:
            return []
        selected: List[Snippet] = []
        for s in snippets:
            if not selected:
                selected.append(s)
                if len(selected) >= max_items:
                    break
                continue
            if s.embedding is None:
                selected.append(s)
            else:
                sims = [ _cosine_similarity(s.embedding, t.embedding) for t in selected if t.embedding is not None ]
                max_sim = max(sims) if sims else 0.0
                score = (s.rerank_score or s.retrieval_score) - self.cfg.diversity_penalty * max_sim
                if score >= self.cfg.min_snippet_score:
                    selected.append(s)
            if len(selected) >= max_items:
                break
        return selected

    async def augment(self, query: str, cfg: Optional[AugmenterConfig] = None) -> AugmentedContext:
        cfg = cfg or self.cfg
        start_time = time.time()
        retrieval_time = 0.0
        try:
            t0 = time.time()
            docs = await asyncio.wait_for(self.retriever.retrieve(query, top_k=cfg.top_k), timeout=max(0.1, cfg.timeout_s * 0.5))
            retrieval_time = time.time() - t0

            if self.reranker and cfg.use_cross_encoder:
                try:
                    docs = await asyncio.wait_for(self.reranker.rerank(query, docs, top_n=cfg.cross_encoder_top_k), timeout=max(0.1, cfg.timeout_s * 0.25))
                except asyncio.TimeoutError:
                    _logger.warning("Document-level cross-encoder timed out; continuing")

            docs = docs[: cfg.top_k]

            all_snippets: List[Snippet] = []
            for d in docs:
                try:
                    all_snippets.extend(self._extract_snippets_from_doc(d))
                except Exception:
                    _logger.exception("Failed to extract snippets from doc: %s", d.get("id"))

            if not all_snippets:
                proc_time = time.time() - start_time
                return AugmentedContext(query=query, snippets=[], total_tokens=0, retrieval_time_s=retrieval_time, processing_time_s=proc_time)

            texts_to_encode: List[str] = []
            map_idx: List[int] = []
            for idx, s in enumerate(all_snippets):
                if cfg.enable_caching and s.text in self._cache:
                    s.embedding = self._cache[s.text]
                else:
                    map_idx.append(idx)
                    texts_to_encode.append(s.text)

            embeddings: List[Vec] = []
            if texts_to_encode:
                batch_size = max(1, cfg.concurrency * 2)
                for i in range(0, len(texts_to_encode), batch_size):
                    batch_texts = texts_to_encode[i:i+batch_size]
                    batch_embs = await self._encode_texts(batch_texts)
                    embeddings.extend(batch_embs)
                emb_i = 0
                for mi in map_idx:
                    emb = embeddings[emb_i]
                    all_snippets[mi].embedding = emb
                    if cfg.enable_caching:
                        self._cache[all_snippets[mi].text] = emb
                    emb_i += 1

            if self.reranker and cfg.use_cross_encoder:
                try:
                    snippet_docs = [{"id": s.doc_id, "text": s.text, "source": s.source, "score": s.retrieval_score} for s in all_snippets]
                    reranked = await asyncio.wait_for(self.reranker.rerank(query, snippet_docs, top_n=cfg.rerank_top_n * 2), timeout=max(0.1, cfg.timeout_s * 0.25))
                    id_to_score = {d.get("id"): float(d.get("score", 0.0)) for d in reranked}
                    for s in all_snippets:
                        s.rerank_score = id_to_score.get(s.doc_id, s.retrieval_score)
                except asyncio.TimeoutError:
                    _logger.warning("Snippet-level cross-encoder timed out; using retrieval scores")
                except Exception:
                    _logger.exception("Snippet-level reranker failed; using retrieval scores")

            query_embedding = None
            try:
                q_embs = await safe_encode(self.encoder, [query], timeout=max(0.1, cfg.timeout_s * 0.25))
                if q_embs:
                    query_embedding = q_embs[0]
            except Exception:
                _logger.debug("Query embedding failed; skipping semantic scoring")

            for s in all_snippets:
                base = s.rerank_score or s.retrieval_score or 0.0
                sem = 0.0
                if query_embedding is not None and s.embedding is not None:
                    sem = _cosine_similarity(query_embedding, s.embedding)
                s.rerank_score = 0.6 * base + 0.4 * sem

            all_snippets.sort(key=lambda x: x.rerank_score or x.retrieval_score, reverse=True)
            candidates = all_snippets[: max(1, cfg.rerank_top_n * 2)]

            if any(s.embedding is not None for s in candidates):
                final = self._diversity_filter(candidates, cfg.rerank_top_n)
            else:
                final = candidates[: cfg.rerank_top_n]

            final = [s for s in final if (s.rerank_score or s.retrieval_score) >= cfg.min_snippet_score]
            final = final[: cfg.rerank_top_n]

            proc_time = time.time() - start_time
            total_tokens = sum(_token_estimate(s.text) for s in final)
            return AugmentedContext(query=query, snippets=final, total_tokens=total_tokens, retrieval_time_s=retrieval_time, processing_time_s=proc_time, is_timeout=False)

        except asyncio.TimeoutError:
            _logger.warning("Augmentation timed out for query=%s", query)
            proc_time = time.time() - start_time
            return AugmentedContext(query=query, snippets=[], total_tokens=0, retrieval_time_s=retrieval_time, processing_time_s=proc_time, is_timeout=True)
        except Exception as e:
            _logger.exception("Unhandled error in augmentation pipeline: %s", e)
            proc_time = time.time() - start_time
            return AugmentedContext(query=query, snippets=[], total_tokens=0, retrieval_time_s=retrieval_time, processing_time_s=proc_time, is_timeout=False)