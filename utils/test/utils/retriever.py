# rag/retriever.py
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

_logger = logging.getLogger("rag.retriever")
_logger.addHandler(logging.NullHandler())

# -------------------------
# Configuration dataclass
# -------------------------
@dataclass
class RetrieverConfig:
    top_k: int = 50
    candidate_pool: int = 200
    timeout_s: Optional[float] = 8.0
    # If your indexer supports a distance->score conversion, set this to True
    normalize_scores: bool = True
    # If True, try to convert returned 'distance' to a score via 1/(1+distance)
    convert_distance_to_score: bool = True

# -------------------------
# Retriever implementation
# -------------------------
class Retriever:
    """
    Retriever robuste et asynchrone.

    Usage:
      cfg = RetrieverConfig(top_k=50, candidate_pool=200)
      retr = Retriever(embedder=embedder, indexer=indexer, cfg=cfg, collection="default")
      docs = await retr.retrieve("ma requête", top_k=20)

    Comportement:
    - Si `retrieve_async` est surchargée dans une sous-classe, elle sera utilisée.
    - Sinon, `_retrieve_sync` est exécutée dans un thread via asyncio.to_thread.
    - Les résultats sont normalisés en dicts contenant: id, text, source, score.
    """

    def __init__(self, embedder: Any = None, indexer: Any = None, cfg: Optional[RetrieverConfig] = None, collection: str = "default"):
        self.embedder = embedder
        self.indexer = indexer
        self.cfg = cfg or RetrieverConfig()
        self.collection = collection

    # -------------------------
    # Fallback sync retrieval (adapter minimal)
    # -------------------------
    def _retrieve_sync(self, query: str, top_k: int = 50, expr: Optional[str] = None, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Fallback synchrone si aucun indexer async n'est disponible.
        - Si indexer expose search(collection, query, top_k, ...) on l'utilise.
        - Si indexer expose _docs (dict), on effectue une recherche textuelle simple.
        - Sinon, retourne une liste vide.
        """
        if self.indexer is None:
            _logger.debug("No indexer available for _retrieve_sync")
            return []

        search_fn = getattr(self.indexer, "search", None)
        if callable(search_fn):
            try:
                # Try multiple calling conventions
                try:
                    res = search_fn(collection=self.collection, query=query, top_k=top_k, expr=expr, params=params)
                except TypeError:
                    # fallback to positional
                    res = search_fn(self.collection, query, top_k)
                # If res is coroutine, run it synchronously (rare for sync path)
                if asyncio.iscoroutine(res):
                    res = asyncio.get_event_loop().run_until_complete(res)
                normalized = []
                for h in (res or []):
                    # h may be dict-like or object; try to access common fields
                    if not isinstance(h, dict):
                        # try attribute access
                        try:
                            h = {
                                "id": getattr(h, "id", None),
                                "text": getattr(h, "text", None),
                                "meta": getattr(h, "meta", None),
                                "score": getattr(h, "score", None),
                                "distance": getattr(h, "distance", None),
                            }
                        except Exception:
                            continue
                    normalized.append({
                        "id": h.get("id") or h.get("doc_id") or h.get("pk"),
                        "text": h.get("text") or (h.get("meta") or {}).get("text", "") or "",
                        "source": (h.get("meta") or {}).get("source") or h.get("source") or None,
                        "score": float(h.get("score", 0.0)) if h.get("score") is not None else (float(h.get("distance")) if h.get("distance") is not None else 0.0),
                        "raw": h,
                    })
                return normalized[:top_k]
            except Exception:
                _logger.exception("indexer.search (sync) failed in _retrieve_sync; falling back to _docs if present")

        # Fallback naive search on indexer._docs (if present)
        if hasattr(self.indexer, "_docs"):
            try:
                docs = list(self.indexer._docs.values())
                qtokens = set(str(query).lower().split())
                scored = []
                for d in docs:
                    text = str(d.get("text", "")).lower()
                    score = sum(1 for t in qtokens if t in text)
                    scored.append({
                        "id": d.get("id"),
                        "text": d.get("text"),
                        "source": (d.get("meta") or {}).get("source"),
                        "score": float(score),
                        "raw": d,
                    })
                scored_sorted = sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]
                return scored_sorted
            except Exception:
                _logger.exception("Fallback _docs search failed in _retrieve_sync")

        # Nothing available
        return []

    # -------------------------
    # Async retrieval hook (override in subclasses if you have async indexer)
    # -------------------------
    async def retrieve_async(self, query: str, top_k: int = 50, expr: Optional[str] = None, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Par défaut, exécute la version synchrone dans un thread.
        Si tu as une implémentation asynchrone native (ex: indexer.search async),
        surcharge cette méthode dans ta classe.
        """
        return await asyncio.to_thread(self._retrieve_sync, query, top_k, expr, params)

    # -------------------------
    # Façade asynchrone publique
    # -------------------------
    async def retrieve(self, query: str, top_k: Optional[int] = None, expr: Optional[str] = None, params: Optional[Dict] = None, timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Façade à utiliser partout : await retriever.retrieve(...)
        - top_k: override cfg.top_k if provided
        - timeout: optional per-call timeout in seconds
        """
        top_k = int(top_k or self.cfg.top_k)
        timeout = timeout if timeout is not None else self.cfg.timeout_s

        coro = self.retrieve_async(query, top_k=top_k, expr=expr, params=params)
        try:
            if timeout:
                docs = await asyncio.wait_for(coro, timeout=timeout)
            else:
                docs = await coro
        except asyncio.TimeoutError:
            _logger.warning("Retriever.retrieve timed out for query=%s (top_k=%s)", query, top_k)
            return []
        except Exception as e:
            _logger.exception("Retriever.retrieve failed for query=%s: %s", query, e)
            return []

        # Normalize results: ensure list of dicts with id,text,source,score
        normalized: List[Dict[str, Any]] = []
        for d in (docs or []):
            if not isinstance(d, dict):
                continue
            # If indexer returned a 'distance' and config asks to convert, do it
            score = None
            if d.get("score") is not None:
                try:
                    score = float(d.get("score"))
                except Exception:
                    score = 0.0
            elif d.get("distance") is not None and self.cfg.convert_distance_to_score:
                try:
                    dist = float(d.get("distance"))
                    score = 1.0 / (1.0 + dist)
                except Exception:
                    score = 0.0
            else:
                score = 0.0

            normalized.append({
                "id": d.get("id") or d.get("doc_id") or d.get("chunk_id") or d.get("raw", {}).get("id"),
                "text": d.get("text") or (d.get("meta") or {}).get("text") or "",
                "source": d.get("source") or (d.get("meta") or {}).get("source") or d.get("path") or None,
                "score": float(score),
                "raw": d.get("raw", d),
            })

        # Optionally normalize scores across returned candidates
        if self.cfg.normalize_scores and normalized:
            scores = [float(x.get("score", 0.0)) for x in normalized]
            mn = min(scores)
            mx = max(scores)
            if mx - mn > 1e-12:
                for x in normalized:
                    x["score"] = (float(x.get("score", 0.0)) - mn) / (mx - mn)
            else:
                # if all equal, set to 1.0 if positive else 0.0
                for x in normalized:
                    x["score"] = 1.0 if mx > 0 else 0.0

        # sort by score desc and trim to top_k
        normalized.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return normalized[:top_k]

    # -------------------------
    # Helper: search by vectors (async-friendly)
    # -------------------------
    async def search(self, query_vecs: Sequence[Sequence[float]], top_k: int = 10, expr: Optional[str] = None, params: Optional[Dict] = None) -> List[List[Dict[str, Any]]]:
        """
        Wrapper asynchrone pour indexer.search when indexer supports vector search.
        Returns a list of result lists (one per query vector).
        """
        if self.indexer is None:
            _logger.warning("search called but no indexer available")
            return [[] for _ in query_vecs]

        search_fn = getattr(self.indexer, "search", None)
        if not callable(search_fn):
            _logger.warning("indexer has no search method")
            return [[] for _ in query_vecs]

        try:
            # Try calling with vector param name or positional
            try:
                res = search_fn(collection=self.collection, query_vecs=list(query_vecs), top_k=top_k, expr=expr, params=params)
            except TypeError:
                res = search_fn(list(query_vecs), top_k)
            if asyncio.iscoroutine(res):
                res = await res
            return res
        except Exception:
            _logger.exception("indexer.search failed")
            return [[] for _ in query_vecs]

    # -------------------------
    # Helper: multi-get
    # -------------------------
    async def mget(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Multi-get wrapper for indexer.mget or indexer._docs fallback.
        """
        if self.indexer is None:
            return [{"id": i} for i in ids]

        mget_fn = getattr(self.indexer, "mget", None)
        try:
            if callable(mget_fn):
                res = mget_fn(collection=self.collection, ids=ids)
                if asyncio.iscoroutine(res):
                    return await res
                return res
            if hasattr(self.indexer, "_docs"):
                return [self.indexer._docs.get(i, {"id": i}) for i in ids]
        except Exception:
            _logger.exception("indexer.mget failed")
        return [{"id": i} for i in ids]