# rag/cross_encoder.py
from __future__ import annotations
import asyncio
import time
import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache

_logger = logging.getLogger("rag.cross_encoder")
_logger.addHandler(logging.NullHandler())

# --- Types ---
Doc = Dict[str, Any]
RankedDoc = Dict[str, Any]

# --- Simple LRU cache wrapper for pair scores ---
def _pair_key(query: str, doc_text: str) -> str:
    h = hashlib.sha256()
    h.update(query.encode("utf-8"))
    h.update(b"\x00")
    h.update(doc_text.encode("utf-8"))
    return h.hexdigest()

# --- Model interface stub
class CrossEncoderModel:
    """
    Interface minimal pour un cross-encoder.
    Implémentation concrète doit fournir `predict_batch(pairs: List[Tuple[str,str]]) -> List[float]`.
    """
    def __init__(self, model_name: str = "cross-encoder/stub"):
        self.model_name = model_name
        # placeholder for real model (transformers, torch, etc.)
        self._warm = False

    async def warmup(self) -> None:
        # effectuer un warmup si nécessaire
        await asyncio.sleep(0.01)
        self._warm = True

    def predict_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Synchrone pour simplicité. Retourne un score float par paire.
        Remplacer par inférence réelle (tokenize + model.forward).
        """
        # heuristique stub : score = longueur de l'intersection token
        out: List[float] = []
        for q, d in pairs:
            qset = set(q.lower().split())
            dset = set(d.lower().split())
            inter = len(qset & dset)
            out.append(float(inter))
        return out

# --- CrossEncoder service class ---
class CrossEncoder:
    def __init__(
        self,
        model: Optional[CrossEncoderModel] = None,
        batch_size: int = 32,
        max_batch_latency_s: float = 0.05,
        cache_size: int = 10000,
    ):
        self.model = model or CrossEncoderModel()
        self.batch_size = batch_size
        self.max_batch_latency_s = max_batch_latency_s
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._closed = False
        # LRU cache for pair scores
        self._score_cache = lru_cache(maxsize=cache_size)(self._compute_score_uncached)
        # ensure warmup
        self._warm_lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._warm_lock:
            if not self._worker_task:
                await self.model.warmup()
                self._worker_task = asyncio.create_task(self._batch_worker())
                _logger.info("CrossEncoder worker started")

    async def stop(self) -> None:
        self._closed = True
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
            _logger.info("CrossEncoder worker stopped")

    # Public rerank sync wrapper
    def rerank(self, query: str, candidates: List[Doc], top_n: Optional[int] = None) -> List[RankedDoc]:
        """
        Synchrone wrapper : utilise predict_batch directement (blocking).
        Utile pour tests ou environnements non-async.
        """
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict_batch(pairs)
        for c, s in zip(candidates, scores):
            c["cross_score"] = float(s)
        ranked = sorted(candidates, key=lambda d: d["cross_score"], reverse=True)
        if top_n:
            ranked = ranked[:top_n]
        return ranked

    # Public async rerank using batching worker
    async def rerank_async(self, query: str, candidates: List[Doc], top_n: Optional[int] = None, timeout: float = 1.0) -> List[RankedDoc]:
        """
        Asynchrone : soumet chaque paire au worker et attend les scores.
        Retourne la liste rerankée.
        """
        # prepare futures for each candidate
        futures: List[asyncio.Future] = []
        for c in candidates:
            fut = asyncio.get_event_loop().create_future()
            await self._queue.put((query, c["text"], fut))
            futures.append(fut)

        # wait for all futures with timeout
        try:
            scores = await asyncio.wait_for(asyncio.gather(*futures), timeout=timeout)
        except asyncio.TimeoutError:
            # fallback : compute synchronously for remaining
            _logger.warning("rerank_async timeout, falling back to sync compute for pending items")
            scores = []
            for fut, c in zip(futures, candidates):
                if fut.done():
                    scores.append(fut.result())
                else:
                    s = self._compute_score_uncached(query, c["text"])
                    scores.append(s)

        for c, s in zip(candidates, scores):
            c["cross_score"] = float(s)
        ranked = sorted(candidates, key=lambda d: d["cross_score"], reverse=True)
        if top_n:
            ranked = ranked[:top_n]
        return ranked

    # Internal worker that batches requests
    async def _batch_worker(self) -> None:
        """
        Regroupe les requêtes en batch et appelle le modèle.
        Chaque item dans queue est (query, doc_text, future)
        """
        try:
            while not self._closed:
                batch: List[Tuple[str, str, asyncio.Future]] = []
                start = time.time()
                # always wait for at least one item
                item = await self._queue.get()
                batch.append(item)
                # gather up to batch_size or until latency exceeded
                while len(batch) < self.batch_size and (time.time() - start) < self.max_batch_latency_s:
                    try:
                        item = self._queue.get_nowait()
                        batch.append(item)
                    except asyncio.QueueEmpty:
                        await asyncio.sleep(0)  # yield
                # prepare pairs and call model
                pairs = [(q, d) for q, d, _ in batch]
                try:
                    scores = await asyncio.get_event_loop().run_in_executor(None, self.model.predict_batch, pairs)
                except Exception as e:
                    _logger.exception("CrossEncoder model predict failed")
                    scores = [0.0] * len(pairs)
                # set futures and cache
                for (q, d, fut), s in zip(batch, scores):
                    try:
                        # update cache via wrapper
                        key = _pair_key(q, d)
                        # populate lru cache by calling cached function
                        try:
                            self._score_cache(q, d)  # may compute and cache
                        except Exception:
                            pass
                        if not fut.done():
                            fut.set_result(float(s))
                    except Exception:
                        if not fut.done():
                            fut.set_result(0.0)
        except asyncio.CancelledError:
            _logger.info("CrossEncoder worker cancelled")
        except Exception:
            _logger.exception("CrossEncoder worker crashed")

    # Uncached compute used by lru_cache wrapper
    def _compute_score_uncached(self, query: str, doc_text: str) -> float:
        try:
            s = self.model.predict_batch([(query, doc_text)])[0]
            return float(s)
        except Exception:
            return 0.0

    # Convenience helper to compute single score (uses cache)
    def score(self, query: str, doc_text: str) -> float:
        return self._score_cache(query, doc_text)