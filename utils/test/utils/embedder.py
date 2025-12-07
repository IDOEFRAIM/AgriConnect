# rag/embedder.py
from __future__ import annotations
import asyncio
import time
import logging
import hashlib
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from functools import lru_cache, partial
from dataclasses import dataclass, field
import numpy as np
import concurrent.futures

_logger = logging.getLogger("rag.embedder")
_logger.addHandler(logging.NullHandler())

# --- Types ---
Vector = List[float]
DocText = str

@dataclass
class EmbedderConfig:
    batch_size: int = 64
    max_batch_latency_s: float = 0.02
    normalize: bool = True  # L2 normalize vectors
    dtype: str = "float32"  # float32 or float16
    cache_size: int = 10000
    use_async_worker: bool = True
    max_workers: int = 4  # for run_in_executor fallback
    device: str = "cpu"  # "cpu" or "cuda"
    quantized: bool = False  # hook for quantized models

class EmbedderModel:
    """
    Minimal model interface to plug real models.
    Implement predict_batch(self, texts: List[str]) -> List[List[float]]
    """
    def __init__(self, dim: int = 768):
        self.dim = dim

    def predict_batch(self, texts: List[str]) -> List[List[float]]:
        # Dummy deterministic embedding: hash-based pseudo-vector for tests
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            vec = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
            # reduce/expand to dim
            if vec.size >= self.dim:
                vec = vec[: self.dim]
            else:
                vec = np.pad(vec, (0, self.dim - vec.size), mode="wrap")
            # normalize to [0,1]
            vec = vec / (np.linalg.norm(vec) + 1e-12)
            out.append(vec.astype(np.float32).tolist())
        return out

class Embedder:
    def __init__(self, model: Optional[EmbedderModel] = None, cfg: Optional[EmbedderConfig] = None):
        self.cfg = cfg or EmbedderConfig()
        self.model = model or EmbedderModel(dim=768)
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._closed = False
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.cfg.max_workers)
        # LRU cache wrapper for encode results
        self._cache = lru_cache(maxsize=self.cfg.cache_size)(self._encode_uncached)
        # metrics
        self.metrics: Dict[str, Any] = {"calls": 0, "cache_hits": 0, "encode_time_s": 0.0}

    # Public sync encode (blocking)
    def encode(self, texts: List[DocText]) -> List[Vector]:
        self.metrics["calls"] += 1
        # use cache per text
        results: List[Vector] = []
        for t in texts:
            key = self._text_key(t)
            try:
                vec = self._cache(key, t)
                self.metrics["cache_hits"] += 1
            except Exception:
                vec = self._encode_uncached(key, t)
            results.append(vec)
        return results

    # Public async encode using worker batching
    async def encode_async(self, texts: List[DocText], timeout: Optional[float] = None) -> List[Vector]:
        """
        Submits texts to internal queue and waits for results.
        If use_async_worker is False, runs encode in executor.
        """
        self.metrics["calls"] += 1
        if not self.cfg.use_async_worker:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self._executor, partial(self.encode, texts))

        # ensure worker started
        if not self._worker_task:
            self._worker_task = asyncio.create_task(self._batch_worker())

        futures: List[asyncio.Future] = []
        for t in texts:
            fut = asyncio.get_event_loop().create_future()
            await self._queue.put((t, fut))
            futures.append(fut)

        try:
            results = await asyncio.wait_for(asyncio.gather(*futures), timeout=timeout)
        except asyncio.TimeoutError:
            _logger.warning("embedder.encode_async timeout, falling back to sync for pending items")
            results = []
            for fut, t in zip(futures, texts):
                if fut.done():
                    results.append(fut.result())
                else:
                    results.append(self.encode([t])[0])
        return results

    # Worker that batches requests
    async def _batch_worker(self) -> None:
        try:
            while not self._closed:
                batch: List[Tuple[DocText, asyncio.Future]] = []
                start = time.time()
                # wait for first item
                item = await self._queue.get()
                batch.append(item)
                # collect until batch_size or latency exceeded
                while len(batch) < self.cfg.batch_size and (time.time() - start) < self.cfg.max_batch_latency_s:
                    try:
                        item = self._queue.get_nowait()
                        batch.append(item)
                    except asyncio.QueueEmpty:
                        await asyncio.sleep(0)  # yield
                texts = [t for t, _ in batch]
                # compute embeddings in executor to avoid blocking loop
                loop = asyncio.get_event_loop()
                t0 = time.time()
                try:
                    vecs = await loop.run_in_executor(self._executor, partial(self.encode, texts))
                except Exception:
                    _logger.exception("embedder model failed in executor")
                    vecs = [[0.0] * self.model.dim for _ in texts]
                dt = time.time() - t0
                self.metrics["encode_time_s"] += dt
                # set futures
                for (_, fut), v in zip(batch, vecs):
                    if not fut.done():
                        fut.set_result(v)
        except asyncio.CancelledError:
            _logger.info("embedder worker cancelled")
        except Exception:
            _logger.exception("embedder worker crashed")

    # Uncached encode used by lru_cache wrapper
    def _encode_uncached(self, key: str, text: str) -> Vector:
        # call model.predict_batch synchronously for single item
        try:
            vec = self.model.predict_batch([text])[0]
            arr = np.asarray(vec, dtype=self.cfg.dtype)
            if self.cfg.normalize:
                norm = np.linalg.norm(arr) + 1e-12
                arr = arr / norm
            return arr.astype(np.float32).tolist()
        except Exception:
            _logger.exception("encode_uncached failed")
            # fallback zero vector
            return [0.0] * getattr(self.model, "dim", 768)

    def _text_key(self, text: str) -> str:
        # stable key for caching; can include device/quant flags
        h = hashlib.sha256()
        h.update(text.encode("utf-8"))
        h.update(self.cfg.device.encode("utf-8"))
        h.update(b"q" if self.cfg.quantized else b"nq")
        return h.hexdigest()

    # Convenience stop
    async def stop(self) -> None:
        self._closed = True
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        self._executor.shutdown(wait=False)

    # Helper to get metrics snapshot
    def metrics_snapshot(self) -> Dict[str, Any]:
        return dict(self.metrics)