# rag/orchestrator.py
from __future__ import annotations
import asyncio
import time
import uuid
import json
import logging
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

# Optional Redis support; fallback to in-memory if not installed/available
try:
    import aioredis 
    _HAS_REDIS = True
except Exception:
    aioredis = None
    _HAS_REDIS = False

from rag.utils.ingestor import Ingestor, IngestConfig
from rag.utils.text_builder import TextBuilder, TBConfig
# Note: embedder, indexer, retriever, reranker, generator should implement expected interfaces.
# They can be passed into Orchestrator on init.

_logger = logging.getLogger("rag.orchestrator")
_logger.addHandler(logging.NullHandler())

# -------------------------
# Models for API
# -------------------------
class JobRequest(BaseModel):
    paths: Optional[List[str]] = None
    agent: str = "general"
    priority: Optional[str] = "normal"
    params: Optional[Dict[str, Any]] = None

class JobStatus(BaseModel):
    request_id: str
    status: str
    created_at: float
    updated_at: float
    steps: List[Dict[str, Any]] = []
    diagnostics: Dict[str, Any] = {}

# -------------------------
# Config and state types
# -------------------------
@dataclass
class OrchestratorConfig:
    redis_url: Optional[str] = None
    queue_key: str = "rag:jobs"
    state_prefix: str = "rag:job:"
    max_workers: int = 4
    poll_interval_s: float = 0.5
    default_collection: str = "default"
    enable_redis: bool = False

# -------------------------
# Simple in-memory queue + state store (fallback)
# -------------------------
class InMemoryQueue:
    def __init__(self):
        self._q: asyncio.Queue = asyncio.Queue()

    async def push(self, item: Dict[str, Any]) -> None:
        await self._q.put(item)

    async def pop(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        try:
            if timeout:
                return await asyncio.wait_for(self._q.get(), timeout=timeout)
            return await self._q.get()
        except asyncio.TimeoutError:
            return None

    def qsize(self) -> int:
        return self._q.qsize()

class InMemoryState:
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def set(self, key: str, value: Dict[str, Any]) -> None:
        async with self._lock:
            self._store[key] = value

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            return self._store.get(key)

    async def update(self, key: str, patch: Dict[str, Any]) -> None:
        async with self._lock:
            cur = self._store.get(key, {})
            cur.update(patch)
            self._store[key] = cur

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

# -------------------------
# Orchestrator
# -------------------------
class Orchestrator:
    def __init__(
        self,
        ingestor: Ingestor,
        text_builder: TextBuilder,
        embedder: Any,
        indexer: Any,
        retriever: Any,
        reranker: Any,
        generator: Any,
        cfg: Optional[OrchestratorConfig] = None,
    ):
        self.ingestor = ingestor
        self.text_builder = text_builder
        self.embedder = embedder
        self.indexer = indexer
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.cfg = cfg or OrchestratorConfig()
        self._running = False
        self._workers: List[asyncio.Task] = []
        # queue and state store (redis if enabled)
        self._redis = None
        if self.cfg.enable_redis and _HAS_REDIS and self.cfg.redis_url:
            self._redis = None  # will be initialized in async init
        self._queue = InMemoryQueue()
        self._state = InMemoryState()
        # concurrency semaphore for workers
        self._sem = asyncio.Semaphore(self.cfg.max_workers)

    async def init_redis(self) -> None:
        if not (self.cfg.enable_redis and _HAS_REDIS and self.cfg.redis_url):
            return
        try:
            self._redis = await aioredis.from_url(self.cfg.redis_url)
        except Exception:
            _logger.exception("failed to connect to redis; falling back to in-memory")
            self._redis = None

    async def start(self) -> None:
        await self.init_redis()
        if self._running:
            return
        self._running = True
        for _ in range(self.cfg.max_workers):
            t = asyncio.create_task(self._worker_loop())
            self._workers.append(t)
        _logger.info("Orchestrator started with %d workers", self.cfg.max_workers)

    async def stop(self) -> None:
        self._running = False
        for t in self._workers:
            t.cancel()
        self._workers = []
        if self._redis:
            try:
                await self._redis.close()
            except Exception:
                pass
        _logger.info("Orchestrator stopped")

    # -------------------------
    # Job lifecycle helpers
    # -------------------------
    async def submit_job(self, paths: Optional[List[str]], agent: str = "general", priority: str = "normal", params: Optional[Dict[str, Any]] = None) -> str:
        request_id = f"job_{uuid.uuid4().hex[:12]}"
        now = time.time()
        job = {
            "request_id": request_id,
            "paths": paths or [],
            "agent": agent,
            "priority": priority,
            "params": params or {},
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "steps": [],
            "diagnostics": {},
        }
        # persist state
        await self._set_state(request_id, job)
        # push to queue
        await self._push_queue(job)
        _logger.info("submitted job %s agent=%s paths=%d", request_id, agent, len(job["paths"]))
        return request_id

    async def get_job(self, request_id: str) -> Optional[Dict[str, Any]]:
        return await self._get_state(request_id)

    async def retry_job(self, request_id: str) -> bool:
        job = await self._get_state(request_id)
        if not job:
            return False
        # reset status and push back to queue
        job["status"] = "queued"
        job["updated_at"] = time.time()
        job["diagnostics"].setdefault("retries", job["diagnostics"].get("retries", 0) + 1)
        await self._set_state(request_id, job)
        await self._push_queue(job)
        return True

    # -------------------------
    # Queue / state abstraction
    # -------------------------
    async def _push_queue(self, job: Dict[str, Any]) -> None:
        if self._redis:
            try:
                await self._redis.lpush(self.cfg.queue_key, json.dumps(job))
                return
            except Exception:
                _logger.exception("redis push failed; falling back to in-memory")
        await self._queue.push(job)

    async def _pop_queue(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        if self._redis:
            try:
                # BRPOP returns (key, value) or None
                res = await self._redis.brpop(self.cfg.queue_key, timeout=timeout or 1)
                if res:
                    _, raw = res
                    return json.loads(raw)
                return None
            except Exception:
                _logger.exception("redis pop failed; falling back to in-memory")
        return await self._queue.pop(timeout=timeout)

    async def _set_state(self, request_id: str, state: Dict[str, Any]) -> None:
        key = self.cfg.state_prefix + request_id
        if self._redis:
            try:
                await self._redis.set(key, json.dumps(state))
                return
            except Exception:
                _logger.exception("redis set failed; falling back to in-memory")
        await self._state.set(key, state)

    async def _get_state(self, request_id: str) -> Optional[Dict[str, Any]]:
        key = self.cfg.state_prefix + request_id
        if self._redis:
            try:
                raw = await self._redis.get(key)
                if raw:
                    return json.loads(raw)
                return None
            except Exception:
                _logger.exception("redis get failed; falling back to in-memory")
        return await self._state.get(key)

    async def _update_state(self, request_id: str, patch: Dict[str, Any]) -> None:
        key = self.cfg.state_prefix + request_id
        if self._redis:
            try:
                raw = await self._redis.get(key)
                if not raw:
                    return
                state = json.loads(raw)
                state.update(patch)
                await self._redis.set(key, json.dumps(state))
                return
            except Exception:
                _logger.exception("redis update failed; falling back to in-memory")
        await self._state.update(key, patch)

    # -------------------------
    # Worker loop
    # -------------------------
    async def _worker_loop(self) -> None:
        while self._running:
            try:
                job = await self._pop_queue(timeout=1.0)
                if not job:
                    await asyncio.sleep(self.cfg.poll_interval_s)
                    continue
                # process job with concurrency control
                async with self._sem:
                    await self._process_job(job)
            except asyncio.CancelledError:
                break
            except Exception:
                _logger.exception("worker loop error")
                await asyncio.sleep(1.0)

    async def _process_job(self, job: Dict[str, Any]) -> None:
        request_id = job.get("request_id")
        if not request_id:
            _logger.warning("skipping job without request_id")
            return
        # update state to running
        now = time.time()
        job_state = await self._get_state(request_id) or job
        job_state.update({"status": "running", "updated_at": now})
        await self._set_state(request_id, job_state)

        try:
            # Step 1: ingest files (paths)
            step_name = "ingest"
            step_info = {"name": step_name, "status": "started", "started_at": time.time()}
            job_state["steps"].append(step_info)
            await self._set_state(request_id, job_state)

            paths = job.get("paths", []) or []
            # ingest_folder returns summary; use ingestor.ingest_folder
            ingest_res = await self.ingestor.ingest_folder(paths)
            step_info.update({"status": "ok", "finished_at": time.time(), "result": ingest_res})
            await self._set_state(request_id, job_state)

            # Step 2: build records and optionally embed & index
            step_name = "build_and_index"
            step_info = {"name": step_name, "status": "started", "started_at": time.time()}
            job_state["steps"].append(step_info)
            await self._set_state(request_id, job_state)

            # For simplicity: iterate files results and process raw_docs via loader/ingestor pipeline
            # Here we assume ingestor produced DLQ and diagnostics; we re-run a simple flow:
            # - For each raw_doc from ingestor, call text_builder.build_records
            # - For each record, call embedder.encode and indexer.upsert
            # Note: embedder and indexer may be sync or async; handle both.

            # Collect raw_docs by reading ingestor results diagnostics if available
            # Fallback: re-run ingestor._read for each path (not ideal but safe)
            raw_docs_all: List[Dict[str, Any]] = []
            # ingest_res["results"] contains per-path results; but ingestor does not return raw_docs by default.
            # To keep orchestrator generic, call ingestor.load_and_dispatch with a dispatcher that collects raw_docs.
            async def collector_dispatcher(raw: Dict[str, Any]) -> None:
                raw_docs_all.append(raw)

            # Use ingestor.load_and_dispatch to stream raw_docs
            await self.ingestor.load_and_dispatch(paths, dispatcher=collector_dispatcher)

            # Build records in batches
            batch_size = 32
            records_to_index: List[Dict[str, Any]] = []
            for i in range(0, len(raw_docs_all), batch_size):
                batch = raw_docs_all[i : i + batch_size]
                # text_builder.process_batch is async
                recs = await self.text_builder.process_batch(batch)
                for r in recs:
                    # attach collection and agent metadata
                    r_meta = r.get("meta", {})
                    r_meta.setdefault("collection", job.get("params", {}).get("collection", self.cfg.default_collection))
                    r_meta.setdefault("agent", job.get("agent"))
                    r["meta"] = r_meta
                    records_to_index.append(r)

            # Step: embed and upsert in chunks
            embed_batch = 64
            for i in range(0, len(records_to_index), embed_batch):
                chunk = records_to_index[i : i + embed_batch]
                texts = [r["text"] for r in chunk]
                # embedder.encode may be sync or async and may accept list
                try:
                    if hasattr(self.embedder, "encode_async"):
                        vecs = await self.embedder.encode_async(texts)
                    else:
                        res = self.embedder.encode(texts)
                        if asyncio.iscoroutine(res):
                            vecs = await res
                        else:
                            vecs = res
                except Exception:
                    _logger.exception("embedder failed for batch; skipping embeddings")
                    vecs = [[] for _ in texts]

                # prepare upsert payloads
                ids = [r["chunk_id"] for r in chunk]
                metas = [r for r in chunk]
                # indexer.upsert may be sync or async
                try:
                    res = self.indexer.upsert(collection=self.cfg.default_collection, ids=ids, vectors=vecs, metas=metas)
                    if asyncio.iscoroutine(res):
                        await res
                except Exception:
                    _logger.exception("indexer.upsert failed for batch; adding to diagnostics")
                    job_state.setdefault("diagnostics", {}).setdefault("index_errors", 0)
                    job_state["diagnostics"]["index_errors"] += 1

            step_info.update({"status": "ok", "finished_at": time.time(), "indexed": len(records_to_index)})
            await self._set_state(request_id, job_state)

            # Step 3: optional post-processing (e.g., warm retriever, precompute rerank features)
            step_name = "postprocess"
            step_info = {"name": step_name, "status": "started", "started_at": time.time()}
            job_state["steps"].append(step_info)
            await self._set_state(request_id, job_state)

            # Example: warm retriever by issuing a dummy query per agent (non-blocking)
            try:
                if hasattr(self.retriever, "retrieve_async"):
                    await self.retriever.retrieve_async("pré-chauffage", top_k=1)
            except Exception:
                _logger.debug("retriever warmup failed (non-fatal)")

            step_info.update({"status": "ok", "finished_at": time.time()})
            await self._set_state(request_id, job_state)

            # Mark job completed
            job_state.update({"status": "ok", "updated_at": time.time()})
            await self._set_state(request_id, job_state)
            _logger.info("job %s completed ok", request_id)

        except Exception as e:
            _logger.exception("job %s failed", request_id)
            job_state.update({"status": "error", "updated_at": time.time()})
            job_state.setdefault("diagnostics", {})["error"] = str(e)
            await self._set_state(request_id, job_state)

    # -------------------------
    # Convenience helpers for external control
    # -------------------------
    async def health(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "workers": len(self._workers),
            "queue_size": (await self._queue.qsize()) if hasattr(self._queue, "qsize") else None,
        }

# -------------------------
# FastAPI wrapper
# -------------------------
def create_app(orchestrator: Orchestrator) -> FastAPI:
    app = FastAPI(title="RAG Orchestrator", version="0.1.0")

    @app.on_event("startup")
    async def startup():
        await orchestrator.start()

    @app.on_event("shutdown")
    async def shutdown():
        await orchestrator.stop()

    @app.post("/jobs", response_model=Dict[str, str])
    async def submit_job(req: JobRequest):
        request_id = await orchestrator.submit_job(paths=req.paths, agent=req.agent, priority=req.priority, params=req.params)
        return {"request_id": request_id}

    @app.get("/jobs/{request_id}", response_model=JobStatus)
    async def get_job(request_id: str):
        state = await orchestrator.get_job(request_id)
        if not state:
            raise HTTPException(status_code=404, detail="job not found")
        return JobStatus(**state)

    @app.post("/jobs/{request_id}/retry", response_model=Dict[str, Any])
    async def retry_job(request_id: str):
        ok = await orchestrator.retry_job(request_id)
        if not ok:
            raise HTTPException(status_code=404, detail="job not found")
        return {"request_id": request_id, "status": "requeued"}

    @app.get("/health", response_model=Dict[str, Any])
    async def health():
        return await orchestrator.health()

    return app

# -------------------------
# Example wiring (to be adapted in real deployment)
# -------------------------
# The following is an example of how to instantiate the orchestrator with simple components.
# Replace Dummy* classes with real implementations.

class DummyEmbedder:
    def __init__(self, dim: int = 128):
        self.dim = dim
    def encode(self, texts: List[str]) -> List[List[float]]:
        # naive deterministic embedding (not useful for production)
        return [[float(len(t) % 10)] * self.dim for t in texts]

class DummyIndexer:
    def __init__(self):
        self._store = {}
    def upsert(self, collection: str, ids: List[str], vectors: List[List[float]], metas: List[Dict[str, Any]]):
        for i, id_ in enumerate(ids):
            self._store[id_] = {"vector": vectors[i] if i < len(vectors) else [], "meta": metas[i]}
        return {"inserted": len(ids)}
    async def search(self, collection: str, query_vecs: List[List[float]], top_k: int, expr: Optional[str] = None, params: Optional[Dict] = None):
        return [[]]
    async def mget(self, collection: str, ids: List[str]):
        return [self._store.get(i, {"id": i}) for i in ids]

class DummyRetriever:
    async def retrieve_async(self, query: str, top_k: int = 10):
        return []

class DummyReranker:
    pass

class DummyGenerator:
    pass

# If run as main, create a simple app using dummy components
if __name__ == "__main__":
    import uvicorn
    # instantiate components
    tb = TextBuilder(TBConfig())
    ing_cfg = IngestConfig()
    ing = Ingestor(text_builder=tb, indexer_adapter=None, cfg=ing_cfg)
    embed = DummyEmbedder()
    indexer = DummyIndexer()
    retriever = DummyRetriever()
    reranker = DummyReranker()
    generator = DummyGenerator()
    orchestrator_cfg = OrchestratorConfig(enable_redis=False, max_workers=2)
    orch = Orchestrator(ingestor=ing, text_builder=tb, embedder=embed, indexer=indexer, retriever=retriever, reranker=reranker, generator=generator, cfg=orchestrator_cfg)
    app = create_app(orch)
    uvicorn.run(app, host="0.0.0.0", port=8000)