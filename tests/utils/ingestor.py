# rag/ingestor.py
from __future__ import annotations
import asyncio
import time
import uuid
import json
import csv
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import aiofiles
import aiohttp

_logger = logging.getLogger("rag.ingestor")
_logger.addHandler(logging.NullHandler())

@dataclass
class IngestConfig:
    max_concurrency: int = 4
    retry_attempts: int = 3
    retry_backoff_s: float = 0.5
    dlq_path: Optional[str] = None
    collection: str = "default"
    pdf_executor_workers: int = 2

class IngestResult(Dict[str, Any]):
    pass

class Ingestor:
    """
    Ingestor responsable de:
    - lire fichiers JSON/CSV/TXT/PDF
    - extraire texte et métadonnées
    - appeler TextBuilder.build_records(raw_doc)
    - dispatcher records vers indexer_adapter.upsert or queue
    - gérer retries, backoff, DLQ, diagnostics
    """

    def __init__(self, text_builder, indexer_adapter=None, cfg: Optional[IngestConfig] = None):
        self.text_builder = text_builder
        self.indexer = indexer_adapter
        self.cfg = cfg or IngestConfig()
        self._sem = asyncio.Semaphore(self.cfg.max_concurrency)
        self._dlq: List[Dict[str, Any]] = []
        self._metrics: Dict[str, Any] = {"ingested": 0, "errors": 0, "chunks": 0, "latencies": []}
        # executor for blocking tasks like PDF extraction
        self._executor = asyncio.ThreadPoolExecutor(max_workers=self.cfg.pdf_executor_workers)

    async def ingest_folder(self, paths: Iterable[str]) -> Dict[str, Any]:
        tasks = [asyncio.create_task(self._ingest_path_safe(p)) for p in paths]
        results = await asyncio.gather(*tasks)
        return {"results": results, "dlq_len": len(self._dlq), "metrics": dict(self._metrics)}

    async def _ingest_path_safe(self, path: str) -> IngestResult:
        async with self._sem:
            try:
                return await self._ingest_path(path)
            except Exception as e:
                _logger.exception("ingest_path failed %s", path)
                self._metrics["errors"] += 1
                err = {"path": path, "status": "error", "diagnostics": {"error": str(e)}}
                self._dlq.append(err)
                await self._maybe_persist_dlq()
                return err

    async def _ingest_path(self, path: str) -> IngestResult:
        start = time.time()
        raw_docs: List[Dict[str, Any]] = []
        try:
            lower = path.lower()
            if lower.endswith(".json"):
                raw_docs = await self._read_json(path)
            elif lower.endswith(".csv"):
                raw_docs = await self._read_csv(path)
            elif lower.endswith(".pdf"):
                raw_docs = [await self._read_pdf(path)]
            else:
                raw_docs = [await self._read_text(path)]
        except Exception as e:
            _logger.exception("failed to read path %s", path)
            self._metrics["errors"] += 1
            self._dlq.append({"path": path, "error": "read_failed", "exc": str(e)})
            await self._maybe_persist_dlq()
            return {"path": path, "status": "error", "diagnostics": {"error": str(e)}}

        emitted = 0
        diagnostics: Dict[str, Any] = {"per_raw": []}
        for raw in raw_docs:
            raw_doc = {
                "raw_id": raw.get("raw_id") or str(uuid.uuid4()),
                "source": path,
                "text": raw.get("text", "") or "",
                "meta": raw.get("meta", {"collection": self.cfg.collection})
            }
            try:
                # build_records is sync and pure; safe to call directly
                records = list(self.text_builder.build_records(raw_doc))
                rec_count = 0
                for rec in records:
                    rec_count += 1
                    emitted += 1
                    self._metrics["chunks"] += 1
                    # dispatch to indexer if provided; indexer.upsert may be async
                    if self.indexer:
                        await self._safe_upsert(rec)
                diagnostics["per_raw"].append({"raw_id": raw_doc["raw_id"], "records": rec_count})
            except Exception as e:
                _logger.exception("text_builder failed for raw %s", raw_doc.get("raw_id"))
                self._metrics["errors"] += 1
                self._dlq.append({"raw": raw_doc, "error": "build_failed", "exc": str(e)})
                await self._maybe_persist_dlq()
                diagnostics["per_raw"].append({"raw_id": raw_doc.get("raw_id"), "error": str(e)})

        elapsed = time.time() - start
        self._metrics["ingested"] += 1
        self._metrics["latencies"].append(elapsed)
        return {"path": path, "status": "ok", "records_emitted": emitted, "time_s": elapsed, "diagnostics": diagnostics}

    async def _safe_upsert(self, rec: Dict[str, Any]) -> None:
        """
        Upsert wrapper with retries and backoff.
        The indexer adapter is expected to implement async upsert(collection, ids, vectors, metas)
        Here we upsert metadata only; vectors will be computed later by embedder pipeline.
        """
        if not self.indexer:
            return
        attempts = 0
        last_exc = None
        collection = rec.get("meta", {}).get("collection", self.cfg.collection)
        ids = [rec["chunk_id"]]
        metas = [rec]
        vectors: List[List[float]] = []  # placeholder: embedding happens later
        while attempts < self.cfg.retry_attempts:
            try:
                # indexer.upsert may be sync or async; support both
                res = self.indexer.upsert(collection=collection, ids=ids, vectors=vectors, metas=metas)
                if asyncio.iscoroutine(res):
                    await res
                return
            except Exception as e:
                attempts += 1
                last_exc = e
                backoff = self.cfg.retry_backoff_s * (2 ** (attempts - 1))
                _logger.warning("indexer.upsert failed attempt %d for %s, retrying in %.2fs", attempts, rec["chunk_id"], backoff)
                await asyncio.sleep(backoff)
        # all retries failed -> DLQ
        _logger.error("indexer.upsert failed permanently for %s: %s", rec["chunk_id"], str(last_exc))
        self._dlq.append({"rec": rec, "error": "indexer_failed", "exc": str(last_exc)})
        self._metrics["errors"] += 1
        await self._maybe_persist_dlq()

    async def _maybe_persist_dlq(self) -> None:
        if not self.cfg.dlq_path:
            return
        try:
            async with aiofiles.open(self.cfg.dlq_path, "a", encoding="utf-8") as f:
                while self._dlq:
                    item = self._dlq.pop(0)
                    await f.write(json.dumps(item, ensure_ascii=False) + "\n")
        except Exception:
            _logger.exception("failed to persist dlq")

    # Readers

    async def _read_json(self, path: str) -> List[Dict[str, Any]]:
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            text = await f.read()
        data = json.loads(text)
        if isinstance(data, dict):
            return [self._normalize_extracted_doc(data)]
        return [self._normalize_extracted_doc(d) for d in data]

    async def _read_csv(self, path: str) -> List[Dict[str, Any]]:
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            text = await f.read()
        reader = csv.DictReader(text.splitlines())
        out: List[Dict[str, Any]] = []
        for row in reader:
            # prefer explicit 'text' column, else join values
            txt = row.get("text") or " ".join([v for v in row.values() if v])
            out.append(self._normalize_extracted_doc({"text": txt, "meta": row}))
        return out

    async def _read_text(self, path: str) -> Dict[str, Any]:
        async with aiofiles.open(path, "r", encoding="utf-8", errors="ignore") as f:
            txt = await f.read()
        return self._normalize_extracted_doc({"text": txt})

    async def _read_pdf(self, path: str) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        def extract_sync(p: str) -> Dict[str, Any]:
            try:
                from pdfminer.high_level import extract_text
                txt = extract_text(p)
                return {"text": txt}
            except Exception as e:
                _logger.exception("pdf extraction failed for %s", p)
                return {"text": "", "meta": {"pdf_error": str(e)}}
        return await loop.run_in_executor(self._executor, extract_sync, path)

    def _normalize_extracted_doc(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "raw_id": doc.get("raw_id"),
            "text": doc.get("text", "") or "",
            "meta": doc.get("meta", {}) or {}
        }

    # Utilities

    async def ingest_url(self, url: str, timeout: float = 10.0) -> IngestResult:
        async with self._sem:
            start = time.time()
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=timeout) as resp:
                        resp.raise_for_status()
                        payload = await resp.text()
                raw = self._normalize_extracted_doc({"text": payload, "meta": {"source_url": url}})
                return await self._process_single_raw_doc(raw, source=url, start=start)
            except Exception as e:
                _logger.exception("ingest_url failed %s", url)
                self._metrics["errors"] += 1
                self._dlq.append({"url": url, "error": "http_failed", "exc": str(e)})
                await self._maybe_persist_dlq()
                return {"url": url, "status": "error", "diagnostics": {"error": str(e)}}

    async def _process_single_raw_doc(self, raw: Dict[str, Any], source: str, start: float) -> IngestResult:
        raw_doc = {"raw_id": raw.get("raw_id") or str(uuid.uuid4()), "source": source, "text": raw.get("text", ""), "meta": raw.get("meta", {})}
        try:
            records = list(self.text_builder.build_records(raw_doc))
            for rec in records:
                await self._safe_upsert(rec)
            elapsed = time.time() - start
            self._metrics["ingested"] += 1
            self._metrics["latencies"].append(elapsed)
            return {"source": source, "status": "ok", "records_emitted": len(records), "time_s": elapsed}
        except Exception as e:
            _logger.exception("processing single raw doc failed for %s", source)
            self._dlq.append({"raw": raw_doc, "error": "process_failed", "exc": str(e)})
            await self._maybe_persist_dlq()
            self._metrics["errors"] += 1
            return {"source": source, "status": "error", "diagnostics": {"error": str(e)}}

    def snapshot(self) -> Dict[str, Any]:
        return {"metrics": dict(self._metrics), "dlq_len": len(self._dlq)}