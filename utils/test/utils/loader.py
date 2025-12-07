# rag/loader.py
from __future__ import annotations
import asyncio
import time
import json
import csv
import os
import fnmatch
import logging
import uuid
import concurrent.futures
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Set, Tuple, Callable

import aiofiles

_logger = logging.getLogger("rag.loader")
_logger.addHandler(logging.NullHandler())

@dataclass
class LoaderConfig:
    recursive: bool = False
    patterns: Tuple[str, ...] = ("*.json", "*.csv", "*.txt", "*.pdf")
    poll_interval_s: float = 2.0
    checkpoint_path: Optional[str] = None
    read_encoding: str = "utf-8"
    ignore_hidden: bool = True
    max_concurrency: int = 4
    pdf_executor_workers: int = 2

class Loader:
    """
    Loader: lightweight component that enumerates files and yields normalized raw_docs.
    - Does NOT perform heavy transformation (delegated to TextBuilder/Ingestor).
    - Supports batch streaming, folder watching (polling), checkpointing processed files.
    - Minimal dependencies; PDF extraction runs in executor to avoid blocking.
    """

    def __init__(self, cfg: Optional[LoaderConfig] = None):
        self.cfg = cfg or LoaderConfig()
        self._processed: Set[str] = set()
        if self.cfg.checkpoint_path:
            try:
                self._load_checkpoint()
            except Exception:
                _logger.exception("failed to load checkpoint, starting fresh")
        self._sem = asyncio.Semaphore(self.cfg.max_concurrency)
        # Use concurrent.futures.ThreadPoolExecutor for blocking PDF extraction
        self._pdf_executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.cfg.pdf_executor_workers)

    # -------------------------
    # Public streaming APIs
    # -------------------------
    async def stream_paths(self, paths: Iterable[str]) -> AsyncIterator[str]:
        """
        Yield file paths matching patterns under given paths.
        paths can be files or directories.
        """
        for p in paths:
            if os.path.isfile(p):
                if self._match_patterns(p):
                    yield p
                continue
            # directory
            if self.cfg.recursive:
                for root, dirs, files in os.walk(p):
                    if self.cfg.ignore_hidden:
                        dirs[:] = [d for d in dirs if not d.startswith(".")]
                    for name in files:
                        fp = os.path.join(root, name)
                        if self._match_patterns(fp):
                            yield fp
            else:
                for name in os.listdir(p):
                    if self.cfg.ignore_hidden and name.startswith("."):
                        continue
                    fp = os.path.join(p, name)
                    if os.path.isfile(fp) and self._match_patterns(fp):
                        yield fp

    async def stream_raw_docs(self, paths: Iterable[str]) -> AsyncIterator[Dict[str, Any]]:
        """
        For each matching file, yield one or more normalized raw_doc dicts:
        {"raw_id","source","text","meta"}
        """
        async for fp in self.stream_paths(paths):
            if fp in self._processed:
                continue
            try:
                async with self._sem:
                    docs = await self._load_file(fp)
                for d in docs:
                    yield d
                self._mark_processed(fp)
            except Exception:
                _logger.exception("failed to load file %s", fp)
                # still mark processed to avoid infinite retry loops
                self._mark_processed(fp)

    async def watch_folder(self, folder: str) -> AsyncIterator[Dict[str, Any]]:
        """
        Polling-based folder watcher. Yields raw_docs as files appear.
        Use with care for large folders; suitable for simple deployments.
        """
        while True:
            try:
                async for raw in self.stream_raw_docs([folder]):
                    yield raw
            except Exception:
                _logger.exception("watch_folder iteration error")
            await asyncio.sleep(self.cfg.poll_interval_s)

    # -------------------------
    # Internal helpers
    # -------------------------
    def _match_patterns(self, path: str) -> bool:
        name = os.path.basename(path)
        for pat in self.cfg.patterns:
            if fnmatch.fnmatch(name.lower(), pat.lower()):
                return True
        return False

    async def _load_file(self, path: str) -> List[Dict[str, Any]]:
        """
        Dispatch by extension and return list of normalized raw docs.
        """
        lower = path.lower()
        if lower.endswith(".json"):
            return await self._read_json(path)
        if lower.endswith(".csv"):
            return await self._read_csv(path)
        if lower.endswith(".pdf"):
            return [await self._read_pdf(path)]
        # default: text
        return [await self._read_text(path)]

    async def _read_json(self, path: str) -> List[Dict[str, Any]]:
        async with aiofiles.open(path, "r", encoding=self.cfg.read_encoding, errors="ignore") as f:
            text = await f.read()
        try:
            data = json.loads(text)
        except Exception:
            _logger.exception("json parse failed for %s", path)
            return [{"raw_id": str(uuid.uuid4()), "source": path, "text": "", "meta": {"error": "json_parse_failed"}}]
        if isinstance(data, dict):
            return [self._normalize_extracted_doc(data, source=path)]
        out = []
        for item in data:
            out.append(self._normalize_extracted_doc(item, source=path))
        return out

    async def _read_csv(self, path: str) -> List[Dict[str, Any]]:
        async with aiofiles.open(path, "r", encoding=self.cfg.read_encoding, errors="ignore") as f:
            text = await f.read()
        try:
            reader = csv.DictReader(text.splitlines())
        except Exception:
            _logger.exception("csv parse failed for %s", path)
            return [{"raw_id": str(uuid.uuid4()), "source": path, "text": "", "meta": {"error": "csv_parse_failed"}}]
        out: List[Dict[str, Any]] = []
        for row in reader:
            txt = row.get("text") or " ".join([v for v in row.values() if v])
            out.append(self._normalize_extracted_doc({"text": txt, "meta": row}, source=path))
        return out

    async def _read_text(self, path: str) -> Dict[str, Any]:
        async with aiofiles.open(path, "r", encoding=self.cfg.read_encoding, errors="ignore") as f:
            txt = await f.read()
        return self._normalize_extracted_doc({"text": txt}, source=path)

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
        return await loop.run_in_executor(self._pdf_executor, extract_sync, path)

    def _normalize_extracted_doc(self, doc: Dict[str, Any], source: Optional[str] = None) -> Dict[str, Any]:
        # Ensure minimal contract for downstream components
        return {
            "raw_id": doc.get("raw_id") or str(uuid.uuid4()),
            "source": source or doc.get("source") or "",
            "text": doc.get("text", "") or "",
            "meta": doc.get("meta", {}) or {}
        }

    # -------------------------
    # Checkpointing / processed tracking
    # -------------------------
    def _mark_processed(self, path: str) -> None:
        self._processed.add(path)
        if self.cfg.checkpoint_path:
            try:
                with open(self.cfg.checkpoint_path, "a", encoding="utf-8") as f:
                    f.write(path + "\n")
            except Exception:
                _logger.exception("failed to persist checkpoint for %s", path)

    def _load_checkpoint(self) -> None:
        if not self.cfg.checkpoint_path or not os.path.exists(self.cfg.checkpoint_path):
            return
        try:
            with open(self.cfg.checkpoint_path, "r", encoding="utf-8") as f:
                for line in f:
                    p = line.strip()
                    if p:
                        self._processed.add(p)
        except Exception:
            _logger.exception("failed to read checkpoint file %s", self.cfg.checkpoint_path)

    # -------------------------
    # Utilities
    # -------------------------
    def reset_checkpoint(self) -> None:
        self._processed.clear()
        if self.cfg.checkpoint_path and os.path.exists(self.cfg.checkpoint_path):
            try:
                os.remove(self.cfg.checkpoint_path)
            except Exception:
                _logger.exception("failed to remove checkpoint file %s", self.cfg.checkpoint_path)

    def processed_count(self) -> int:
        return len(self._processed)

    # -------------------------
    # Convenience runner
    # -------------------------
    async def load_and_dispatch(self, paths: Iterable[str], dispatcher: Optional[Callable[[Dict[str, Any]], Any]] = None) -> Dict[str, Any]:
        """
        Helper that streams raw_docs and optionally dispatches each to a provided dispatcher callable.
        Dispatcher can be sync or async. Returns summary metrics.
        """
        start = time.time()
        count = 0
        async for raw in self.stream_raw_docs(paths):
            count += 1
            if dispatcher:
                try:
                    res = dispatcher(raw)
                    if asyncio.iscoroutine(res):
                        await res
                except Exception:
                    _logger.exception("dispatcher failed for raw %s", raw.get("raw_id"))
            # continue streaming
        elapsed = time.time() - start
        return {"loaded": count, "time_s": elapsed, "processed": self.processed_count()}