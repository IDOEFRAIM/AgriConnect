# rag/indexer_milvus.py
from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)

_logger = logging.getLogger("rag.indexer_milvus")
_logger.addHandler(logging.NullHandler())


@dataclass
class MilvusConfig:
    host: str = "127.0.0.1"
    port: str = "19530"
    timeout_s: float = 5.0
    default_index_params: Optional[Dict[str, Any]] = None


class MilvusIndexer:
    def __init__(self, cfg: Optional[MilvusConfig] = None):
        default_params = {"index_type": "HNSW", "metric_type": "L2", "params": {"M": 16, "efConstruction": 200}}
        self.cfg = cfg or MilvusConfig(default_index_params=default_params)
        self._conn_alias: Optional[str] = None

    def connect(self, alias: str = "default") -> None:
        """Connect to Milvus server and store alias."""
        try:
            connections.connect(alias=alias, host=self.cfg.host, port=self.cfg.port)
            self._conn_alias = alias
            _logger.info("Connected to Milvus %s:%s (alias=%s)", self.cfg.host, self.cfg.port, alias)
        except Exception as e:
            _logger.exception("Failed to connect to Milvus %s:%s: %s", self.cfg.host, self.cfg.port, e)
            raise

    def _collection_exists(self, name: str) -> bool:
        try:
            return utility.has_collection(name)
        except Exception:
            # If utility fails (no connection), treat as not existing
            _logger.debug("utility.has_collection failed for %s", name, exc_info=True)
            return False

    def create_collection(self, name: str, dim: int, metric: str = "L2", index_params: Optional[Dict] = None) -> None:
        """
        Create a collection with a float vector field named 'embedding' and a JSON meta field.
        If the collection already exists, this is a no-op.
        """
        if self._collection_exists(name):
            _logger.info("Collection %s already exists", name)
            return

        index_params = index_params or self.cfg.default_index_params or {"index_type": "HNSW", "metric_type": metric, "params": {"M": 16, "efConstruction": 200}}

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=256),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="meta", dtype=DataType.JSON),
        ]
        schema = CollectionSchema(fields, description="RAG collection")
        try:
            coll = Collection(name, schema=schema)
            # create_index may raise if index already exists or params invalid
            try:
                coll.create_index(field_name="embedding", index_params=index_params)
            except Exception:
                _logger.debug("create_index raised an exception (may already exist): %s", name, exc_info=True)
            coll.load()
            _logger.info("Created collection %s dim=%d metric=%s", name, dim, metric)
        except Exception as e:
            _logger.exception("Failed to create collection %s: %s", name, e)
            raise

    def upsert(self, collection: str, ids: List[str], vectors: List[List[float]], metas: Optional[List[Dict]] = None, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Insert vectors and metadata into the collection.
        Returns a dict with inserted count, time_s and primary keys.
        """
        timeout = timeout or self.cfg.timeout_s
        if not self._collection_exists(collection):
            raise RuntimeError(f"collection {collection} does not exist")

        coll = Collection(collection)
        metas = metas or [{} for _ in ids]

        start = time.perf_counter()
        try:
            entities = [ids, vectors, metas]
            res = coll.insert(entities)
            coll.flush()
            dt = time.perf_counter() - start
            # res.primary_keys may be a list-like of inserted keys
            pks = getattr(res, "primary_keys", None)
            pks_out = [str(x) for x in pks] if pks is not None else []
            return {"inserted": len(ids), "time_s": round(dt, 6), "ids": pks_out}
        except Exception as e:
            _logger.exception("Failed to insert into collection %s: %s", collection, e)
            raise

    def search(self, collection: str, query_vecs: List[List[float]], top_k: int = 10, expr: Optional[str] = None, params: Optional[Dict] = None) -> List[List[Dict]]:
        """
        Perform a vector search. Returns a list (per query vector) of hits as dicts:
        { "id": str, "score": float, "meta": dict }
        """
        if not self._collection_exists(collection):
            raise RuntimeError(f"collection {collection} does not exist")

        coll = Collection(collection)
        # default search params (can be overridden)
        search_params = params or {"metric_type": "L2", "params": {"ef": 64}}
        try:
            results = coll.search(query_vecs, "embedding", search_params, limit=top_k, expr=expr, output_fields=["id", "meta"])
        except Exception as e:
            _logger.exception("Milvus search failed for collection %s: %s", collection, e)
            raise

        out: List[List[Dict]] = []
        for res in results:
            hits: List[Dict] = []
            for hit in res:
                try:
                    # hit.entity may be a dict-like; fallback to attributes if needed
                    ent = getattr(hit, "entity", None)
                    if ent and isinstance(ent, dict):
                        ent_id = ent.get("id")
                        ent_meta = ent.get("meta")
                    else:
                        # some pymilvus versions expose .id and .entity.get
                        ent_id = None
                        ent_meta = None
                        try:
                            ent_id = hit.entity.get("id") if hasattr(hit, "entity") and hasattr(hit.entity, "get") else getattr(hit, "id", None)
                        except Exception:
                            ent_id = getattr(hit, "id", None)
                        try:
                            ent_meta = hit.entity.get("meta") if hasattr(hit, "entity") and hasattr(hit.entity, "get") else None
                        except Exception:
                            ent_meta = None

                    # distance / score: pymilvus returns distance; for L2 smaller is better.
                    # We keep the raw distance as score; caller can interpret or convert.
                    distance = getattr(hit, "distance", None)
                    score = float(distance) if distance is not None else 0.0

                    hits.append({"id": str(ent_id) if ent_id is not None else "", "score": score, "meta": ent_meta})
                except Exception:
                    _logger.exception("Failed to parse hit from Milvus search result", exc_info=True)
            out.append(hits)
        return out

    def delete_by_ids(self, collection: str, ids: List[str]) -> None:
        if not self._collection_exists(collection):
            return
        coll = Collection(collection)
        try:
            expr = "id in [" + ",".join(f"'{i}'" for i in ids) + "]"
            coll.delete(expr)
            coll.flush()
        except Exception:
            _logger.exception("Failed to delete ids from collection %s", collection)

    def drop_collection(self, collection: str) -> None:
        if self._collection_exists(collection):
            try:
                Collection(collection).drop()
                _logger.info("Dropped collection %s", collection)
            except Exception:
                _logger.exception("Failed to drop collection %s", collection)

    def backup(self, collection: str, path: str) -> None:
        """Request a backup via Milvus utility API."""
        try:
            utility.backup(backup_name=f"backup_{collection}_{int(time.time())}", backup_path=path, collections=[collection])
            _logger.info("Backup requested for %s to %s", collection, path)
        except Exception:
            _logger.exception("Backup request failed for %s", collection)

    def restore(self, backup_name: str, path: str) -> None:
        try:
            utility.restore(backup_name=backup_name, backup_path=path)
            _logger.info("Restore requested %s from %s", backup_name, path)
        except Exception:
            _logger.exception("Restore request failed %s from %s", backup_name, path)