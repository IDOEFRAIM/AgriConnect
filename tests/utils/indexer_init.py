# rag/indexer_milvus.py - VERSION CORRIGÉE (text dans meta)
from __future__ import annotations

import time
import json
import requests
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
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
    host: str = "localhost"
    port: int = 19530
    timeout_s: float = 10.0
    default_index_params: Optional[Dict[str, Any]] = None


class MilvusIndexer:
    """Indexer Milvus - stocke le texte dans meta["text"]"""

    def __init__(self, cfg: Optional[MilvusConfig] = None):
        default_params = {
            "index_type": "HNSW",
            "metric_type": "L2",
            "params": {"M": 16, "efConstruction": 200},
        }
        self.cfg = cfg or MilvusConfig(default_index_params=default_params)
        self._conn_alias: Optional[str] = None
        self.last_error: Optional[str] = None
        self.last_op_time: Optional[float] = None

    def _record_error(self, msg: str) -> None:
        self.last_error = msg
        _logger.error(msg)

    def wait_for_milvus(self, host: str, port: int = 19530, timeout: int = 60):
        url = f"http://{host}:{port}/api/v1/health"
        start = time.time()
        while time.time() - start < timeout:
            try:
                r = requests.get(url)
                if r.status_code == 200:
                    data = r.json()
                    if data.get("status") in ("ok", "healthy"):
                        print("Milvus is healthy")
                        return True
            except Exception:
                pass
            time.sleep(2)
        raise RuntimeError("Milvus did not become healthy in time")
    
    def connect(self, alias: str = "default") -> Dict[str, Any]:
        """Wait for Milvus health, then connect via gRPC."""
        start = time.perf_counter()
        try:
            # Ensure Milvus is healthy before connecting
            self.wait_for_milvus(host=self.cfg.host, port=19530, timeout=self.cfg.timeout_s)

            connections.connect(
                alias=alias,
                host=self.cfg.host,
                port=int(self.cfg.port),   # ensure port is int
                timeout=self.cfg.timeout_s
            )
            self._conn_alias = alias
            dt = time.perf_counter() - start
            _logger.info("✓ Connected to Milvus %s:%s in %.3fs",
                         self.cfg.host, self.cfg.port, dt)
            self.last_op_time = dt
            return {"ok": True, "time_s": round(dt, 3), "alias": alias}
        except Exception as e:
            dt = time.perf_counter() - start
            msg = f"Failed to connect: {e}"
            self._record_error(msg)
            return {"ok": False, "time_s": round(dt, 3), "error": msg}



    def _collection_exists(self, name: str) -> bool:
        try:
            return utility.has_collection(name, using=self._conn_alias)
        except Exception:
            return False

    def create_collection(self, name: str, dim: int, metric: str = "L2", index_params: Optional[Dict] = None) -> Dict[str, Any]:
        start = time.perf_counter()
        
        if self._collection_exists(name):
            dt = time.perf_counter() - start
            _logger.info("Collection %s already exists", name)
            return {"ok": True, "created": False, "time_s": round(dt, 3)}

        # Schéma simple : id, embedding, meta (text stocké dans meta)
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=256),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="meta", dtype=DataType.JSON),
        ]
        
        schema = CollectionSchema(fields, description=f"RAG collection (dim={dim})")
        
        try:
            Collection(name, schema=schema, using=self._conn_alias)
            dt = time.perf_counter() - start
            _logger.info("✓ Created collection %s (dim=%d) in %.3fs", name, dim, dt)
            return {"ok": True, "created": True, "time_s": round(dt, 3), "dim": dim}
        except Exception as e:
            dt = time.perf_counter() - start
            msg = f"Failed to create collection: {e}"
            self._record_error(msg)
            return {"ok": False, "created": False, "time_s": round(dt, 3), "error": msg}

    def upsert(
        self,
        collection: str,
        ids: List[str],
        vectors: List[List[float]],
        metas: Optional[List[Dict]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        IMPORTANT: Le texte doit être dans metas[i]["text"]
        """
        start = time.perf_counter()
        
        if not ids or not vectors or len(ids) != len(vectors):
            return {"ok": False, "inserted": 0, "time_s": 0.0, "error": "Invalid input"}
        
        if not self._collection_exists(collection):
            return {"ok": False, "inserted": 0, "time_s": 0.0, "error": "Collection not found"}

        coll = Collection(collection, using=self._conn_alias)
        metas = metas or [{} for _ in ids]
        metas = [m if isinstance(m, dict) else {} for m in metas]

        try:
            entities = [ids, vectors, metas]
            res = coll.insert(entities, timeout=timeout)
            coll.flush()
            
            dt = time.perf_counter() - start
            _logger.info("✓ Inserted %d items in %.3fs", len(ids), dt)
            
            return {"ok": True, "inserted": len(ids), "time_s": round(dt, 3), "ids": ids[:5]}
        except Exception as e:
            dt = time.perf_counter() - start
            msg = f"Insert failed: {e}"
            self._record_error(msg)
            _logger.exception("Insert error:")
            return {"ok": False, "inserted": 0, "time_s": round(dt, 3), "error": msg}



    def build_index(self, collection: str, index_params: Optional[Dict] = None, field_name: str = "embedding") -> Dict[str, Any]:
        start = time.perf_counter()
        
        if not self._collection_exists(collection):
            return {"ok": False, "time_s": 0.0, "error": "Collection not found"}

        coll = Collection(collection, using=self._conn_alias)
        index_params = index_params or self.cfg.default_index_params

        try:
            has_index = False
            try:
                has_index = len(coll.indexes) > 0
            except Exception:
                pass
            
            if not has_index:
                _logger.info("Creating index on %s", field_name)
                coll.create_index(field_name=field_name, index_params=index_params)
            
            _logger.info("Loading collection...")
            coll.load()
            
            num_entities = coll.num_entities
            dt = time.perf_counter() - start
            _logger.info("✓ Loaded %d entities in %.3fs", num_entities, dt)
            
            return {"ok": True, "time_s": round(dt, 3), "loaded": True, "num_entities": num_entities}
        except Exception as e:
            dt = time.perf_counter() - start
            msg = f"Build index failed: {e}"
            self._record_error(msg)
            return {"ok": False, "time_s": round(dt, 3), "error": msg}

    def search(
        self,
        collection: str,
        query_vecs: List[List[float]],
        top_k: int = 10,
        expr: Optional[str] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Retourne hits avec text extrait de meta["text"]
        """
        start = time.perf_counter()
        
        if not query_vecs:
            return {"ok": False, "results": [], "num_queries": 0, "time_s": 0.0, "error": "Empty query"}
        
        if not self._collection_exists(collection):
            return {"ok": False, "results": [], "num_queries": 0, "time_s": 0.0, "error": "Collection not found"}

        coll = Collection(collection, using=self._conn_alias)
        
        try:
            num_entities = coll.num_entities
            if num_entities == 0:
                return {"ok": True, "results": [[] for _ in query_vecs], "num_queries": len(query_vecs), "total_hits": 0, "time_s": 0.0}
        except Exception:
            pass
        
        search_params = params or {"metric_type": "L2", "params": {"ef": 64}}
        
        try:
            results = coll.search(
                data=query_vecs,
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["meta"],
            )
            
            out = []
            for res in results:
                hits = []
                for hit in res:
                    # ID
                    hit_id = None
                    try:
                        hit_id = hit.id
                    except AttributeError:
                        try:
                            hit_id = hit.entity.get("id")
                        except Exception:
                            pass
                    
                    # Distance → Score
                    distance = getattr(hit, "distance", 0.0)
                    score = 1.0 / (1.0 + float(distance)) if distance is not None else 0.0
                    
                    # Meta (contient le text)
                    meta = {}
                    try:
                        meta = hit.entity.get("meta", {})
                        if not isinstance(meta, dict):
                            meta = {}
                    except Exception:
                        meta = {}
                    
                    # Extraire text de meta
                    text = meta.get("text", "")
                    
                    hits.append({
                        "id": str(hit_id) if hit_id else None,
                        "score": float(score),
                        "distance": float(distance),
                        "text": str(text),  # ← EXTRAIT de meta["text"]
                        "meta": meta
                    })
                
                out.append(hits)
            
            dt = time.perf_counter() - start
            total_hits = sum(len(h) for h in out)
            _logger.info("✓ Search: %d queries → %d hits in %.3fs", len(query_vecs), total_hits, dt)
            
            return {"ok": True, "results": out, "num_queries": len(query_vecs), "total_hits": total_hits, "time_s": round(dt, 3)}
        except Exception as e:
            dt = time.perf_counter() - start
            msg = f"Search failed: {e}"
            self._record_error(msg)
            _logger.exception("Search error:")
            return {"ok": False, "results": [], "num_queries": 0, "time_s": round(dt, 3), "error": msg}

    def delete_by_ids(self, collection: str, ids: List[str]) -> Dict[str, Any]:
        start = time.perf_counter()
        if not self._collection_exists(collection):
            return {"ok": False, "error": "collection not found", "time_s": 0.0}
        
        try:
            coll = Collection(collection, using=self._conn_alias)
            expr = "id in [" + ",".join(f"'{i}'" for i in ids) + "]"
            coll.delete(expr)
            coll.flush()
            dt = time.perf_counter() - start
            return {"ok": True, "time_s": round(dt, 3), "deleted": len(ids)}
        except Exception as e:
            dt = time.perf_counter() - start
            return {"ok": False, "error": str(e), "time_s": round(dt, 3)}

    def drop_collection(self, collection: str) -> Dict[str, Any]:
        if self._collection_exists(collection):
            try:
                Collection(collection, using=self._conn_alias).drop()
                _logger.info("✓ Dropped collection %s", collection)
                return {"ok": True}
            except Exception as e:
                return {"ok": False, "error": str(e)}
        return {"ok": True, "message": "collection did not exist"}

    def check_collection_state(self, collection: str) -> Dict[str, Any]:
        try:
            if not self._collection_exists(collection):
                return {"ok": True, "exists": False, "num_entities": 0, "is_loaded": False, "has_index": False}
            
            coll = Collection(collection, using=self._conn_alias)
            num = int(coll.num_entities) if hasattr(coll, "num_entities") else -1
            has_index = len(coll.indexes) > 0 if hasattr(coll, "indexes") else False
            
            return {"ok": True, "exists": True, "num_entities": num, "is_loaded": num > 0, "has_index": has_index}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def verify_top1(self, collection: str, sample_id: str, embedder: Any, top_k: int = 5) -> Dict[str, Any]:
        try:
            if not self._collection_exists(collection):
                return {"ok": False, "error": "collection not found"}

            coll = Collection(collection, using=self._conn_alias)
            
            text = None
            try:
                qres = coll.query(expr=f"id == '{sample_id}'", output_fields=["meta"])
                if qres:
                    meta = qres[0].get("meta", {})
                    text = meta.get("text", "") if isinstance(meta, dict) else ""
            except Exception:
                pass
            
            if not text:
                text = sample_id
            
            if hasattr(embedder, "encode"):
                qvec = embedder.encode([text])[0]
            elif hasattr(embedder, "encode_async"):
                import asyncio
                qvec = asyncio.run(embedder.encode_async([text]))[0]
            else:
                return {"ok": False, "error": "embedder has no encode method"}
            
            search_res = self.search(collection, [qvec], top_k=top_k)
            if not search_res.get("ok"):
                return {"ok": False, "error": "search failed"}
            
            results = search_res.get("results", [[]])[0]
            result_ids = [r.get("id") for r in results]
            
            rank = None
            for i, rid in enumerate(result_ids):
                if rid == sample_id:
                    rank = i
                    break
            
            return {"ok": True, "found_top1": rank == 0, "rank": rank, "result_ids": result_ids}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def init_collection_from_folder(
        self,
        collection: str,
        embedder: Any,
        folder: str | Path,
        dim: Optional[int] = None,
        file_glob: str = "*.json",
        text_field: str = "text",
        id_field: str = "id",
        meta_field: Optional[str] = "meta",
        batch_size: int = 128,
        metric: str = "L2",
        index_params: Optional[Dict] = None,
        connect_alias: str = "default",
        verify_top1_after: bool = True,
        sample_verify_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        summary = {"created": False, "indexed": 0, "errors": 0, "steps": {}, "time_s": 0.0}
        start_total = time.perf_counter()

        folder = Path(folder)
        if not folder.exists():
            return {"ok": False, "error": "folder not found"}

        if self._conn_alias is None:
            conn_res = self.connect(alias=connect_alias)
            summary["steps"]["connect"] = conn_res
            if not conn_res.get("ok"):
                summary["errors"] += 1
                return {"ok": False, "summary": summary}

        files = list(folder.glob(file_glob))
        if not files:
            return {"ok": True, "summary": summary}

        records = []
        for f in files:
            try:
                raw = json.loads(f.read_text(encoding="utf-8"))
                items = raw if isinstance(raw, list) else [raw]
                
                for it in items:
                    text = it.get(text_field) if isinstance(it, dict) else None
                    if not text and meta_field and isinstance(it, dict):
                        meta = it.get(meta_field, {})
                        text = meta.get("text") if isinstance(meta, dict) else None
                    
                    if not text:
                        continue
                    
                    rec_id = str(it.get(id_field) or f"{f.stem}_{hashlib.sha1(text.encode()).hexdigest()[:8]}")
                    meta = it.get(meta_field, {}) if isinstance(it, dict) else {}
                    if isinstance(meta, dict):
                        meta.setdefault("source", str(f))
                        meta["text"] = text  # ← STOCKER le texte dans meta
                    
                    records.append({"id": rec_id, "text": text, "meta": meta if isinstance(meta, dict) else {"source": str(f), "text": text}})
            except Exception:
                _logger.exception("Failed to parse file")
                summary["errors"] += 1

        if not records:
            return {"ok": False, "error": "No valid records"}

        # Get dimension
        try:
            sample_text = records[0]["text"]
            if hasattr(embedder, "encode"):
                sample_emb = embedder.encode([sample_text])[0]
            else:
                import asyncio
                sample_emb = asyncio.run(embedder.encode_async([sample_text]))[0]
            dim = dim or len(sample_emb)
        except Exception as e:
            return {"ok": False, "error": f"Failed to encode: {e}"}

        # Create collection
        create_res = self.create_collection(collection, dim=dim, metric=metric)
        summary["steps"]["create_collection"] = create_res
        summary["created"] = create_res.get("created", False)
        if not create_res.get("ok"):
            summary["errors"] += 1
            return {"ok": False, "summary": summary}

        # Insert batches
        total_inserted = 0
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            ids = [r["id"] for r in batch]
            texts = [r["text"] for r in batch]
            metas = [r["meta"] for r in batch]
            
            try:
                if hasattr(embedder, "encode"):
                    vecs = embedder.encode(texts)
                else:
                    import asyncio
                    vecs = asyncio.run(embedder.encode_async(texts))
                
                insert_res = self.upsert(collection, ids, vecs, metas)
                if insert_res.get("ok"):
                    total_inserted += insert_res.get("inserted", 0)
                else:
                    summary["errors"] += 1
            except Exception:
                _logger.exception("Batch insert failed")
                summary["errors"] += 1

        summary["indexed"] = total_inserted

        # Build index
        index_res = self.build_index(collection, index_params=index_params)
        summary["steps"]["build_index"] = index_res
        if not index_res.get("ok"):
            summary["errors"] += 1

        # Verify
        if verify_top1_after and total_inserted > 0:
            verify_id = sample_verify_id or records[0]["id"]
            verify_res = self.verify_top1(collection, verify_id, embedder)
            summary["steps"]["verify_top1"] = verify_res

        summary["time_s"] = round(time.perf_counter() - start_total, 3)
        summary["ok"] = summary["errors"] == 0
        
        _logger.info("✓ Initialized: %d docs in %.3fs", total_inserted, summary["time_s"])
        return {"ok": summary["ok"], "summary": summary}