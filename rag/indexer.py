# milvus_ops.py - FICHIER COMPLET
from __future__ import annotations
import time
import json
import logging
import argparse
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from enum import Enum
from pathlib import Path

_logger = logging.getLogger("milvus_ops")

# Lazy import placeholders for pymilvus
try:
    from pymilvus import (
        connections,
        utility,
        FieldSchema,
        CollectionSchema,
        DataType,
        Collection,
        MilvusException,
    )
    PYMILVUS_AVAILABLE = True
except ImportError:
    connections = None
    utility = None
    FieldSchema = None
    CollectionSchema = None
    DataType = None
    Collection = None
    MilvusException = Exception
    PYMILVUS_AVAILABLE = False


# --- CONFIGURATION ---
class MilvusScalarType(Enum):
    """Supported scalar field types for filtering"""
    INT64 = "INT64"
    VARCHAR = "VARCHAR"
    FLOAT = "FLOAT"
    DOUBLE = "DOUBLE"
    BOOL = "BOOL"
    
    def to_datatype(self):
        """Convert to pymilvus DataType"""
        if not PYMILVUS_AVAILABLE:
            raise RuntimeError("pymilvus not available")
        return getattr(DataType, self.value)


@dataclass
class ScalarFieldConfig:
    """Configuration for a scalar field"""
    name: str
    field_type: MilvusScalarType
    max_length: Optional[int] = None
    description: str = ""
    
    def to_field_schema(self) -> FieldSchema:
        """Convert to pymilvus FieldSchema"""
        kwargs = {"name": self.name, "dtype": self.field_type.to_datatype()}
        
        if self.field_type == MilvusScalarType.VARCHAR:
            if not self.max_length:
                raise ValueError(f"max_length required for VARCHAR field {self.name}")
            kwargs["max_length"] = self.max_length
        
        return FieldSchema(**kwargs)


@dataclass
class IndexerConfig:
    # Connection
    host: str = "127.0.0.1"
    port: str = "19530"
    alias: str = "default"
    connect_timeout: float = 10.0
    
    # Collection schema
    collection_name: str = "rag_collection"
    id_field: str = "chunk_id"
    vector_field: str = "vector"
    text_field: str = "text"
    metadata_json_field: str = "metadata_json"
    dim: int = 384
    
    # Index configuration
    index_type: str = "HNSW"
    metric_type: str = "IP"
    index_params: Dict[str, Any] = field(
        default_factory=lambda: {"M": 32, "efConstruction": 200}
    )
    
    # Search parameters
    ef_search: int = 128
    
    # Batch operations
    insert_batch_size: int = 512
    timeout: float = 30.0
    
    # Backup
    backup_root: str = "/tmp/milvus_backups"
    
    # Scalar fields
    scalar_fields: List[ScalarFieldConfig] = field(
        default_factory=lambda: [
            ScalarFieldConfig("doc_id", MilvusScalarType.VARCHAR, 128, "Source document ID"),
            ScalarFieldConfig("source", MilvusScalarType.VARCHAR, 256, "Data source"),
            ScalarFieldConfig("created_at", MilvusScalarType.INT64, None, "Unix timestamp"),
            ScalarFieldConfig("chunk_index", MilvusScalarType.INT64, None, "Chunk position"),
        ]
    )
    
    auto_load: bool = True
    consistency_level: str = "Strong"


# --- MAIN CLASS ---
class MilvusIndexerOps:
    """Production-grade Milvus indexer for RAG pipelines"""
    
    def __init__(self, cfg: IndexerConfig):
        if not PYMILVUS_AVAILABLE:
            raise RuntimeError("pymilvus is required. Install with: pip install pymilvus")
        
        self.cfg = cfg
        self._lock = threading.RLock()
        self._collection: Optional[Collection] = None
        self._connected = False
        self._connect()
        
        _logger.info(f"MilvusIndexerOps initialized: {cfg.host}:{cfg.port}, collection={cfg.collection_name}")

    # ==================== CONNECTION ====================
    
    def _connect(self) -> None:
        """Establish connection to Milvus server"""
        with self._lock:
            if self._connected:
                return
            
            try:
                connections.connect(
                    alias=self.cfg.alias,
                    host=self.cfg.host,
                    port=self.cfg.port,
                    timeout=self.cfg.connect_timeout
                )
                self._connected = True
                _logger.info(f"Connected to Milvus: {self.cfg.host}:{self.cfg.port}")
            except Exception as e:
                self._connected = False
                _logger.error(f"Milvus connection failed: {e}", exc_info=True)
                raise

    def reconnect(self) -> None:
        """Reconnect to Milvus"""
        with self._lock:
            try:
                connections.disconnect(self.cfg.alias)
            except Exception:
                pass
            
            self._connected = False
            self._collection = None
            self._connect()

    # ==================== COLLECTION MANAGEMENT ====================
    
    def ensure_collection(self, overwrite: bool = False) -> None:
        """Create or load collection"""
        if not self._connected:
            self._connect()
        
        with self._lock:
            try:
                collection_exists = utility.has_collection(self.cfg.collection_name, using=self.cfg.alias)
                
                if collection_exists:
                    if overwrite:
                        _logger.warning(f"Dropping collection: {self.cfg.collection_name}")
                        utility.drop_collection(self.cfg.collection_name, using=self.cfg.alias)
                    else:
                        self._collection = Collection(self.cfg.collection_name, using=self.cfg.alias)
                        _logger.info(f"Using existing collection: {self.cfg.collection_name}")
                        if self.cfg.auto_load:
                            self.load_collection()
                        return
                
                # Create new collection
                schema = self._build_schema()
                self._collection = Collection(
                    name=self.cfg.collection_name,
                    schema=schema,
                    using=self.cfg.alias,
                    consistency_level=self.cfg.consistency_level
                )
                
                _logger.info(f"Created collection: {self.cfg.collection_name} with {len(schema.fields)} fields")
                
                # Create vector index
                self.create_index()
                
                # Load into memory
                if self.cfg.auto_load:
                    self.load_collection()
                
            except Exception as e:
                _logger.error(f"ensure_collection failed: {e}", exc_info=True)
                raise

    def _build_schema(self) -> CollectionSchema:
        """Build collection schema"""
        fields: List[FieldSchema] = []
        
        # Primary key
        fields.append(FieldSchema(
            name=self.cfg.id_field,
            dtype=DataType.VARCHAR,
            is_primary=True,
            auto_id=False,
            max_length=256
        ))
        
        # Vector field
        fields.append(FieldSchema(
            name=self.cfg.vector_field,
            dtype=DataType.FLOAT_VECTOR,
            dim=self.cfg.dim
        ))
        
        # Text field
        fields.append(FieldSchema(
            name=self.cfg.text_field,
            dtype=DataType.VARCHAR,
            max_length=65535
        ))
        
        # Metadata JSON
        fields.append(FieldSchema(
            name=self.cfg.metadata_json_field,
            dtype=DataType.VARCHAR,
            max_length=65535
        ))
        
        # Scalar fields
        for field_cfg in self.cfg.scalar_fields:
            try:
                fields.append(field_cfg.to_field_schema())
            except Exception as e:
                _logger.warning(f"Failed to add scalar field {field_cfg.name}: {e}")
        
        schema = CollectionSchema(
            fields=fields,
            description=f"RAG collection with HNSW index (dim={self.cfg.dim})"
        )
        
        return schema

    def create_index(self, field_name: Optional[str] = None, index_params: Optional[Dict[str, Any]] = None) -> None:
        """Create HNSW index on vector field"""
        if self._collection is None:
            self.ensure_collection()
        
        field = field_name or self.cfg.vector_field
        
        # Check if index exists
        try:
            indexes = self._collection.indexes
            if any(idx.field_name == field for idx in indexes):
                _logger.info(f"Index already exists on {field}")
                return
        except Exception:
            pass
        
        # Build index parameters
        params = index_params or {
            "index_type": self.cfg.index_type,
            "metric_type": self.cfg.metric_type,
            "params": self.cfg.index_params
        }
        
        try:
            _logger.info(f"Creating index on {field}: {params}")
            self._collection.create_index(field_name=field, index_params=params)
            _logger.info(f"Index created successfully on {field}")
            
            if self.cfg.auto_load:
                self._collection.load()
        except Exception as e:
            _logger.error(f"create_index failed: {e}", exc_info=True)
            raise

    def drop_collection(self) -> None:
        """Drop the collection permanently"""
        try:
            utility.drop_collection(self.cfg.collection_name, using=self.cfg.alias)
            _logger.info(f"Dropped collection: {self.cfg.collection_name}")
            self._collection = None
        except Exception as e:
            _logger.error(f"drop_collection failed: {e}", exc_info=True)
            raise

    # ==================== INSERT / UPSERT ====================
    
    def insert(
        self,
        records: Sequence[Dict[str, Any]],
        id_field: Optional[str] = None,
        text_field: Optional[str] = None,
        vector_field: Optional[str] = None,
        meta_json_field: Optional[str] = None,
        batch_size: Optional[int] = None,
        timeout: Optional[float] = None,
        skip_on_error: bool = True
    ) -> Dict[str, Any]:
        """Insert records into Milvus collection"""
        if self._collection is None:
            self.ensure_collection()
        
        start_time = time.time()
        
        # Field mappings
        idf = id_field or self.cfg.id_field
        tf = text_field or self.cfg.text_field
        vf = vector_field or self.cfg.vector_field
        mjf = meta_json_field or self.cfg.metadata_json_field
        bs = batch_size or self.cfg.insert_batch_size
        timeout_val = timeout or self.cfg.timeout
        
        inserted = 0
        failed = 0
        errors: List[str] = []
        
        # Prepare records
        rows: List[Dict[str, Any]] = []
        
        for idx, rec in enumerate(records):
            try:
                doc_id = rec.get(idf)
                vec = rec.get(vf)
                text = rec.get(tf, "")
                
                if doc_id is None:
                    raise ValueError(f"Missing {idf}")
                if vec is None:
                    raise ValueError(f"Missing {vf}")
                if not isinstance(vec, list):
                    vec = list(vec)
                
                # Build Milvus record
                milvus_rec = {
                    idf: str(doc_id),
                    vf: vec,
                    tf: text,
                }
                
                # Serialize metadata
                meta = rec.get("meta") or {}
                try:
                    milvus_rec[mjf] = json.dumps(meta, ensure_ascii=False)
                except Exception:
                    milvus_rec[mjf] = "{}"
                
                # Add scalar fields
                for field_cfg in self.cfg.scalar_fields:
                    field_name = field_cfg.name
                    if field_name in rec:
                        value = rec[field_name]
                        
                        # Type coercion
                        if field_cfg.field_type == MilvusScalarType.VARCHAR:
                            value = str(value)[:field_cfg.max_length or 65535]
                        elif field_cfg.field_type == MilvusScalarType.INT64:
                            value = int(value)
                        elif field_cfg.field_type in (MilvusScalarType.FLOAT, MilvusScalarType.DOUBLE):
                            value = float(value)
                        elif field_cfg.field_type == MilvusScalarType.BOOL:
                            value = bool(value)
                        
                        milvus_rec[field_name] = value
                
                rows.append(milvus_rec)
                
            except Exception as e:
                failed += 1
                error_msg = f"Record {idx} (id={rec.get(idf)}): {str(e)}"
                errors.append(error_msg)
                _logger.debug(f"Prepare failed: {error_msg}")
                
                if not skip_on_error:
                    raise
        
        # Batch insert
        for batch_start in range(0, len(rows), bs):
            batch = rows[batch_start:batch_start + bs]
            
            try:
                self._collection.insert(batch, timeout=timeout_val)
                inserted += len(batch)
                _logger.debug(f"Inserted batch {batch_start//bs + 1}: {len(batch)} records")
            except Exception as e:
                failed += len(batch)
                error_msg = f"Batch {batch_start//bs + 1} failed: {str(e)}"
                errors.append(error_msg)
                _logger.error(error_msg, exc_info=True)
                
                if not skip_on_error:
                    raise
        
        # Flush
        try:
            self._collection.flush()
        except Exception as e:
            _logger.warning(f"Flush failed: {e}")
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        result = {
            "inserted": inserted,
            "failed": failed,
            "total": len(records),
            "errors": errors[:20],
            "elapsed_ms": elapsed_ms
        }
        
        _logger.info(f"Insert complete: {inserted} inserted, {failed} failed, {elapsed_ms}ms")
        
        return result

    def upsert(self, records: Sequence[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Upsert records"""
        return self.insert(records, **kwargs)

    # ==================== SEARCH ====================
    
    def search(
        self,
        query_vec: Union[List[float], List[List[float]]],
        top_k: int = 10,
        expr: Optional[str] = None,
        ef: Optional[int] = None,
        output_fields: Optional[List[str]] = None,
        timeout: Optional[float] = None
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """Search for similar vectors"""
        if self._collection is None:
            self.ensure_collection()
        
        # Normalize input to batch
        if isinstance(query_vec[0], (int, float)):
            query_vecs = [query_vec]
            single_query = True
        else:
            query_vecs = query_vec
            single_query = False
        
        # Search parameters
        ef_val = ef or self.cfg.ef_search
        search_params = {
            "metric_type": self.cfg.metric_type,
            "params": {"ef": ef_val}
        }
        
        # Output fields
        if output_fields is None:
            output_fields = [self.cfg.text_field, self.cfg.metadata_json_field]
            output_fields.extend([f.name for f in self.cfg.scalar_fields])
        
        try:
            results = self._collection.search(
                data=query_vecs,
                anns_field=self.cfg.vector_field,
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=output_fields,
                timeout=timeout or self.cfg.timeout,
                consistency_level=self.cfg.consistency_level
            )
            
            # Parse results
            all_hits = []
            for result_set in results:
                hits = []
                for hit in result_set:
                    hit_data = {
                        "id": hit.id,
                        "score": float(hit.score),
                        "distance": float(hit.distance) if hasattr(hit, "distance") else float(hit.score)
                    }
                    
                    # Extract entity fields
                    entity = hit.entity
                    
                    hit_data["text"] = entity.get(self.cfg.text_field, "")
                    
                    # Deserialize metadata
                    meta_json = entity.get(self.cfg.metadata_json_field, "{}")
                    try:
                        hit_data["meta"] = json.loads(meta_json) if meta_json else {}
                    except Exception:
                        hit_data["meta"] = {}
                    
                    # Scalar fields
                    for field_cfg in self.cfg.scalar_fields:
                        field_name = field_cfg.name
                        if field_name in entity:
                            hit_data[field_name] = entity[field_name]
                    
                    hits.append(hit_data)
                
                all_hits.append(hits)
            
            return all_hits[0] if single_query else all_hits
            
        except Exception as e:
            _logger.error(f"Search failed: {e}", exc_info=True)
            raise

    def search_by_text(self, embedder, query_text: str, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """Search by text query"""
        query_vec = embedder.encode([query_text], is_query=True)[0]
        return self.search(query_vec, top_k=top_k, **kwargs)

    # ==================== BACKUP / RESTORE ====================
    
    def create_backup(self, backup_name: Optional[str] = None, backup_root: Optional[str] = None) -> Dict[str, Any]:
        """Create collection backup"""
        backup_root = backup_root or self.cfg.backup_root
        backup_name = backup_name or f"{self.cfg.collection_name}_{int(time.time())}"
        
        try:
            Path(backup_root).mkdir(parents=True, exist_ok=True)
            res = utility.create_backup(
                backup_name=backup_name,
                collection_names=[self.cfg.collection_name],
                using=self.cfg.alias
            )
            _logger.info(f"Backup created: {backup_name}")
            return {"ok": True, "backup_name": backup_name, "backup_root": backup_root, "result": res}
        except Exception as e:
            _logger.error(f"Backup failed: {e}", exc_info=True)
            return {"ok": False, "error": str(e)}

    def restore_backup(self, backup_name: str, backup_root: Optional[str] = None, overwrite: bool = False) -> Dict[str, Any]:
        """Restore collection from backup"""
        backup_root = backup_root or self.cfg.backup_root
        
        try:
            if overwrite and utility.has_collection(self.cfg.collection_name, using=self.cfg.alias):
                utility.drop_collection(self.cfg.collection_name, using=self.cfg.alias)
            
            res = utility.restore_backup(
                backup_name=backup_name,
                collection_names=[self.cfg.collection_name],
                using=self.cfg.alias
            )
            self._collection = Collection(self.cfg.collection_name, using=self.cfg.alias)
            _logger.info(f"Restored from backup: {backup_name}")
            return {"ok": True, "result": res}
        except Exception as e:
            _logger.error(f"Restore failed: {e}", exc_info=True)
            return {"ok": False, "error": str(e)}

    def list_backups(self, backup_root: Optional[str] = None) -> Dict[str, Any]:
        """List available backups"""
        try:
            res = utility.list_backups(using=self.cfg.alias)
            return {"ok": True, "backups": res}
        except Exception as e:
            _logger.error(f"List backups failed: {e}", exc_info=True)
            return {"ok": False, "error": str(e)}

    # ==================== MAINTENANCE ====================
    
    def flush(self) -> None:
        """Flush data to storage"""
        if self._collection is None:
            return
        try:
            self._collection.flush()
            _logger.info(f"Flushed collection: {self.cfg.collection_name}")
        except Exception as e:
            _logger.error(f"Flush failed: {e}", exc_info=True)

    def compact(self) -> Dict[str, Any]:
        """Compact collection"""
        try:
            res = utility.compact(self.cfg.collection_name, using=self.cfg.alias)
            _logger.info(f"Compaction started for: {self.cfg.collection_name}")
            return {"ok": True, "result": res}
        except Exception as e:
            _logger.error(f"Compact failed: {e}", exc_info=True)
            return {"ok": False, "error": str(e)}

    def load_collection(self) -> None:
        """Load collection into memory"""
        if self._collection is None:
            self.ensure_collection()
        try:
            self._collection.load()
            _logger.info(f"Loaded collection: {self.cfg.collection_name}")
        except Exception as e:
            _logger.error(f"Load failed: {e}", exc_info=True)
            raise

    def release_collection(self) -> None:
        """Release collection from memory"""
        if self._collection is None:
            return
        try:
            self._collection.release()
            _logger.info(f"Released collection: {self.cfg.collection_name}")
        except Exception as e:
            _logger.error(f"Release failed: {e}", exc_info=True)

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        if self._collection is None:
            return {"error": "Collection not initialized"}
        try:
            stats = self._collection.num_entities
            return {
                "collection_name": self.cfg.collection_name,
                "num_entities": stats,
                "schema": {"dim": self.cfg.dim, "metric": self.cfg.metric_type, "index": self.cfg.index_type}
            }
        except Exception as e:
            _logger.error(f"Get stats failed: {e}", exc_info=True)
            return {"error": str(e)}

    # ==================== HEALTH & UTILITIES ====================
    
    def health(self) -> Dict[str, Any]:
        """Health check"""
        try:
            collections = utility.list_collections(using=self.cfg.alias)
            collection_exists = self.cfg.collection_name in collections
            
            stats = {}
            if collection_exists and self._collection:
                try:
                    stats = self.get_collection_stats()
                except Exception:
                    pass
            
            return {
                "ok": True,
                "connected": self._connected,
                "collection_exists": collection_exists,
                "collections": collections,
                "stats": stats
            }
        except Exception as e:
            _logger.error(f"Health check failed: {e}", exc_info=True)
            return {"ok": False, "error": str(e)}

    def set_ef(self, ef: int) -> None:
        """Update ef_search parameter"""
        self.cfg.ef_search = int(ef)
        _logger.info(f"Updated ef_search to {ef}")

    def close(self) -> None:
        """Close connection"""
        try:
            if self._collection:
                try:
                    self._collection.release()
                except Exception:
                    pass
            if connections and self._connected:
                try:
                    connections.disconnect(self.cfg.alias)
                    self._connected = False
                    _logger.info("Disconnected from Milvus")
                except Exception:
                    pass
        except Exception as e:
            _logger.error(f"Close failed: {e}", exc_info=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ==================== CLI ====================

def _cli_create(args):
    cfg = IndexerConfig(host=args.host, port=args.port, alias=args.alias, collection_name=args.collection, dim=args.dim)
    ops = MilvusIndexerOps(cfg)
    ops.ensure_collection(overwrite=args.overwrite)
    print(json.dumps({"ok": True, "collection": args.collection}))


def _cli_insert(args):
    cfg = IndexerConfig(host=args.host, port=args.port, alias=args.alias, collection_name=args.collection, dim=args.dim)
    ops = MilvusIndexerOps(cfg)
    
    records = []
    with open(args.file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception as e:
                _logger.warning(f"Failed to parse line: {e}")
    
    res = ops.insert(records, batch_size=args.batch_size)
    print(json.dumps(res, indent=2))


def _cli_search(args):
    cfg = IndexerConfig(host=args.host, port=args.port, alias=args.alias, collection_name=args.collection, dim=args.dim)
    ops = MilvusIndexerOps(cfg)
    query_vec = json.loads(args.vector)
    results = ops.search(query_vec, top_k=args.top_k, expr=args.expr)
    print(json.dumps(results, indent=2, ensure_ascii=False))


def _cli_stats(args):
    cfg = IndexerConfig(host=args.host, port=args.port, alias=args.alias, collection_name=args.collection, dim=args.dim)
    ops = MilvusIndexerOps(cfg)
    stats = ops.get_collection_stats()
    print(json.dumps(stats, indent=2))


def _cli_health(args):
    cfg = IndexerConfig(host=args.host, port=args.port, alias=args.alias, collection_name=args.collection, dim=args.dim)
    ops = MilvusIndexerOps(cfg)
    health = ops.health()
    print(json.dumps(health, indent=2))


def _cli_drop(args):
    cfg = IndexerConfig(host=args.host, port=args.port, alias=args.alias, collection_name=args.collection, dim=args.dim)
    ops = MilvusIndexerOps(cfg)
    ops.drop_collection()
    print(json.dumps({"ok": True, "message": f"Dropped {args.collection}"}))


def main():
    parser = argparse.ArgumentParser(prog="milvus_ops", description="Milvus indexer operations for RAG pipelines")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default="19530")
    parser.add_argument("--alias", default="default")
    parser.add_argument("--collection", default="rag_collection")
    parser.add_argument("--dim", type=int, default=384)
    
    subparsers = parser.add_subparsers(dest="cmd")
    
    # create
    p_create = subparsers.add_parser("create", help="Create collection")
    p_create.add_argument("--overwrite", action="store_true")
    p_create.set_defaults(func=_cli_create)
    
    # insert
    p_insert = subparsers.add_parser("insert", help="Insert records from JSONL")
    p_insert.add_argument("--file", required=True)
    p_insert.add_argument("--batch-size", type=int, default=512)
    p_insert.set_defaults(func=_cli_insert)
    
    # search
    p_search = subparsers.add_parser("search", help="Search by vector")
    p_search.add_argument("--vector", required=True, help="JSON array")
    p_search.add_argument("--top-k", type=int, default=10)
    p_search.add_argument("--expr", default=None, help="Filter expression")
    p_search.set_defaults(func=_cli_search)
    
    # stats
    p_stats = subparsers.add_parser("stats", help="Get collection stats")
    p_stats.set_defaults(func=_cli_stats)
    
    # health
    p_health = subparsers.add_parser("health", help="Health check")
    p_health.set_defaults(func=_cli_health)
    
    # drop
    p_drop = subparsers.add_parser("drop", help="Drop collection")
    p_drop.set_defaults(func=_cli_drop)
    
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    
    try:
        args.func(args)
    except Exception as e:
        _logger.error(f"Command failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()




'i'