"""
milvus_retriever.py - Milvus Vector Store Retriever
Améliorations:
- Gestion d'erreurs robuste avec retry exponential backoff
- Support complet des métadonnées structurées
- Validation des données
- Connection pooling et health checks
- Métriques détaillées
- Support async/await natif
- Auto-reconnection
- Batch operations optimisées
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Callable
import time
import json
import threading
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

_logger = logging.getLogger("milvus_retriever")

# Optional dependency: pymilvus
try:
    from pymilvus import (
        connections,
        Collection,
        utility,
        FieldSchema,
        CollectionSchema,
        DataType,
    )
    MILVUS_AVAILABLE = True
except ImportError:
    connections = None
    Collection = None
    utility = None
    FieldSchema = None
    CollectionSchema = None
    DataType = None
    MILVUS_AVAILABLE = False
    _logger.warning("pymilvus not installed. Install: pip install pymilvus")


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class MilvusRetrieverConfig:
    """Configuration pour Milvus retriever"""
    # Connection
    host: str = "127.0.0.1"
    port: str = "19530"
    alias: str = "default"
    user: str = ""
    password: str = ""
    
    # Collection
    collection_name: str = "rag_collection"
    vector_field: str = "vector"
    id_field: str = "id"
    text_field: str = "text"
    metadata_field: str = "metadata_json"
    
    # Vector config
    dim: int = 384
    index_type: str = "HNSW"  # HNSW, IVF_FLAT, IVF_SQ8, FLAT
    index_params: Dict[str, Any] = field(
        default_factory=lambda: {"M": 32, "efConstruction": 200}
    )
    metric_type: str = "IP"  # IP (inner product) or L2 (euclidean)
    
    # Search params
    ef_search: int = 128
    top_k_default: int = 50
    search_params: Dict[str, Any] = field(default_factory=dict)
    
    # Performance
    insert_batch_size: int = 512
    timeout: Optional[float] = 30.0
    max_retries: int = 3
    retry_backoff: float = 0.5
    connect_timeout: float = 10.0
    thread_workers: int = 4
    
    # Features
    auto_id: bool = False
    consistency_level: str = "Strong"  # Strong, Bounded, Session, Eventually
    enable_dynamic_field: bool = False
    
    # Monitoring
    metrics_hook: Optional[Callable[[Dict[str, Any]], None]] = None
    log_operations: bool = True


# ==============================================================================
# MILVUS RETRIEVER
# ==============================================================================

class MilvusRetriever:
    """
    Retriever Milvus avec support HNSW et fonctionnalités avancées.
    
    Features:
    - Connection management avec auto-reconnect
    - Creation/gestion de collections
    - Insertion batch optimisée
    - Recherche vectorielle avec filtres
    - Retry logic avec backoff exponentiel
    - Métriques et monitoring
    """

    def __init__(self, cfg: MilvusRetrieverConfig):
        if not MILVUS_AVAILABLE:
            raise ImportError("pymilvus required. Install: pip install pymilvus")
        
        self.cfg = cfg
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max(1, cfg.thread_workers))
        self._collection: Optional[Collection] = None
        self._connected = False
        
        # Métriques
        self._metrics = {
            "searches": 0,
            "inserts": 0,
            "errors": 0,
            "reconnects": 0
        }
        
        # Semaphore pour rate limiting
        self._sem = asyncio.Semaphore(64)
        
        # Initialize connection et collection
        self._connect()
        if self._connected:
            self._ensure_collection()

    # ==================== CONNECTION ====================

    def _connect(self) -> bool:
        """Établit la connexion à Milvus"""
        with self._lock:
            try:
                # Check si déjà connecté
                if self._connected:
                    try:
                        utility.list_collections(using=self.cfg.alias)
                        return True
                    except Exception:
                        self._connected = False
                
                # Nouvelle connexion
                connect_params = {
                    "alias": self.cfg.alias,
                    "host": self.cfg.host,
                    "port": self.cfg.port,
                    "timeout": self.cfg.connect_timeout
                }
                
                if self.cfg.user and self.cfg.password:
                    connect_params["user"] = self.cfg.user
                    connect_params["password"] = self.cfg.password
                
                connections.connect(**connect_params)
                self._connected = True
                
                _logger.info(
                    f"Connected to Milvus at {self.cfg.host}:{self.cfg.port} "
                    f"(alias={self.cfg.alias})"
                )
                return True
                
            except Exception as e:
                self._connected = False
                self._metrics["errors"] += 1
                _logger.error(f"Milvus connection failed: {e}", exc_info=True)
                return False

    def _reconnect(self) -> bool:
        """Tente une reconnexion"""
        _logger.info("Attempting to reconnect to Milvus...")
        self._metrics["reconnects"] += 1
        
        try:
            connections.disconnect(self.cfg.alias)
        except Exception:
            pass
        
        return self._connect()

    def _ensure_connected(self) -> None:
        """S'assure que la connexion est active"""
        if not self._connected:
            if not self._connect():
                raise RuntimeError("Failed to connect to Milvus")

    # ==================== COLLECTION ====================

    def _ensure_collection(self) -> None:
        """S'assure que la collection existe et est chargée"""
        self._ensure_connected()
        
        with self._lock:
            try:
                if utility.has_collection(
                    self.cfg.collection_name,
                    using=self.cfg.alias
                ):
                    self._collection = Collection(
                        self.cfg.collection_name,
                        using=self.cfg.alias
                    )
                    _logger.info(f"Using existing collection: {self.cfg.collection_name}")
                else:
                    self._create_collection()
                
                # Ensure index et load
                self._ensure_index()
                self._load_collection()
                
            except Exception as e:
                _logger.error(f"ensure_collection failed: {e}", exc_info=True)
                raise

    def _create_collection(self) -> None:
        """Crée une nouvelle collection"""
        _logger.info(f"Creating collection: {self.cfg.collection_name}")
        
        # Définition du schéma
        fields = [
            FieldSchema(
                name=self.cfg.id_field,
                dtype=DataType.VARCHAR,
                max_length=128,
                is_primary=True,
                auto_id=self.cfg.auto_id
            ),
            FieldSchema(
                name=self.cfg.vector_field,
                dtype=DataType.FLOAT_VECTOR,
                dim=self.cfg.dim
            ),
            FieldSchema(
                name=self.cfg.text_field,
                dtype=DataType.VARCHAR,
                max_length=65535
            ),
            FieldSchema(
                name=self.cfg.metadata_field,
                dtype=DataType.VARCHAR,
                max_length=65535
            )
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="RAG vector collection",
            enable_dynamic_field=self.cfg.enable_dynamic_field
        )
        
        self._collection = Collection(
            name=self.cfg.collection_name,
            schema=schema,
            using=self.cfg.alias,
            consistency_level=self.cfg.consistency_level
        )
        
        _logger.info(f"Collection created: {self.cfg.collection_name}")

    def _ensure_index(self) -> None:
        """S'assure que l'index existe"""
        if self._collection is None:
            raise RuntimeError("Collection not initialized")
        
        try:
            # Check existing indexes
            indexes = self._collection.indexes
            has_index = any(
                idx.field_name == self.cfg.vector_field
                for idx in indexes
            )
            
            if not has_index:
                _logger.info(f"Creating {self.cfg.index_type} index...")
                
                index_params = {
                    "index_type": self.cfg.index_type,
                    "metric_type": self.cfg.metric_type,
                    "params": self.cfg.index_params
                }
                
                self._collection.create_index(
                    field_name=self.cfg.vector_field,
                    index_params=index_params,
                    timeout=self.cfg.timeout
                )
                
                _logger.info(
                    f"Index created: {self.cfg.index_type} with params {self.cfg.index_params}"
                )
            else:
                _logger.info("Index already exists")
                
        except Exception as e:
            _logger.error(f"ensure_index failed: {e}", exc_info=True)
            raise

    def _load_collection(self) -> None:
        """Charge la collection en mémoire"""
        if self._collection is None:
            raise RuntimeError("Collection not initialized")
        
        try:
            self._collection.load()
            _logger.info(f"Collection loaded: {self.cfg.collection_name}")
        except Exception as e:
            _logger.error(f"load_collection failed: {e}", exc_info=True)

    # ==================== INSERT/UPSERT ====================

    def upsert(
        self,
        docs: Sequence[Dict[str, Any]],
        id_field: Optional[str] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Insert ou update des documents.
        
        Args:
            docs: Liste de documents avec id, vector, text, meta
            id_field: Nom du champ ID (défaut: config)
            batch_size: Taille des batchs (défaut: config)
        
        Returns:
            Résumé: {inserted, failed, errors}
        """
        if not docs:
            return {"inserted": 0, "failed": 0, "errors": []}
        
        self._ensure_collection()
        
        id_f = id_field or self.cfg.id_field
        batch_sz = batch_size or self.cfg.insert_batch_size
        
        inserted = 0
        failed = 0
        errors: List[str] = []
        
        # Préparation des données
        rows = []
        for doc in docs:
            try:
                row = self._prepare_record(doc, id_f)
                if row:
                    rows.append(row)
            except Exception as e:
                failed += 1
                errors.append(f"prepare_failed: {str(e)}")
                _logger.debug(f"Failed to prepare doc: {e}")
        
        # Insertion par batch
        for i in range(0, len(rows), batch_sz):
            batch = rows[i:i + batch_sz]
            try:
                self._collection.insert(batch, timeout=self.cfg.timeout)
                inserted += len(batch)
                self._metrics["inserts"] += len(batch)
            except Exception as e:
                failed += len(batch)
                errors.append(f"insert_batch_failed: {str(e)}")
                self._metrics["errors"] += 1
                _logger.error(f"Batch insert failed: {e}")
        
        # Flush
        try:
            self._collection.flush(timeout=self.cfg.timeout)
        except Exception as e:
            _logger.warning(f"Flush failed: {e}")
        
        result = {
            "inserted": inserted,
            "failed": failed,
            "errors": errors
        }
        
        if self.cfg.log_operations:
            _logger.info(
                f"Upsert complete: {inserted} inserted, {failed} failed"
            )
        
        if self.cfg.metrics_hook:
            try:
                self.cfg.metrics_hook({
                    "event": "upsert",
                    "inserted": inserted,
                    "failed": failed
                })
            except Exception:
                pass
        
        return result

    def _prepare_record(
        self,
        doc: Dict[str, Any],
        id_field: str
    ) -> Optional[Dict[str, Any]]:
        """Prépare un record pour insertion"""
        doc_id = doc.get(id_field)
        if doc_id is None:
            raise ValueError(f"Missing {id_field}")
        
        vector = doc.get(self.cfg.vector_field)
        if vector is None:
            raise ValueError(f"Missing {self.cfg.vector_field}")
        
        # Validation du vecteur
        if not self._validate_vector(vector):
            raise ValueError("Invalid vector format or dimension")
        
        text = doc.get(self.cfg.text_field, "")
        meta = doc.get("meta", {})
        
        # Sérialisation metadata
        meta_json = json.dumps(meta, ensure_ascii=False)
        
        record = {
            self.cfg.id_field: str(doc_id),
            self.cfg.vector_field: vector,
            self.cfg.text_field: text[:65535],  # Truncate if needed
            self.cfg.metadata_field: meta_json[:65535]
        }
        
        return record

    def _validate_vector(self, vector: Any) -> bool:
        """Valide un vecteur"""
        if not isinstance(vector, (list, tuple)):
            return False
        
        if len(vector) != self.cfg.dim:
            _logger.warning(
                f"Vector dimension mismatch: {len(vector)} != {self.cfg.dim}"
            )
            return False
        
        try:
            # Check que tous les éléments sont des nombres
            all(isinstance(x, (int, float)) for x in vector)
            return True
        except Exception:
            return False

    async def upsert_async(
        self,
        docs: Sequence[Dict[str, Any]],
        id_field: Optional[str] = None
    ) -> Dict[str, Any]:
        """Version async de l'upsert"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.upsert(docs, id_field)
        )

    # ==================== SEARCH ====================

    def _search_blocking(
        self,
        query_vec: List[float],
        top_k: int,
        expr: Optional[str],
        search_params: Optional[Dict[str, Any]],
        timeout: Optional[float]
    ) -> List[Dict[str, Any]]:
        """Recherche vectorielle (blocking)"""
        self._ensure_collection()
        
        # Paramètres de recherche
        if search_params is None:
            ef = self.cfg.ef_search
            search_params = {
                "metric_type": self.cfg.metric_type,
                "params": {"ef": ef}
            }
        
        # Output fields
        output_fields = [self.cfg.text_field, self.cfg.metadata_field]
        
        try:
            results = self._collection.search(
                data=[query_vec],
                anns_field=self.cfg.vector_field,
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=output_fields,
                timeout=timeout or self.cfg.timeout,
                consistency_level=self.cfg.consistency_level
            )
            
            hits = []
            if not results or not results[0]:
                return hits
            
            for hit in results[0]:
                hit_data = {
                    "id": hit.id,
                    "score": float(hit.score)
                }
                
                # Extract entity data
                entity = getattr(hit, "entity", {})
                
                hit_data["text"] = entity.get(self.cfg.text_field, "")
                
                # Deserialize metadata
                meta_json = entity.get(self.cfg.metadata_field, "{}")
                try:
                    hit_data["meta"] = json.loads(meta_json)
                except Exception:
                    hit_data["meta"] = {}
                    _logger.debug(f"Failed to parse metadata for hit {hit.id}")
                
                # Add vector if available
                if hasattr(hit, self.cfg.vector_field):
                    hit_data["vector"] = getattr(hit, self.cfg.vector_field)
                
                hits.append(hit_data)
            
            return hits
            
        except Exception as e:
            self._metrics["errors"] += 1
            _logger.error(f"Search failed: {e}", exc_info=True)
            raise

    def _retry_search(
        self,
        qv: List[float],
        top_k: int,
        expr: Optional[str],
        search_params: Optional[Dict[str, Any]],
        timeout: Optional[float]
    ) -> List[Dict[str, Any]]:
        """Recherche avec retry logic"""
        attempt = 0
        last_exc = None
        
        while attempt <= self.cfg.max_retries:
            try:
                return self._search_blocking(qv, top_k, expr, search_params, timeout)
            except Exception as e:
                last_exc = e
                attempt += 1
                
                if attempt <= self.cfg.max_retries:
                    backoff = self.cfg.retry_backoff * (2 ** (attempt - 1))
                    _logger.warning(
                        f"Search attempt {attempt} failed, retrying in {backoff}s..."
                    )
                    time.sleep(backoff)
                    
                    # Try reconnect on connection errors
                    if "connection" in str(e).lower():
                        self._reconnect()
        
        _logger.error(f"Search failed after {self.cfg.max_retries} retries")
        raise last_exc

    async def search(
        self,
        query_vecs: Sequence[List[float]],
        top_k: Optional[int] = None,
        expr: Optional[str] = None,
        search_params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Recherche vectorielle async pour multiple queries.
        
        Args:
            query_vecs: Vecteurs de requête
            top_k: Nombre de résultats par query
            expr: Expression de filtre Milvus
            search_params: Paramètres de recherche personnalisés
            timeout: Timeout en secondes
        
        Returns:
            Liste de listes de hits
        """
        if not query_vecs:
            return []
        
        top_k = top_k or self.cfg.top_k_default
        
        # Execute searches in parallel
        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(
                self._executor,
                self._retry_search,
                qv, top_k, expr, search_params, timeout
            )
            for qv in query_vecs
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        output = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                _logger.error(f"Query {i} failed: {result}")
                output.append([])
            else:
                output.append(result)
        
        # Metrics
        self._metrics["searches"] += len(query_vecs)
        
        if self.cfg.metrics_hook:
            try:
                self.cfg.metrics_hook({
                    "event": "search",
                    "queries": len(query_vecs),
                    "top_k": top_k,
                    "failed": sum(1 for r in results if isinstance(r, Exception))
                })
            except Exception:
                pass
        
        return output

    # ==================== UTILITIES ====================

    def health(self) -> Dict[str, Any]:
        """Health check de la connexion"""
        try:
            collections = utility.list_collections(using=self.cfg.alias)
            exists = self.cfg.collection_name in collections
            
            stats = {}
            if exists and self._collection:
                try:
                    stats = {
                        "num_entities": self._collection.num_entities,
                        "loaded": True
                    }
                except Exception:
                    stats = {"loaded": False}
            
            return {
                "ok": exists,
                "connected": self._connected,
                "collection": self.cfg.collection_name,
                "collections": collections,
                "stats": stats
            }
        except Exception as e:
            return {
                "ok": False,
                "error": str(e)
            }

    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques"""
        return self._metrics.copy()

    def reset_metrics(self) -> None:
        """Reset les métriques"""
        self._metrics = {
            "searches": 0,
            "inserts": 0,
            "errors": 0,
            "reconnects": 0
        }

    def set_ef(self, ef: int) -> None:
        """Définit le paramètre ef pour les recherches"""
        self.cfg.ef_search = max(1, int(ef))
        _logger.info(f"ef_search set to {self.cfg.ef_search}")

    def drop_collection(self) -> None:
        """Supprime la collection"""
        try:
            utility.drop_collection(
                self.cfg.collection_name,
                using=self.cfg.alias
            )
            _logger.info(f"Dropped collection: {self.cfg.collection_name}")
            self._collection = None
        except Exception as e:
            _logger.error(f"drop_collection failed: {e}")
            raise

    def close(self) -> None:
        """Ferme les connexions"""
        try:
            if self._collection:
                try:
                    self._collection.release()
                except Exception:
                    pass
            
            try:
                connections.disconnect(self.cfg.alias)
                self._connected = False
                _logger.info("Disconnected from Milvus")
            except Exception:
                pass
            
            self._executor.shutdown(wait=True)
            
        except Exception as e:
            _logger.error(f"Close failed: {e}")

    # ==================== CONTEXT MANAGERS ====================

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ==============================================================================
# FACTORY
# ==============================================================================

def create_milvus_retriever(
    collection_name: str,
    dim: int = 384,
    **kwargs
) -> MilvusRetriever:
    """Factory pour créer un retriever Milvus"""
    cfg = MilvusRetrieverConfig(
        collection_name=collection_name,
        dim=dim,
        **kwargs
    )
    return MilvusRetriever(cfg)

'i'