from __future__ import annotations
import asyncio
import time
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, Iterable, List, Optional, Protocol, Union

# ---------- Core Data Structures for Granularity and Clarity ----------

@dataclass
class Record:
    """Structure canonique pour une donnée traitée dans le pipeline."""
    # Clé unique pour l'identification et la déduplication
    id: Optional[str] = field(default=None) 
    # Contenu principal (ex: texte, JSON sérialisé)
    content: str = field(default="")
    # Métadonnées pour le traçage, le hachage, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class PipelineContext:
    """Contexte d'exécution et de configuration par connecteur."""
    connector_id: str
    run_id: str = field(default_factory=lambda: hashlib.sha256(str(time.time()).encode()).hexdigest()[:8])
    source_config: Dict[str, Any] = field(default_factory=dict)


# ---------- Protocols / Interfaces (Modularité et Scalabilité) ----------

class SourceConnector(Protocol):
    """Interface pour lire des données brutes de la source."""
    id: str
    async def read_async(self) -> AsyncIterator[Any]:
        ...

class CheckpointStore(Protocol):
    """Interface pour stocker l'état du pipeline."""
    def get(self, key: str) -> Optional[Any]:
        ...
    def set(self, key: str, value: Any) -> None:
        ...

class Deduplicator(Protocol):
    """Interface pour gérer la déduplication et le Change Data Capture (CDC)."""
    async def is_duplicate(self, record: Record) -> bool:
        ...
    async def mark(self, record: Record) -> None:
        ...

class Sink(Protocol):
    """Interface pour consommer les lots de données normalisées."""
    async def consume_batch(self, batch: List[Record]) -> None:
        ...

class DataPipelineTracer(Protocol):
    """Interface unifiée pour les métriques et la journalisation des erreurs (Industriel)."""
    async def log_event(self, event_name: str, **kwargs: Any) -> None:
        ...
    async def log_error(self, event_type: str, message: str, details: Dict[str, Any], raw_data: Any = None) -> None:
        """event_type: 'read_error', 'normalization_error', 'flush_error'"""
        ...

# ---------- Config ----------

@dataclass
class LoaderConfig:
    """Configuration granulaire pour le chargement et la résilience."""
    batch_size: int = 256
    max_batch_time: float = 1.0
    concurrency: int = 4
    retry_attempts: int = 3
    retry_backoff: float = 0.5
    checkpoint_interval_batches: int = 10
    dead_letter_path: Optional[str] = None
    # Permet de définir un champ de Record pour l'ID si le champ par défaut n'est pas utilisé
    id_field_key: str = "id" 

# ---------- Utilities ----------

def _content_hash(text: str) -> str:
    """Génère un hachage SHA256 du contenu."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _now_ts() -> float:
    """Timestamp actuel."""
    return time.time()

# ---------- Simple implementations ----------

class FileCheckpointStore:
    """Implémentation simple d'un CheckpointStore basé sur un fichier JSON."""
    def __init__(self, path: str = "loader_checkpoints.json"):
        self.path = path
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self._store = json.load(f)
        except Exception:
            self._store = {}

    def get(self, key: str) -> Optional[Any]:
        return self._store.get(key)

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._store, f, ensure_ascii=False, indent=2)
        except Exception:
            # Enregistrement silencieux de l'échec de l'écriture
            pass

class LRUDeduplicator:
    """Implémentation simple de Deduplicator basée sur un cache LRU en mémoire."""
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self._cache = {}
        self._order = []

    def _get_key(self, record: Record) -> str:
        """Détermine la clé de déduplication à partir de l'enregistrement."""
        key = record.id
        if not key:
            key = record.metadata.get("content_hash")
        if not key:
            # Hachage du contenu (ou de ses premiers 4096 chars) en dernier recours
            key = _content_hash(record.content[:4096])
        return key

    async def is_duplicate(self, record: Record) -> bool:
        key = self._get_key(record)
        if key in self._cache:
            # Met à jour l'ordre LRU (déplace à la fin)
            try:
                self._order.remove(key)
            except ValueError:
                pass
            self._order.append(key)
            return True
        return False

    async def mark(self, record: Record) -> None:
        key = self._get_key(record)
        if key not in self._cache:
            self._cache[key] = True
            self._order.append(key)
            # Gestion de la capacité LRU
            if len(self._order) > self.capacity:
                old = self._order.pop(0)
                self._cache.pop(old, None)

class MemorySink:
    """Implémentation simple de Sink qui stocke les enregistrements en mémoire."""
    def __init__(self):
        self.storage: List[Record] = []

    async def consume_batch(self, batch: List[Record]) -> None:
        """Simule une opération d'E/S asynchrone pour écrire le lot."""
        await asyncio.sleep(0.01) 
        self.storage.extend(batch)

class NullTracer:
    """Implémentation vide du Tracer pour les cas où le traçage n'est pas nécessaire."""
    async def log_event(self, event_name: str, **kwargs: Any) -> None:
        pass
    async def log_error(self, event_type: str, message: str, details: Dict[str, Any], raw_data: Any = None) -> None:
        print(f"[{event_type.upper()} ERROR] {message}: {details} (Raw: {str(raw_data)[:50]}...)")

# ---------- Loader core ----------

class Loader:
    """Cœur du pipeline de chargement de données asynchrone."""
    def __init__(
        self,
        cfg: LoaderConfig,
        connectors: List[SourceConnector],
        normalizer: Any,  # doit implémenter normalize_stream(raw_iterable, context)
        sink: Sink,
        checkpoint: Optional[CheckpointStore] = None,
        deduplicator: Optional[Deduplicator] = None,
        pre_hooks: Optional[List[Callable[[Any], Any]]] = None,
        post_hooks: Optional[List[Callable[[Record], Record]]] = None,
        tracer: Optional[DataPipelineTracer] = None,
    ):
        self.cfg = cfg
        self.connectors = connectors
        self.normalizer = normalizer
        self.sink = sink
        self.checkpoint = checkpoint or FileCheckpointStore()
        self.dedup = deduplicator
        self.pre_hooks = pre_hooks or []
        self.post_hooks = post_hooks or []
        # Le traçage structuré est au cœur de l'aspect "industriel"
        self.tracer: DataPipelineTracer = tracer or NullTracer()
        self._sem = asyncio.Semaphore(cfg.concurrency)

    async def _process_connector(self, connector: SourceConnector, ctx_data: Dict[str, Any]):
        """Traite les données d'un seul connecteur en respectant les limites de concurrence."""
        ctx = PipelineContext(connector_id=connector.id, source_config=ctx_data)
        await self.tracer.log_event("connector_start", connector_id=ctx.connector_id, run_id=ctx.run_id)

        batch: List[Record] = []
        last_flush = _now_ts()
        processed_batches = 0
        dead_letter = []
        records_read = 0

        # Utilisation du sémaphore pour contrôler la concurrence globale
        async with self._sem:
            async for raw in connector.read_async():
                records_read += 1
                try:
                    # 1. Pré-Hooks (traitement brut)
                    for h in self.pre_hooks:
                        raw = h(raw)

                    # 2. Normalisation (peut générer 0 à N Records)
                    normalized_records = []
                    # Le normaliseur est supposé renvoyer une liste ou un générateur d'objets Record (ou convertibles)
                    for rec in self.normalizer.normalize_stream([raw], context=ctx_data):
                        # Assure que le résultat est un objet Record pour la cohérence
                        record_obj = Record(**rec) if isinstance(rec, dict) else rec

                        # Calculer le hachage de contenu pour la déduplication et les métadonnées
                        if "content_hash" not in record_obj.metadata:
                            record_obj.metadata["content_hash"] = _content_hash(record_obj.content[:4096])

                        # 3. Post-Hooks (traitement du Record)
                        for ph in self.post_hooks:
                            record_obj = ph(record_obj)

                        # 4. Vérification de la déduplication
                        if self.dedup and await self.dedup.is_duplicate(record_obj):
                            await self.tracer.log_event("dedup_skipped", connector_id=ctx.connector_id, hash=record_obj.metadata["content_hash"])
                            continue

                        normalized_records.append(record_obj)

                    # 5. Gestion des lots
                    for rec in normalized_records:
                        batch.append(rec)

                        now = _now_ts()
                        # Déclencheur: Taille du lot OU temps écoulé
                        if len(batch) >= self.cfg.batch_size or (now - last_flush) >= self.cfg.max_batch_time:
                            await self._flush_with_retries(batch, ctx)
                            batch = []
                            last_flush = now
                            processed_batches += 1
                            
                            # Point de contrôle (Checkpoint)
                            if processed_batches % self.cfg.checkpoint_interval_batches == 0:
                                self._checkpoint_progress(ctx)

                except Exception as e:
                    # Gestion granulaire des erreurs de normalisation/pipeline
                    error_details = {"exception": str(e), "record_read_count": records_read}
                    await self.tracer.log_error(
                        "normalization_error", 
                        f"Échec de la normalisation/du hook pour l'enregistrement #{records_read}", 
                        error_details, 
                        raw_data=raw
                    )
                    dead_letter.append(error_details | {"raw_data": str(raw)})
                    continue

        # 6. Flush final
        if batch:
            await self._flush_with_retries(batch, ctx)
            self._checkpoint_progress(ctx)
        
        await self.tracer.log_event("connector_end", connector_id=ctx.connector_id, total_read=records_read, total_batches=processed_batches)

        # 7. Persistance de la Dead Letter Queue
        if dead_letter and self.cfg.dead_letter_path:
            try:
                with open(self.cfg.dead_letter_path, "a", encoding="utf-8") as f:
                    for d in dead_letter:
                        f.write(json.dumps(d, ensure_ascii=False) + "\n")
            except Exception as e:
                await self.tracer.log_error("dlq_write_error", f"Échec de l'écriture DLQ: {str(e)}", {"path": self.cfg.dead_letter_path})


    async def _flush_with_retries(self, batch: List[Record], ctx: PipelineContext):
        """Tente d'envoyer le lot au Sink avec une politique de réessai."""
        attempt = 0
        batch_size = len(batch)
        while attempt <= self.cfg.retry_attempts:
            try:
                await self.sink.consume_batch(batch)
                
                # Marquer les entrées pour la déduplication après le succès
                if self.dedup:
                    for rec in batch:
                        await self.dedup.mark(rec)
                        
                await self.tracer.log_event("batch_sent", connector_id=ctx.connector_id, size=batch_size, attempt=attempt + 1)
                return  # Succès

            except Exception as e:
                attempt += 1
                error_details = {"exception": str(e), "attempt": attempt, "batch_size": batch_size}
                
                await self.tracer.log_error(
                    "flush_error", 
                    f"Échec de l'envoi du lot, tentative {attempt}", 
                    error_details, 
                    batch_data=batch
                )

                if attempt <= self.cfg.retry_attempts:
                    await asyncio.sleep(self.cfg.retry_backoff * attempt)
                else:
                    break  # Échec après la dernière tentative

        # Abandon du lot après toutes les tentatives
        await self.tracer.log_event("flush_giveup", connector_id=ctx.connector_id, size=batch_size)


    def _checkpoint_progress(self, ctx: PipelineContext):
        """Enregistre l'état du connecteur dans le CheckpointStore."""
        key = f"checkpoint:{ctx.connector_id}"
        # Sauvegarde du contexte (qui peut contenir des marqueurs de progression)
        value = {"ts": _now_ts(), "source_config": ctx.source_config}
        try:
            self.checkpoint.set(key, value)
        except Exception as e:
            # Erreur silencieuse de checkpoint, journalisée uniquement si un Tracer est présent
            asyncio.create_task(self.tracer.log_error("checkpoint_error", f"Échec du checkpoint: {str(e)}", {"connector_id": ctx.connector_id}))


    async def run(self, ctx_per_connector: Optional[Dict[str, Dict[str, Any]]] = None):
        """Lance l'exécution de tous les connecteurs en parallèle."""
        ctx_per_connector = ctx_per_connector or {}
        tasks = []
        for conn in self.connectors:
            conn_id = getattr(conn, "id", conn.__class__.__name__)
            ctx_data = ctx_per_connector.get(conn_id, {})
            # Crée une tâche pour chaque connecteur
            tasks.append(asyncio.create_task(self._process_connector(conn, ctx_data)))
        
        await asyncio.gather(*tasks)

# ---------- Example connectors for testing ----------

class FileConnector:
    """Lit les lignes d'un fichier de manière asynchrone."""
    def __init__(self, id: str, path: str, encoding: str = "utf-8"):
        self.id = id
        self.path = path
        self.encoding = encoding

    async def read_async(self) -> AsyncIterator[Dict[str, Any]]:
        loop = asyncio.get_event_loop()
        
        def _read_lines():
            """Lecture synchrone dans un thread pool pour éviter de bloquer l'event loop."""
            try:
                with open(self.path, "r", encoding=self.encoding) as f:
                    return [line.rstrip("\n") for line in f]
            except Exception as e:
                # Signale une erreur de lecture/fichier
                return [{"error": str(e), "path": self.path}]

        lines = await loop.run_in_executor(None, _read_lines)
        
        for line_data in lines:
            if isinstance(line_data, dict) and "error" in line_data:
                # Cette erreur de lecture sera gérée par le Tracer dans _process_connector
                raise IOError(line_data["error"])
            
            # La donnée brute est empaquetée.
            yield {"source_tag": "text_line", "data": line_data}

            'ig'