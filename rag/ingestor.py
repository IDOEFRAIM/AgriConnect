"""
ingestion_pipeline.py - Pipeline d'Ingestion RAG Amélioré
Améliorations:
- Gestion d'erreurs robuste avec retry logic
- Validation des données
- Métriques et monitoring
- Async support pour meilleure performance
- Déduplication des chunks
- Memory-efficient streaming
"""
from __future__ import annotations
import uuid
import logging
import time
import hashlib
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Generator, Set
from collections import defaultdict
import re

# Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
_logger = logging.getLogger("ingestion_pipeline")


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

@dataclass
class NormalizerConfig:
    """Configuration pour le normalizer"""
    chunk_size_tokens: int = 200
    chunk_overlap_tokens: int = 50
    language_detection: bool = False
    min_chunk_length: int = 20
    max_chunk_length: int = 1000
    normalize_whitespace: bool = True
    lowercase: bool = False
    keep_metadata_fields: Tuple[str, ...] = ("source", "date", "region", "language")
    id_prefix: str = "doc"
    enable_deduplication: bool = True


@dataclass
class IngestionConfig:
    """Configuration pour le pipeline d'ingestion"""
    batch_size: int = 32
    max_retries: int = 3
    retry_delay_s: float = 1.0
    enable_async: bool = False
    log_progress_every: int = 100
    validate_vectors: bool = True
    metrics_enabled: bool = True


# ==============================================================================
# 2. NORMALIZER
# ==============================================================================

class Normalizer:
    """Normalise et découpe les documents en chunks"""
    
    def __init__(self, cfg: NormalizerConfig):
        self.cfg = cfg
        self._seen_hashes: Set[str] = set() if cfg.enable_deduplication else None
        self._stats = {
            "total_docs": 0,
            "total_chunks": 0,
            "duplicates_skipped": 0,
            "errors": 0
        }

    def normalize_stream(
        self,
        raw_items: Iterable[Dict[str, Any]]
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream de normalisation des documents"""
        
        for raw in raw_items:
            try:
                self._stats["total_docs"] += 1
                
                # Validation
                if not self._validate_raw_item(raw):
                    _logger.warning(f"Invalid item skipped: {raw.get('raw_id')}")
                    continue
                
                # Génération de l'ID document
                doc_id = self._generate_doc_id(raw)
                text = raw.get("text", "").strip()
                
                if not text:
                    _logger.debug(f"Empty text for doc {doc_id}")
                    continue
                
                # Nettoyage du texte
                if self.cfg.normalize_whitespace:
                    text = self._normalize_whitespace(text)
                
                if self.cfg.lowercase:
                    text = text.lower()
                
                # Découpage en chunks
                chunks = self._create_chunks(text)
                
                # Métadonnées
                metadata = self._extract_metadata(raw)
                
                # Génération des chunks
                for chunk_index, chunk_text in enumerate(chunks):
                    # Vérification longueur
                    if len(chunk_text) < self.cfg.min_chunk_length:
                        continue
                    
                    # Déduplication
                    text_hash = self._hash_text(chunk_text)
                    if self._seen_hashes is not None:
                        if text_hash in self._seen_hashes:
                            self._stats["duplicates_skipped"] += 1
                            continue
                        self._seen_hashes.add(text_hash)
                    
                    chunk_id = f"{doc_id}_c{chunk_index}"
                    
                    yield {
                        "id": doc_id,
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_index,
                        "text": chunk_text,
                        "meta": metadata,
                        "text_hash": text_hash,
                        "length_tokens": len(chunk_text.split()),
                        "source": raw.get("source", "unknown")
                    }
                    
                    self._stats["total_chunks"] += 1
                    
            except Exception as e:
                self._stats["errors"] += 1
                _logger.exception(f"Normalization failed for item {raw.get('raw_id')}: {e}")

    def _validate_raw_item(self, raw: Dict[str, Any]) -> bool:
        """Valide un élément brut"""
        if not isinstance(raw, dict):
            return False
        if "text" not in raw or not raw["text"]:
            return False
        return True

    def _generate_doc_id(self, raw: Dict[str, Any]) -> str:
        """Génère un ID unique pour le document"""
        raw_id = raw.get('raw_id') or raw.get('id') or uuid.uuid4()
        return f"{self.cfg.id_prefix}_{raw_id}"

    def _normalize_whitespace(self, text: str) -> str:
        """Normalise les espaces blancs"""
        # Remplace multiples espaces par un seul
        text = re.sub(r'\s+', ' ', text)
        # Nettoie les espaces en début/fin
        return text.strip()

    def _create_chunks(self, text: str) -> List[str]:
        """Découpe le texte en chunks avec overlap"""
        tokens = text.split()
        
        if not tokens:
            return []
        
        chunks = []
        win_size = max(1, self.cfg.chunk_size_tokens)
        step = max(1, win_size - self.cfg.chunk_overlap_tokens)
        
        for start in range(0, len(tokens), step):
            end = min(len(tokens), start + win_size)
            chunk_text = " ".join(tokens[start:end]).strip()
            
            # Limite la longueur maximale
            if len(chunk_text.split()) > self.cfg.max_chunk_length:
                chunk_text = " ".join(chunk_text.split()[:self.cfg.max_chunk_length])
            
            if chunk_text:
                chunks.append(chunk_text)
            
            if end >= len(tokens):
                break
        
        return chunks

    def _extract_metadata(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Extrait les métadonnées pertinentes"""
        raw_meta = raw.get("metadata", {})
        metadata = {}
        
        for field in self.cfg.keep_metadata_fields:
            if field in raw_meta:
                metadata[field] = raw_meta[field]
        
        return metadata

    def _hash_text(self, text: str) -> str:
        """Calcule un hash SHA256 du texte"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de normalisation"""
        return self._stats.copy()

    def reset_stats(self) -> None:
        """Réinitialise les statistiques"""
        self._stats = {
            "total_docs": 0,
            "total_chunks": 0,
            "duplicates_skipped": 0,
            "errors": 0
        }
        if self._seen_hashes is not None:
            self._seen_hashes.clear()


# ==============================================================================
# 3. MOCK COMPONENTS
# ==============================================================================

class MockEmbedder:
    """Simule le service de génération d'embeddings"""
    
    def __init__(self, vector_dim: int = 128, latency_ms: float = 10.0):
        self.vector_dim = vector_dim
        self.latency_ms = latency_ms
        _logger.info(f"MockEmbedder initialisé (dim={vector_dim})")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Génère des vecteurs pour les textes"""
        if not texts:
            return []
        
        # Simule la latence
        time.sleep(self.latency_ms / 1000.0)
        
        vectors = []
        for text in texts:
            # Vecteur basé sur un hash du texte
            base_hash = hash(text) % 10000
            vector = [(base_hash + i) / 10000.0 for i in range(self.vector_dim)]
            vectors.append(vector)
        
        return vectors

    async def embed_texts_async(self, texts: List[str]) -> List[List[float]]:
        """Version async de l'embedding"""
        await asyncio.sleep(self.latency_ms / 1000.0)
        return self.embed_texts(texts)


class MockVectorStore:
    """Simule une base de données vectorielle"""
    
    def __init__(self, index_name: str):
        self.index_name = index_name
        self.data_store: Dict[str, Dict[str, Any]] = {}
        self._upsert_count = 0
        _logger.info(f"MockVectorStore '{index_name}' initialisé")

    def upsert_records(self, records: List[Dict[str, Any]]) -> bool:
        """Insère ou met à jour les enregistrements"""
        try:
            for rec in records:
                if "chunk_id" not in rec or "vector" not in rec:
                    _logger.warning("Record invalide ignoré")
                    continue
                
                point_id = rec["chunk_id"]
                self.data_store[point_id] = rec
                self._upsert_count += 1
            
            _logger.info(f"Upserted {len(records)} records dans '{self.index_name}'")
            return True
            
        except Exception as e:
            _logger.error(f"Upsert failed: {e}")
            return False

    async def upsert_records_async(self, records: List[Dict[str, Any]]) -> bool:
        """Version async de l'upsert"""
        await asyncio.sleep(0.01)  # Simule I/O
        return self.upsert_records(records)

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques"""
        return {
            "total_records": len(self.data_store),
            "total_upserts": self._upsert_count,
            "index_name": self.index_name
        }


# ==============================================================================
# 4. INGESTION PIPELINE
# ==============================================================================

class IngestionPipeline:
    """Pipeline d'ingestion complet avec gestion d'erreurs et métriques"""
    
    def __init__(
        self,
        normalizer_cfg: NormalizerConfig,
        ingestion_cfg: IngestionConfig,
        vector_store: MockVectorStore,
        embedder: MockEmbedder
    ):
        self.normalizer = Normalizer(normalizer_cfg)
        self.cfg = ingestion_cfg
        self.vector_store = vector_store
        self.embedder = embedder
        
        self._metrics = {
            "batches_processed": 0,
            "batches_failed": 0,
            "total_processing_time_s": 0.0,
            "total_chunks": 0
        }

    def run(self, raw_items: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """Exécute le pipeline de façon synchrone"""
        _logger.info("Démarrage du pipeline d'ingestion")
        start_time = time.time()
        
        chunk_stream = self.normalizer.normalize_stream(raw_items)
        batch: List[Dict[str, Any]] = []
        
        for chunk in chunk_stream:
            batch.append(chunk)
            
            # Log progression
            if self.cfg.log_progress_every > 0 and \
               self._metrics["total_chunks"] % self.cfg.log_progress_every == 0 and \
               self._metrics["total_chunks"] > 0:
                _logger.info(f"Progression: {self._metrics['total_chunks']} chunks traités")
            
            # Traitement par batch
            if len(batch) >= self.cfg.batch_size:
                self._process_batch_with_retry(batch)
                batch = []

        # Dernier batch
        if batch:
            self._process_batch_with_retry(batch)
        
        # Statistiques finales
        elapsed = time.time() - start_time
        self._metrics["total_processing_time_s"] = elapsed
        
        result = self._build_result()
        self._log_summary(result)
        
        return result

    async def run_async(self, raw_items: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """Exécute le pipeline de façon asynchrone"""
        _logger.info("Démarrage du pipeline d'ingestion (async)")
        start_time = time.time()
        
        chunk_stream = self.normalizer.normalize_stream(raw_items)
        batch: List[Dict[str, Any]] = []
        
        for chunk in chunk_stream:
            batch.append(chunk)
            
            if len(batch) >= self.cfg.batch_size:
                await self._process_batch_async_with_retry(batch)
                batch = []

        if batch:
            await self._process_batch_async_with_retry(batch)
        
        elapsed = time.time() - start_time
        self._metrics["total_processing_time_s"] = elapsed
        
        result = self._build_result()
        self._log_summary(result)
        
        return result

    def _process_batch_with_retry(self, batch: List[Dict[str, Any]]) -> bool:
        """Traite un batch avec retry logic"""
        for attempt in range(self.cfg.max_retries):
            try:
                success = self._process_batch(batch)
                if success:
                    self._metrics["batches_processed"] += 1
                    self._metrics["total_chunks"] += len(batch)
                    return True
                
            except Exception as e:
                _logger.error(f"Batch processing failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.cfg.max_retries - 1:
                    time.sleep(self.cfg.retry_delay_s * (attempt + 1))
                else:
                    self._metrics["batches_failed"] += 1
                    return False
        
        return False

    async def _process_batch_async_with_retry(self, batch: List[Dict[str, Any]]) -> bool:
        """Version async du traitement avec retry"""
        for attempt in range(self.cfg.max_retries):
            try:
                success = await self._process_batch_async(batch)
                if success:
                    self._metrics["batches_processed"] += 1
                    self._metrics["total_chunks"] += len(batch)
                    return True
                
            except Exception as e:
                _logger.error(f"Async batch processing failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.cfg.max_retries - 1:
                    await asyncio.sleep(self.cfg.retry_delay_s * (attempt + 1))
                else:
                    self._metrics["batches_failed"] += 1
                    return False
        
        return False

    def _process_batch(self, batch: List[Dict[str, Any]]) -> bool:
        """Traite un batch: embedding + stockage"""
        if not batch:
            return True
        
        # 1. Extraction des textes
        texts = [item["text"] for item in batch]
        
        # 2. Embedding
        _logger.debug(f"Embedding de {len(texts)} textes")
        vectors = self.embedder.embed_texts(texts)
        
        if len(vectors) != len(batch):
            _logger.error(f"Vector count mismatch: {len(vectors)} != {len(batch)}")
            return False
        
        # 3. Validation des vecteurs
        if self.cfg.validate_vectors:
            if not self._validate_vectors(vectors):
                _logger.error("Vector validation failed")
                return False
        
        # 4. Préparation des points
        points = self._prepare_points(batch, vectors)
        
        # 5. Stockage
        _logger.debug(f"Upsert de {len(points)} points")
        success = self.vector_store.upsert_records(points)
        
        return success

    async def _process_batch_async(self, batch: List[Dict[str, Any]]) -> bool:
        """Version async du traitement de batch"""
        if not batch:
            return True
        
        texts = [item["text"] for item in batch]
        
        # Embedding async
        vectors = await self.embedder.embed_texts_async(texts)
        
        if len(vectors) != len(batch):
            return False
        
        if self.cfg.validate_vectors:
            if not self._validate_vectors(vectors):
                return False
        
        points = self._prepare_points(batch, vectors)
        
        # Upsert async
        success = await self.vector_store.upsert_records_async(points)
        
        return success

    def _prepare_points(
        self,
        batch: List[Dict[str, Any]],
        vectors: List[List[float]]
    ) -> List[Dict[str, Any]]:
        """Prépare les points pour l'upsert"""
        points = []
        
        for chunk, vector in zip(batch, vectors):
            point = {
                "chunk_id": chunk["chunk_id"],
                "vector": vector,
                "metadata": {
                    "source": chunk["source"],
                    "doc_id": chunk["id"],
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                    "text_hash": chunk["text_hash"],
                    **chunk["meta"]
                }
            }
            points.append(point)
        
        return points

    def _validate_vectors(self, vectors: List[List[float]]) -> bool:
        """Valide les vecteurs générés"""
        if not vectors:
            return False
        
        expected_dim = len(vectors[0])
        
        for vec in vectors:
            if not vec or len(vec) != expected_dim:
                return False
            if not all(isinstance(v, (int, float)) for v in vec):
                return False
        
        return True

    def _build_result(self) -> Dict[str, Any]:
        """Construit le résultat final avec toutes les métriques"""
        normalizer_stats = self.normalizer.get_stats()
        store_stats = self.vector_store.get_stats()
        
        return {
            "success": self._metrics["batches_failed"] == 0,
            "pipeline_metrics": self._metrics,
            "normalizer_stats": normalizer_stats,
            "store_stats": store_stats,
            "processing_time_s": round(self._metrics["total_processing_time_s"], 2)
        }

    def _log_summary(self, result: Dict[str, Any]) -> None:
        """Log le résumé de l'exécution"""
        _logger.info("=" * 60)
        _logger.info("RÉSUMÉ DU PIPELINE D'INGESTION")
        _logger.info("=" * 60)
        _logger.info(f"Succès: {result['success']}")
        _logger.info(f"Temps total: {result['processing_time_s']}s")
        _logger.info(f"Chunks traités: {result['pipeline_metrics']['total_chunks']}")
        _logger.info(f"Batches traités: {result['pipeline_metrics']['batches_processed']}")
        _logger.info(f"Batches échoués: {result['pipeline_metrics']['batches_failed']}")
        _logger.info(f"Duplicatas ignorés: {result['normalizer_stats']['duplicates_skipped']}")
        _logger.info(f"Records dans store: {result['store_stats']['total_records']}")
        _logger.info("=" * 60)


# ==============================================================================
# 5. DÉMONSTRATION
# ==============================================================================

if __name__ == "__main__":
    _logger.info("=== DÉMONSTRATION DU PIPELINE D'INGESTION ===\n")
    
    # Données de test
    RAW_DATA = [
        {
            "raw_id": 1,
            "source": "rapport_financier_2024.pdf",
            "text": (
                "Le premier trimestre a vu une augmentation de 15% des revenus liés aux services cloud. "
                "Cependant, la division R&D a connu des difficultés avec un dépassement de budget de 5 millions d'euros. "
                "L'adoption de l'IA générative dans nos outils internes est prévue pour le troisième trimestre, "
                "ce qui devrait rationaliser nos coûts opérationnels de 8%. La direction reste optimiste, "
                "visant une croissance annuelle de 20% malgré les défis macroéconomiques actuels. "
                "Le déploiement de la nouvelle architecture de microservices a été finalisé en Mars."
            ),
            "metadata": {"date": "2024-04-15", "region": "EU", "language": "fr"}
        },
        {
            "raw_id": 2,
            "source": "politique_securite.txt",
            "text": (
                "Tous les employés doivent changer leur mot de passe tous les 90 jours. "
                "L'accès aux données clients via l'API est strictement réservé au département de support. "
                "Toute tentative de connexion non autorisée sera signalée immédiatement. "
                "La politique de sécurité interdit le partage de tout token de session ou clé privée. "
                "Pour les transactions, nous utilisons un système de cryptage avancé."
            ),
            "metadata": {"date": "2023-10-01", "region": "GLOBAL", "language": "fr"}
        },
        {
            "raw_id": 3,
            "source": "guide_utilisateur.md",
            "text": (
                "Bienvenue dans notre plateforme. Ce guide vous aidera à démarrer rapidement. "
                "Commencez par créer un compte et configurer votre profil. "
                "Vous pouvez ensuite accéder aux différentes fonctionnalités via le menu principal."
            ),
            "metadata": {"date": "2024-01-10", "region": "EU", "language": "fr"}
        }
    ]
    
    # Configuration
    norm_cfg = NormalizerConfig(
        chunk_size_tokens=40,
        chunk_overlap_tokens=10,
        min_chunk_length=20,
        enable_deduplication=True
    )
    
    ingest_cfg = IngestionConfig(
        batch_size=8,
        max_retries=3,
        log_progress_every=5,
        validate_vectors=True,
        metrics_enabled=True
    )
    
    # Composants
    embedder = MockEmbedder(vector_dim=128, latency_ms=5.0)
    vector_store = MockVectorStore(index_name="rag_production")
    
    # Exécution
    pipeline = IngestionPipeline(norm_cfg, ingest_cfg, vector_store, embedder)
    result = pipeline.run(RAW_DATA)
    
    # Affichage des résultats
    print("\n" + "=" * 60)
    print("APERÇU DES DONNÉES STOCKÉES")
    print("=" * 60)
    
    for i, (key, item) in enumerate(list(vector_store.data_store.items())[:5], 1):
        print(f"\n[{i}] ID: {key}")
        print(f"Doc: {item['metadata']['doc_id']}")
        print(f"Source: {item['metadata']['source']}")
        print(f"Texte: {item['metadata']['text'][:80]}...")
        print(f"Vecteur dim: {len(item['vector'])}")
    
    print("\n" + "=" * 60)
    _logger.info("=== FIN DE LA DÉMONSTRATION ===")



    'i'