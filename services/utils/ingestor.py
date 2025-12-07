import logging
import json
from pathlib import Path
from typing import Dict, Any, List

from services.utils import UniversalIndexer,EmbeddingService
from rag.vector_store import VectorStoreHandler

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag.ingestor")

class DataIngestor:
    """
    Classe responsable de l'ingestion des données brutes vers la base vectorielle.
    Gère le cycle de vie des données (Nettoyage -> Embedding -> Indexation).
    """

    def __init__(self):
        self.indexer = UniversalIndexer()
        self.store_handler = VectorStoreHandler()
        self.embedder = EmbeddingService()  # <-- ajout embedder
        self.processed_data_dir = Path(getattr(config, 'PROCESSED_DATA_DIR', './data/processed'))

    def _determine_source_type(self, data: Dict[str, Any]) -> str:
        if data.get('content_type') in ['pdf', 'pdf_via_html', 'html_article']:
            return "BULLETIN_PDF"
        if data.get('features_data') is not None and data.get('category') not in ['health', 'agriculture', 'logistics']:
            return "METEO_VECTOR"
        if data.get('category') in config.WATCHLIST_CATEGORIES:
            return "METEO_ALERT"
        if data.get('category') == 'agriculture' or 'crop_health' in data.get('tags', []):
            return "AGRI_REPORT"
        if data.get('category') == 'health' or 'epidemic_risk' in data.get('tags', []):
            return "HEALTH_DATA"
        if data.get('category') == 'logistics' or 'infrastructure_status' in data.get('tags', []):
            return "INFRA_ALERT"
        return "UNKNOWN"

    def ingest_data_from_orchestrator(self, scraped_data: List[Dict[str, Any]]):
        logger.info(f"--- Démarrage de l'Ingestion RAG pour {len(scraped_data)} éléments ---")

        # Nettoyage sélectif (ex: supprimer uniquement les données expirées)
        self.store_handler.delete_by_source("METEO_ALERT")
        self.store_handler.delete_by_source("METEO_VECTOR")
        logger.info("🧹 Données METEO/ALERT obsolètes supprimées.")

        indexed_count = 0

        # Préparer batch pour embedding
        texts_to_embed = [d.get('text_content', '') for d in scraped_data if d.get('text_content')]
        if texts_to_embed:
            vectors = self.embedder.embed_documents(texts_to_embed)
        else:
            vectors = []

        for idx, data in enumerate(scraped_data):
            source_type = self._determine_source_type(data)
            try:
                if source_type == "BULLETIN_PDF":
                    self.indexer.index_document({
                        "title": data.get('title'),
                        "period": data.get('period'),
                        "download_url": data.get('download_url'),
                        "text_content": data.get('text_content', ''),
                        "vector": vectors[idx] if idx < len(vectors) else [0.0]*self.embedder.dimension
                    })
                    indexed_count += 1

                elif source_type in ["METEO_VECTOR", "METEO_ALERT"]:
                    self.indexer.index_meteo_data(
                        features=data.get('features_data', []),
                        category=data.get('category'),
                        source_url=data.get('snapshot_path', 'MapViewer')
                    )
                    indexed_count += len(data.get('features_data', []))

                elif source_type in ["AGRI_REPORT", "HEALTH_DATA", "INFRA_ALERT"]:
                    self.indexer.index_structured_data(data, source_type)
                    indexed_count += 1

                else:
                    logger.warning(f"Type de donnée non indexé/non reconnu : {source_type}")

            except Exception as e:
                logger.error(f"❌ Échec de l'indexation pour {data.get('category', source_type)}: {e}")

        logger.info(f"✅ Ingestion terminée. {indexed_count} entrées ajoutées au total.")