import logging
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, List

# Configuration des chemins pour importer les modules voisins
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from .indexer import UniversalIndexer
    from .vector_store import VectorStoreHandler
    from . import config
except ImportError:
    from services.utils.indexer import UniversalIndexer
    from vector_store import VectorStoreHandler
    import config
    
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag.ingestor")

class DataIngestor:
    """
    Classe responsable de l'ingestion des données brutes vers la base vectorielle.
    Gère le cycle de vie des données (Nettoyage -> Indexation).
    """

    def __init__(self):
        self.indexer = UniversalIndexer()
        self.store_handler = VectorStoreHandler()
        # Chemin vers les données traitées (simule un stockage S3 ou une DB)
        self.processed_data_dir = Path(getattr(config, 'PROCESSED_DATA_DIR', './data/processed'))

    def _determine_source_type(self, data: Dict[str, Any]) -> str:
        """Détermine le type de source à partir des métadonnées pour le filtrage."""
        if data.get('content_type') in ['pdf', 'pdf_via_html', 'html_article']:
            return "BULLETIN_PDF"
        if data.get('features_data') is not None:
            return "METEO_VECTOR"
        if data.get('category') in config.WATCHLIST_CATEGORIES:
            return "METEO_ALERT"
        
        return "UNKNOWN"

    def ingest_data_from_orchestrator(self, scraped_data: List[Dict[str, Any]]):
        """
        Ingère une liste de résultats de scraping (cartes ou bulletins) en un seul lot.
        Gère le nettoyage préalable si les données sont temporelles (ex: météo).
        """
        logger.info(f"--- Démarrage de l'Ingestion RAG pour {len(scraped_data)} éléments ---")

        # Logique de suppression des anciennes alertes avant d'indexer les nouvelles
        # Nous supposons que les alertes météo sont temporelles et doivent être rafraîchies
        self.store_handler.delete_by_source("METEO_ALERT")
        self.store_handler.delete_by_source("METEO_VECTOR")
        
        indexed_count = 0
        
        for data in scraped_data:
            source_type = self._determine_source_type(data)

            try:
                if source_type == "BULLETIN_PDF":
                    # Pour les documents, on utilise l'indexation par chunks
                    self.indexer.index_document({
                        "title": data.get('title'),
                        "period": data.get('period'),
                        "download_url": data.get('download_url'),
                        "text_content": data.get('text_content', '')
                    })
                    indexed_count += 1
                    
                elif source_type == "METEO_VECTOR" or source_type == "METEO_ALERT":
                    # Pour les vecteurs météo/inondation
                    self.indexer.index_meteo_data(
                        features=data.get('features_data', []), 
                        category=data.get('category'), 
                        source_url=data.get('snapshot_path', 'MapViewer')
                    )
                    indexed_count += len(data.get('features_data', []))

                else:
                    logger.warning(f"Type de donnée non indexé : {source_type}")
                    
            except Exception as e:
                logger.error(f"❌ Échec de l'indexation pour {data.get('category')}: {e}")
                
        logger.info(f"✅ Ingestion terminée. {indexed_count} entrées ajoutées au total.")

    # Note : Cette classe pourrait également contenir des méthodes pour 
    # lire directement des fichiers d'un dossier (ex: ingest_local_pdfs_from_dir)