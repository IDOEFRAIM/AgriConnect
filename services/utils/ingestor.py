import logging
import json
from pathlib import Path
from typing import Dict, Any, List

from services.utils.embedding import EmbeddingService
from services.utils.indexer import UniversalIndexer
from rag.components.vector_store import VectorStoreHandler
from rag.utils.chunker import TextChunker
from rag.processors.textualizer import MeteoProcessor # <-- Import du processeur

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag.ingestor")

class DataIngestor:
    """
    Classe responsable de l'ingestion ETL (Extract, Transform, Load) des données.
    Processus : 
    1. Réception donnée brute -> 2. Textualisation & Enrichissement -> 3. Chunking -> 4. Embedding -> 5. FAISS
    """

    def __init__(self):
        self.chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        self.store_handler = VectorStoreHandler()
        self.embedder = EmbeddingService()

    def _determine_source_type(self, data: Dict[str, Any]) -> str:
        """Détermine le type de source (Métier) à partir de la structure de la donnée."""
        metadata = data.get("metadata", {})
        if "source_type" in metadata:
            return metadata["source_type"]
        
        # Heuristiques
        if "raw_data" in metadata or "Highcharts" in str(data):
            return "METEO_VECTOR"
        if "market" in str(data) or "prix" in str(data).lower():
            return "MARKET_REPORT"
            
        return "GENERAL_DOC"

    def ingest_data_from_orchestrator(self, scraped_data: List[Dict[str, Any]]):
        """Pipeline complet d'ingestion robuste avec étape de textualisation."""
        logger.info(f"--- 🚀 Démarrage Ingestion RAG ({len(scraped_data)} items) ---")

        self.store_handler.delete_by_source("METEO_ALERT")
        
        documents_buffer = []

        for data in scraped_data:
            source_type = self._determine_source_type(data)
            
            # --- PHASE DE TEXTUALISATION (Pré-Chunking) ---
            # Transformer la donnée brute (JSON) en texte narratif riche
            raw_text = ""
            
            if source_type == "METEO_VECTOR" and "raw_data" in data.get("metadata", {}):
                # Cas spécial : JSON Météo brute (Highcharts)
                city = data.get("metadata", {}).get("city", "Ville Inconnue")
                raw_series = data["metadata"]["raw_data"][0]["series"][0]["data"]
                # C'est ici que la magie opère : JSON -> Texte Agronome
                raw_text = MeteoProcessor.process_highcharts_series(city, raw_series)
                logger.info(f"📝 Textualisation Météo pour {city}: Succès")

            else:
                # Cas standard : le texte existe déjà (PDF parsé)
                raw_text = data.get("content") or data.get("text_content") or ""

            if not raw_text:
                continue

            # --- PHASE DE CHUNKING ---
            base_doc = {
                "source_type": source_type,
                "title": data.get("title", "Sans titre"),
                "source": data.get("title", "Sans titre"), # Alignement pour Retriever
                "url": data.get("url") or data.get("downloaded_path"),
                "created_at": data.get("timestamp"),
                "zone_id": data.get("metadata", {}).get("city") or "General" 
            }
            
            chunks = self.chunker.split_text(raw_text)
            
            for i, chunk_text in enumerate(chunks):
                doc_chunk = base_doc.copy()
                doc_chunk["text_content"] = chunk_text
                doc_chunk["chunk_index"] = i
                documents_buffer.append(doc_chunk)

        # ... (Embedding et Indexation inchangés) ...
        # 3. Embedding en Batch (Performance)
        logger.info(f"⚡ Calcul des embeddings pour {len(documents_buffer)} chunks...")
        batch_texts = [d["text_content"] for d in documents_buffer]
        vectors = self.embedder.model.encode(batch_texts) 
        
        for i, doc in enumerate(documents_buffer):
            doc["vector"] = vectors[i].tolist() 

        self.store_handler.add_documents(documents_buffer)
        logger.info(f"✅ Ingestion terminée avec Textualisation. {len(documents_buffer)} nouveaux vecteurs.")