import logging
import time
import os
from typing import Dict, Any, List, Optional
# Imports des dépendances réelles (assumées)
from services.utils.embedding import EmbeddingService
from rag.vector_store import VectorStoreHandler
from services.utils.pdf_parser import PDFParser 

logger = logging.getLogger("UniversalIndexer")
# Configuration de base pour l'exemple
INDEXER_CONFIG = {
    'CHUNK_SIZE': 1000,
    'CHUNK_OVERLAP': 100
}

# ====================================================================
# UNIVERSAL INDEXER
# ====================================================================

class UniversalIndexer:
    """
    Gère l'indexation de différents types de données (documents, données structurées) 
    dans la base de données vectorielle.
    """

    def __init__(self):
        # Initialisation des dépendances réelles importées
        self.embedder = EmbeddingService()
        # CORRECTION CRITIQUE: L'erreur "Dimension incorrecte: 384 vs 1536" 
        # indique que l'EmbeddingService produit des vecteurs de dimension 384.
        # Nous initialisons VectorStoreHandler avec la dimension correcte (384) 
        # pour qu'elle corresponde à l'output de l'embedder.
        self.store = VectorStoreHandler(dimension=384)
        self.parser = PDFParser()
        self.chunk_size = INDEXER_CONFIG['CHUNK_SIZE']
        self.chunk_overlap = INDEXER_CONFIG['CHUNK_OVERLAP']

    # --- Méthodes de support ---

    def _chunk_text(self, text: str) -> List[str]:
        """Découpe un texte long en segments (chunks) pour l'indexation."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            # Déplace le point de départ avec un chevauchement
            start += self.chunk_size - self.chunk_overlap
        
        logger.debug(f"Texte découpé en {len(chunks)} chunks.")
        return chunks

    # --- Logiques d'Indexation Spécifiques ---

    def index_document(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Indexe les documents longs (PDF, articles, etc.) en les découpant.
        Utilisé pour la catégorie 'agronomic_bulletins'.
        """
        total_indexed_chunks = 0
        all_vectors_data = [] # Pour collecter toutes les données avant l'insertion

        for doc in documents:
            
            if not isinstance(doc, dict):
                logger.warning(f"Document ignoré: Élément de type {type(doc).__name__} trouvé dans la liste d'indexation des bulletins, attendu 'dict'.")
                continue
            
            doc_id = str(doc.get('id', time.time()))
            title = doc.get('title', 'Document sans titre')
            url = doc.get('url', 'N/A')
            file_path = doc.get('file_path') 
            extracted_content = doc.get('content') 
            
            logger.info(f"⚙️ Indexation du document : {title} ({url})")
            
            try:
                full_text = None
                
                if extracted_content:
                    full_text = extracted_content
                elif file_path:
                    full_text = self.parser.extract_text_from_path(file_path)
                else:
                    logger.warning(f"Document ignoré: Ni 'file_path' ni 'content' n'est présent pour {title}.")
                    continue
                
                if not full_text or not full_text.strip():
                    logger.warning(f"Document ignoré: Contenu vide après extraction pour {title}.")
                    continue

                # 1. Découpage en fragments
                chunks = self._chunk_text(full_text)
                
                # 2. Création des embeddings
                vectors = self.embedder.embed_documents(chunks)
                
                # 3. Préparation pour l'insertion
                for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                    # Cette structure est interne à l'indexeur et sera transformée plus tard
                    all_vectors_data.append({
                        'id': f"{doc_id}-{i}", 
                        'vector': vector,
                        'metadata': {
                            'source': 'agronomic_bulletins',
                            'title': title,
                            'url': url,
                            'chunk_index': i,
                            'text_content': chunk, # Le texte réel du fragment
                        }
                    })
                
                total_indexed_chunks += len(vectors)
                
            except Exception as e:
                logger.error(f"Erreur DÉTAILLÉE lors de l'indexation de {title}: {type(e).__name__} - {e}", exc_info=False)
                continue
        
        # 4. Insertion groupée dans le VectorStore après restructuration
        if all_vectors_data:
            # Transformation des données pour correspondre à la signature de VectorStoreHandler.insert_data
            vectors_list = [d['vector'] for d in all_vectors_data]
            texts_list = [d['metadata']['text_content'] for d in all_vectors_data]
            source_types_list = [d['metadata']['source'] for d in all_vectors_data]
            
            # Nous retirons 'text_content' des métadonnées pour éviter la duplication.
            clean_metadatas = []
            for d in all_vectors_data:
                meta = d['metadata'].copy()
                del meta['text_content'] 
                meta['external_id'] = d['id']
                clean_metadatas.append(meta)

            self.store.insert_data(vectors_list, texts_list, source_types_list, clean_metadatas)

        return {"status": "SUCCESS", "message": f"{total_indexed_chunks} fragments de documents ont été indexés."}


    def index_meteo_data(self, meteo_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Indexe les données structurées de prévisions météo.
        """
        total_indexed = 0
        # Listes pour l'insertion groupée
        vectors_list, texts_list, source_types_list, metadatas_list = [], [], [], []
        
        for item in meteo_data_list:
            
            if not isinstance(item, dict):
                logger.warning(f"Prévision ignorée: Élément de type {type(item).__name__} trouvé dans la liste météo, attendu 'dict'.")
                continue 

            data_path = item.get('data_path')
            city = item.get('city', 'Location Inconnue')
            preview = item.get('content_preview', 'Prévision météo générale.')
            
            if not data_path:
                logger.warning(f"Prévision ignorée: Le chemin des données (data_path) est manquant pour {city}.")
                continue
            
            logger.info(f"⚙️ Préparation des données météo structurées pour {city} ({data_path})")
            
            try:
                # 1. Création d'un contenu textuel descriptif (ce qui sera stocké et recherché)
                text_to_embed = f"Prévision Météo pour {city}: {preview}. Le fichier complet se trouve à {data_path}."
                
                # 2. Création de l'embedding
                vector = self.embedder.embed_query(text_to_embed)
                
                # 3. Préparation pour l'insertion
                vector_id = f"meteo-{city.replace(' ', '_')}-{os.path.basename(data_path).split('.')[0]}"
                
                # Métadonnées à stocker DANS FAISS (sans le texte)
                meta = {
                    'source': 'weather_forecast',
                    'type': 'structured_forecast',
                    'location': city,
                    'data_path': data_path,
                    'external_id': vector_id,
                }
                
                # Ajout aux listes groupées
                vectors_list.append(vector)
                texts_list.append(text_to_embed)
                source_types_list.append('weather_forecast')
                metadatas_list.append(meta)
                total_indexed += 1
                
            except Exception as e:
                logger.error(f"Erreur DÉTAILLÉE lors de la préparation météo pour {city}: {type(e).__name__} - {e}")
                continue
        
        # 4. Insertion groupée
        if vectors_list:
            self.store.insert_data(vectors_list, texts_list, source_types_list, metadatas_list)

        return {"status": "SUCCESS", "message": f"{total_indexed} prévisions météo ont été indexées."}


    def index_data(self, data: List[Dict[str, Any]], category: str) -> Dict[str, Any]:
        """
        Point d'entrée principal pour le routage de l'indexation.
        """
        if not data:
            logger.warning(f"Aucune donnée à indexer pour la catégorie {category}.")
            return {"status": "WARNING", "message": "Aucune donnée fournie pour l'indexation."}

        logger.info(f"Routage de {len(data)} documents vers l'indexeur pour la catégorie '{category}'...")

        if category == "agronomic_bulletins":
            return self.index_document(data)
        
        elif category == "weather_forecast":
            return self.index_meteo_data(data)
            
        else:
            logger.warning(f"Catégorie d'indexation non supportée ou non reconnue: {category}. Données ignorées.")
            return {"status": "WARNING", "message": f"Catégorie '{category}' non supportée. Indexation ignorée."}