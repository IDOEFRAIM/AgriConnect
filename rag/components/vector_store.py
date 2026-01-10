import logging
import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import faiss

# --- Configuration du Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rag.VectorStoreHandler")

class VectorStoreHandler:
    """
    Gestionnaire de base vectorielle (Vector Store) local utilisant FAISS.
    Simule une DB robuste (Milvus/Pinecone) mais reste 100% local et fichier-flat pour la portabilité.
    Utilise IndexIDMap pour permettre la suppression ciblée par ID.
    """
    
    def __init__(self, index_path: str = "data/vector_store/agriconnect.index", metadata_path: str = "data/vector_store/metadata.json", dimension: int = 384):
        """
        :param index_path: Chemin vers l'index FAISS physique.
        :param metadata_path: Chemin vers le stockage JSON des métadonnées (le texte associé).
        :param dimension: Dimension des vecteurs (384 pour MiniLM, 768 pour SBERT base, etc.)
        """
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.dimension = dimension
        
        # Ensure directory exists
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # FAISS Initialization
        self.metadata = self._load_metadata()
        self.index = self._load_or_create_index()
        
    def _load_or_create_index(self):
        if self.index_path.exists():
            try:
                logger.info(f"Loading FAISS index from {self.index_path}")
                index = faiss.read_index(str(self.index_path))
                # Check if already IndexIDMap, otherwise wrap (migration)
                if not isinstance(index, faiss.IndexIDMap) and not isinstance(index, faiss.IndexIDMap2):
                     # Note: Wrapping an existing populated flat index is complex without rebuilding IDs.
                     # Delete and recreate if format is incorrect for this 'Robust' version.
                     logger.warning("Existing index is not IDMap. Migration (rebuild) needed. Resetting for cleanliness.")
                     return self._create_new_index()
                return index
            except Exception as e:
                logger.error(f"Error loading index : {e}. Creating new one.")
        
        return self._create_new_index()

    def _create_new_index(self):
        logger.info(f"Creating new FAISS IDMap index (Dimension: {self.dimension})")
        # IndexFlatIP = Index Flat Inner Product (for cosine similarity with normalized vectors)
        # IndexFlatL2 = Euclidean Distance
        quantizer = faiss.IndexFlatL2(self.dimension)
        return faiss.IndexIDMap(quantizer)

    def _load_metadata(self) -> Dict[int, Dict[str, Any]]:
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    # JSON keys are always strings, convert back to int
                    data = json.load(f)
                    return {int(k): v for k, v in data.items()}
            except Exception as e:
                logger.error(f"Erreur chargement métadonnées : {e}")
        return {}

    def _save(self):
        """Persistance sur disque."""
        try:
            faiss.write_index(self.index, str(self.index_path))
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            logger.debug("Sauvegarde Index + Metadata terminée.")
        except Exception as e:
            logger.error(f"Erreur sauvegarde vector store: {e}")

    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Ajoute des documents à la base avec gestion manuelle des IDs.
        """
        if not documents:
            return
            
        vectors = []
        valid_docs = []
        
        for doc in documents:
            vec = doc.get("vector")
            if vec is None:
                continue
                
            # Vérification et Normalisation dimension
            np_vec = np.array(vec, dtype='float32')
            if np_vec.shape[0] != self.dimension:
                logger.warning(f"Dimension vecteur incorrecte ({np_vec.shape[0]} vs {self.dimension}). Ignoré.")
                continue
            
            vectors.append(np_vec)
            valid_docs.append(doc)

        if not vectors:
            return

        np_vectors = np.stack(vectors)
        
        # Unique ID generation based on current max
        start_id = max(self.metadata.keys()) + 1 if self.metadata else 0
        ids = np.arange(start_id, start_id + len(valid_docs)).astype('int64')
        
        # FAISS add with explicit IDs
        self.index.add_with_ids(np_vectors, ids)
        
        # Mise à jour metadata
        for i, doc in enumerate(valid_docs):
            meta = doc.copy()
            if "vector" in meta: del meta["vector"]
            self.metadata[int(ids[i])] = meta
            
        self._save()
        logger.info(f"Ajouté {len(valid_docs)} documents (IDs {start_id} à {start_id + len(valid_docs) - 1}). Total index: {self.index.ntotal}")

    def delete_by_source(self, source_type: str):
        """
        Supprime tous les documents correspondant à un certain type de source (ex: 'METEO_ALERT').
        Essentiel pour rafraîchir les données sans dupliquer.
        """
        ids_to_remove = [k for k, v in self.metadata.items() if v.get("source_type") == source_type]
        
        if not ids_to_remove:
            logger.info(f"Aucun document de type '{source_type}' à supprimer.")
            return

        logger.info(f"Suppression de {len(ids_to_remove)} documents de type '{source_type}'...")
        
        # Suppression FAISS
        ids_np = np.array(ids_to_remove, dtype='int64')
        self.index.remove_ids(ids_np)
        
        # Suppression Metadata
        for k in ids_to_remove:
            del self.metadata[k]
            
        self._save()
        logger.info(f"Suppression terminée. Reste {self.index.ntotal} documents.")

    def search(self, query_vector: List[float], k: int = 4, source_filter: str = None) -> List[Dict[str, Any]]:
        """
        Recherche sémantique pure.
        :param source_filter: Filtrer sur le champ 'source_type' des métadonnées (post-filtering simplifié).
        """
        if self.index.ntotal == 0:
            return []

        np_query = np.array([query_vector], dtype='float32')
        k = min(k, self.index.ntotal)
        
        # Recherche FAISS
        distances, indices = self.index.search(np_query, k * 3 if source_filter else k) # Fetch more if filtering
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1: continue
            
            meta = self.metadata.get(int(idx))
            if not meta: continue
            
            # Application du Filtre Logiciel (Post-Filtering)
            # FAISS ne supporte pas le filtrage natif facilement sans complexité.
            if source_filter and meta.get("source_type") != source_filter:
                continue
                
            res = meta.copy()
            res['score'] = float(dist) # Distance L2 (plus petit = mieux). Si InnerProduct, plus grand = mieux.
            results.append(res)
            
            if len(results) >= k:
                break
                
        return results
