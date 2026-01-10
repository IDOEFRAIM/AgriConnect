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
        
        # Assurer que le dossier existe
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialisation FAISS
        self.index = self._load_or_create_index()
        self.metadata = self._load_metadata()
        
    def _load_or_create_index(self):
        if self.index_path.exists():
            try:
                logger.info(f"Chargement de l'index FAISS depuis {self.index_path}")
                return faiss.read_index(str(self.index_path))
            except Exception as e:
                logger.error(f"Erreur chargement index : {e}. Création d'un nouveau.")
        
        logger.info(f"Création d'un nouvel index FAISS (Dimension: {self.dimension})")
        # IndexFlatIP = Index Flat Inner Product (pour similarité cosinus avec vecteurs normalisés)
        # IndexFlatL2 = Distance Euclidienne
        return faiss.IndexFlatL2(self.dimension)

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
        Ajoute des documents à la base.
        Chaque doc doit avoir au moins: 'vector' (List[float]) et 'text_content'.
        """
        if not documents:
            return
            
        vectors = []
        start_id = self.index.ntotal
        
        for i, doc in enumerate(documents):
            vec = doc.get("vector")
            if vec is None:
                continue
                
            # Vérification et Normalisation dimension
            np_vec = np.array(vec, dtype='float32')
            if np_vec.shape[0] != self.dimension:
                logger.warning(f"Dimension vecteur incorrecte ({np_vec.shape[0]} vs {self.dimension}). Ignoré.")
                continue
            
            vectors.append(np_vec)
            
            # Stockage metadata (sans le vecteur lourd)
            meta = doc.copy()
            if "vector" in meta: del meta["vector"]
            self.metadata[start_id + i] = meta

        if vectors:
            np_vectors = np.stack(vectors)
            self.index.add(np_vectors)
            self._save()
            logger.info(f"✅ Ajouté {len(vectors)} vecteurs. Total: {self.index.ntotal}")

    def search(self, query_vector: List[float], k: int = 5, source_filter: Optional[str] = None, vector_filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Recherche sémantique.
        :param query_vector: Vecteur de la requête.
        :param k: Top K résultats.
        :param source_filter: (Optionnel) Filtrer par 'source_type' dans les métadonnées.
        """
        np_query = np.array([query_vector], dtype='float32')
        
        # FAISS search
        distances, indices = self.index.search(np_query, k * 4) # On cherche plus large pour filtrer après
        
        results = []
        found_indices = indices[0]
        found_dists = distances[0]
        
        for idx, dist in zip(found_indices, found_dists):
            if idx == -1: continue # Padding FAISS
            
            idx = int(idx)
            meta = self.metadata.get(idx, {})
            
            # --- FILTRAGE MÉTIER ---
            if source_filter and meta.get("source_type") != source_filter:
                continue
            
            if vector_filters:
                # Ex: {"zone_id": "Boucle du Mouhoun"}
                match = True
                for key, val in vector_filters.items():
                    if meta.get(key) != val:
                        match = False
                        break
                if not match: continue

            # Score = Distance L2. Plus petit = Mieux.
            # Conversion en score de similarité (approx) pour l'UI : 1 / (1 + dist)
            score = 1 / (1 + dist)
            
            results.append({
                "content": meta.get("text_content"),
                "metadata": meta,
                "score": float(score),
                "distance": float(dist)
            })
            
            if len(results) >= k:
                break
                
        return results

    def delete_by_source(self, source_type: str):
        """
        Suppression douce logicielle (les vecteurs restent dans FAISS mais la metadata est marquée).
        Pour une vraie suppression FAISS, il faut reconstruire l'index (IndexIDMap), complexe pour ce MVP.
        Ici on supprime juste les metadatas, donc le search filtrera les vecteurs fantômes.
        """
        keys_to_delete = []
        for idx, meta in self.metadata.items():
            if meta.get("source_type") == source_type:
                keys_to_delete.append(idx)
        
        for k in keys_to_delete:
            del self.metadata[k]
            
        self._save()
        logger.info(f"Suppression logique de {len(keys_to_delete)} documents de type {source_type}.")
        """Lance l'exécution de tous les agents et retourne tous les résultats collectés."""
        all_collected_data = []
        for category, agent in self.agents.items():
            results = self.run_agent_and_collect(category, agent)
            all_collected_data.extend(results)
        return all_collected_data


# --- DÉMONSTRATION DE LA PIPELINE DÉCOUPLÉE ---
if __name__ == '__main__':
    # Initialisation des composants
    DB_PATH = "data/orchestrator_final.db"
    
    # 1. PRÉPARATION DES OUTILS (StorageManager pour la persistance)
    try:
        # Assurez-vous que le StorageManager est bien instancié après la correction
        storage = StorageManager(db_path=DB_PATH)
        store = VectorStoreHandler()
        embedder = EmbeddingService()
        reranker = Reranker()
    
    except Exception as e:
        logger.error(f"Erreur fatale d'initialisation des services: {e}. Vérifiez storage_manager.py.")
        exit()

    # 2. DÉFINITION DE LA TÂCHE
    agents_map = {
        "METEO": ScraperAgent("METEO"),
        "SUBVENTION": ScraperAgent("SUBVENTION"),
        "ALERTE_INONDATION": ScraperAgent("ALERTE_INONDATION"),
    }
    zones_list = ["Paris", "Lyon", "Marseille"]

    # 3. L'ORCHESTRATEUR (Exécution pure)
    orchestrator = ScraperOrchestrator(agents_map, zones_list)
    
    print("\n[Étape 1] 🚀 Lancement de l'Orchestrateur pour collecter les données...")
    final_collected_data = orchestrator.run_pipeline() 
    
    print(f"\n[Étape 1 Terminé] Total des enregistrements collectés par l'Orchestrateur : {len(final_collected_data)}")

    # 4. LE FLUX DE TRAITEMENT AVAL (Persistance, Caching, Vectorisation)
    print("\n[Étape 2] 💾 Démarrage du Traitement Aval (Persistence et Caching/Déduplication)...")
    
    processed_count = 0
    for item in final_collected_data:
        # Stockage de la donnée brute dans la table dynamique
        is_new = storage.save_raw_data(
            zone_id=item["zone_id"],
            category=item["category"],
            data=item["data"],
            effective_date=item["acquisition_time"],
            source_url=item["data"].get("source_url")
        )
        
        # Le même 'item' serait envoyé à un Vector Store SEULEMENT s'il est nouveau ou modifié
        if is_new:
            store.index_data(item["category"], item["data"]) 
            processed_count += 1

    print(f"\n[Étape 2 Terminé] Total des NOUVEAUX enregistrements persistés (après déduplication) : {processed_count}")

    # 5. Testons la robustesse/déduplication en relançant l'Orchestrateur
    print("\n[Étape 3] 🔄 Relance de la pipeline (pour tester la déduplication)...")
    second_run_data = orchestrator.run_pipeline()
    processed_count_second = 0
    for item in second_run_data:
        is_new = storage.save_raw_data(
            zone_id=item["zone_id"],
