# rag/setup_with_real_data.py
"""
Configuration de la pipeline RAG avec de vraies données.
Ce script vous guide pour connecter vos composants réels.
"""

from __future__ import annotations
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# =========================================================================
# Option 1: Utiliser vos composants existants
# =========================================================================

def setup_with_existing_components():
    """
    Si vous avez déjà des composants implémentés, utilisez cette fonction.
    """
    print("=" * 70)
    print("SETUP AVEC COMPOSANTS EXISTANTS")
    print("=" * 70)
    
    # Décommenter et adapter selon vos imports
    """
    from rag.embedder import Embedder, EmbedderConfig
    from rag.retriever import Retriever, RetrieverConfig
    from rag.reranker import Reranker, RerankConfig
    from rag.cross_encoder import CrossEncoder
    from rag.indexer_milvus import MilvusIndexer, MilvusConfig
    
    # 1. Setup Indexer
    milvus_cfg = MilvusConfig(
        host="localhost",
        port="19530",
        default_index_params={
            "index_type": "HNSW",
            "metric_type": "L2",
            "params": {"M": 16, "efConstruction": 200}
        }
    )
    indexer = MilvusIndexer(cfg=milvus_cfg)
    indexer.connect()
    
    # 2. Setup Embedder
    embedder_cfg = EmbedderConfig(
        batch_size=64,
        normalize=True,
        dtype="float32",
        cache_size=10000
    )
    embedder = Embedder(cfg=embedder_cfg)
    
    # 3. Setup Retriever
    retriever_cfg = RetrieverConfig(
        top_k=50,
        candidate_pool=200,
        timeout_s=8.0
    )
    retriever = Retriever(
        embedder=embedder,
        indexer=indexer,
        cfg=retriever_cfg,
        collection="bulletins_meteo"
    )
    
    # 4. Setup Reranker
    cross_encoder = CrossEncoder()
    reranker_cfg = RerankConfig(
        cross_weight=0.6,
        vector_weight=0.4,
        score_normalize=True,
        top_n=20
    )
    reranker = Reranker(
        cross_encoder=cross_encoder,
        cfg=reranker_cfg
    )
    
    return embedder, retriever, reranker
    """
    
    print("\n⚠️  Cette fonction nécessite vos composants réels.")
    print("    Décommentez et adaptez le code ci-dessus.")
    return None, None, None


# =========================================================================
# Option 2: Charger des données depuis des fichiers JSON
# =========================================================================

async def load_bulletins_from_json(directory: Path) -> List[Dict[str, Any]]:
    """
    Charge tous les bulletins depuis un répertoire de fichiers JSON.
    
    Args:
        directory: Chemin vers le dossier contenant les JSON
        
    Returns:
        Liste de documents formatés
    """
    print(f"\n📂 Chargement des bulletins depuis: {directory}")
    
    if not directory.exists():
        print(f"  ✗ Répertoire introuvable: {directory}")
        return []
    
    documents = []
    json_files = list(directory.glob("*.json"))
    
    print(f"  📄 {len(json_files)} fichiers JSON trouvés")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Si c'est un seul document
            if isinstance(data, dict):
                data = [data]
            
            # Normaliser chaque document
            for item in data:
                doc = {
                    'id': item.get('id') or item.get('doc_id') or f"{json_file.stem}_{len(documents)}",
                    'source': str(json_file),
                    'text': item.get('text') or item.get('content') or '',
                    'meta': item.get('meta', {})
                }
                
                # Extraire le texte des chunks si présent
                if 'chunks' in item:
                    doc['chunks'] = item['chunks']
                
                if doc['text']:
                    documents.append(doc)
        
        except Exception as e:
            print(f"  ✗ Erreur lors du chargement de {json_file.name}: {e}")
    
    print(f"  ✓ {len(documents)} documents chargés")
    return documents


# =========================================================================
# Option 3: Créer un index simple en mémoire
# =========================================================================

class SimpleInMemoryIndex:
    """
    Index simple en mémoire pour démarrer rapidement.
    Remplacer par Milvus/Qdrant en production.
    """
    
    def __init__(self, embedder: Any):
        self.embedder = embedder
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Dict[str, List[float]] = {}
    
    async def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Indexe des documents avec leurs embeddings."""
        print(f"\n🔧 Indexation de {len(documents)} documents...")
        
        texts = [doc['text'] for doc in documents]
        
        # Générer les embeddings
        if hasattr(self.embedder, 'encode_async'):
            embeddings = await self.embedder.encode_async(texts)
        else:
            embeddings = self.embedder.encode(texts)
        
        # Stocker
        for doc, emb in zip(documents, embeddings):
            doc_id = doc['id']
            self.documents[doc_id] = doc
            self.embeddings[doc_id] = emb
        
        print(f"  ✓ {len(self.documents)} documents indexés")
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calcule la similarité cosine."""
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
    
    async def search(
        self,
        query: str,
        top_k: int = 50
    ) -> List[Dict[str, Any]]:
        """Recherche les documents les plus similaires."""
        # Encoder la query
        if hasattr(self.embedder, 'encode_async'):
            query_emb = (await self.embedder.encode_async([query]))[0]
        else:
            query_emb = self.embedder.encode([query])[0]
        
        # Calculer les similarités
        scored_docs = []
        for doc_id, doc in self.documents.items():
            doc_emb = self.embeddings[doc_id]
            score = self._cosine_similarity(query_emb, doc_emb)
            
            scored_docs.append({
                'id': doc_id,
                'doc_id': doc_id,
                'text': doc['text'],
                'source': doc['source'],
                'meta': doc['meta'],
                'score': score,
                **doc
            })
        
        # Trier et retourner top_k
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        return scored_docs[:top_k]


class SimpleRetrieverWrapper:
    """Wrapper pour utiliser SimpleInMemoryIndex comme retriever."""
    
    def __init__(self, index: SimpleInMemoryIndex):
        self.index = index
    
    async def retrieve(self, query: str, top_k: int = 50, **kwargs) -> List[Dict[str, Any]]:
        """Interface compatible avec la pipeline."""
        return await self.index.search(query, top_k)


# =========================================================================
# Configuration Complète avec Pipeline
# =========================================================================

async def setup_complete_pipeline():
    """
    Setup complet de la pipeline avec toutes les options.
    """
    print("\n" + "=" * 70)
    print("CONFIGURATION COMPLÈTE DE LA PIPELINE")
    print("=" * 70)
    
    # 1. Choix de l'embedder
    print("\n1️⃣  Configuration de l'Embedder")
    print("    Options disponibles:")
    print("    a) MockEmbedder (pour tests)")
    print("    b) Sentence-Transformers (production)")
    print("    c) OpenAI Embeddings (production)")
    
    # Pour cet exemple, utilisons MockEmbedder
    from rag.utils.mockExample import MockEmbedder
    embedder = MockEmbedder(dim=128)
    print(f"    ✓ MockEmbedder configuré (128 dim)")
    
    # 2. Chargement des données
    print("\n2️⃣  Chargement des Données")
    
    # Option A: Charger depuis JSON
    bulletins_dir = Path("bulletins_json")
    if bulletins_dir.exists():
        documents = await load_bulletins_from_json(bulletins_dir)
    else:
        print(f"    ⚠️  Répertoire {bulletins_dir} introuvable")
        print(f"    📦 Utilisation des données d'exemple")
        from rag.utils.mockExample import SAMPLE_BULLETINS
        documents = SAMPLE_BULLETINS
    
    # 3. Création de l'index
    print("\n3️⃣  Création de l'Index")
    index = SimpleInMemoryIndex(embedder)
    await index.index_documents(documents)
    
    # 4. Setup retriever
    print("\n4️⃣  Configuration du Retriever")
    retriever = SimpleRetrieverWrapper(index)
    print("    ✓ Retriever configuré")
    
    # 5. Setup reranker
    print("\n5️⃣  Configuration du Reranker")
    from rag.utils.mockExample import MockReranker
    reranker = MockReranker()
    print("    ✓ Reranker configuré")
    
    # 6. Création de la pipeline
    print("\n6️⃣  Création de la Pipeline")
    from rag.utils.mockExample import AugmentationPipeline, AugmentationConfig
    
    config = AugmentationConfig(
        top_k=20,
        rerank_top_n=10,
        use_cross_encoder=True,
        enable_caching=True,
        enable_metrics=True
    )
    
    pipeline = AugmentationPipeline(
        retriever=retriever,
        encoder=embedder,
        reranker=reranker,
        cfg=config
    )
    
    print("    ✓ Pipeline créée avec succès!")
    
    return pipeline, documents


# =========================================================================
# Test de la Pipeline
# =========================================================================

async def test_pipeline(pipeline: Any):
    """Teste la pipeline avec des requêtes."""
    print("\n" + "=" * 70)
    print("TEST DE LA PIPELINE")
    print("=" * 70)
    
    test_queries = [
        "Quelle est la tendance pluviométrique pour septembre 2025 ?",
        "Quelles sont les prévisions pour octobre ?",
        "Quelle est la température moyenne ?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Requête {i}: {query}")
        
        response = await pipeline.augment(query)
        result = response.to_dict()
        
        print(f"  ✓ {len(result['context']['snippets'])} snippets trouvés")
        print(f"  ✓ Temps: {result['diagnostics']['timings']['total_time_s']:.3f}s")
        
        if result['context']['snippets']:
            top_snippet = result['context']['snippets'][0]
            print(f"  📄 Top snippet: {top_snippet['text'][:100]}...")


# =========================================================================
# Guide d'intégration
# =========================================================================

def print_integration_guide():
    """Affiche un guide pour intégrer vos propres composants."""
    print("\n" + "=" * 70)
    print("GUIDE D'INTÉGRATION")
    print("=" * 70)
    
    guide = """
📚 ÉTAPES POUR INTÉGRER VOS COMPOSANTS RÉELS

1. EMBEDDER
   ✓ Doit implémenter: encode(texts: List[str]) -> List[List[float]]
   ✓ Optionnel: encode_async(texts: List[str]) -> List[List[float]]
   
   Exemple avec Sentence-Transformers:
   ```python
   from sentence_transformers import SentenceTransformer
   
   class ProductionEmbedder:
       def __init__(self, model_name="all-MiniLM-L6-v2"):
           self.model = SentenceTransformer(model_name)
       
       def encode(self, texts):
           return self.model.encode(texts).tolist()
       
       async def encode_async(self, texts):
           return await asyncio.to_thread(self.encode, texts)
   ```

2. RETRIEVER
   ✓ Doit implémenter: retrieve(query: str, top_k: int) -> List[Dict]
   ✓ Format de retour: [{"id", "text", "source", "meta", "score"}, ...]
   
   Voir: rag/retriever.py pour l'interface complète

3. RERANKER (optionnel mais recommandé)
   ✓ Doit implémenter: rerank(query: str, docs: List[Dict], top_n: int) -> List[Dict]
   ✓ Optionnel: rerank_async(...)
   
   Voir: rag/reranker.py pour l'interface complète

4. DONNÉES
   ✓ Format JSON recommandé: {"id", "text", "meta": {...}}
   ✓ Placer dans: bulletins_json/*.json
   ✓ Ou utiliser l'indexation programmatique

5. LANCER LA PIPELINE
   ```python
   from augmentation_pipeline import AugmentationPipeline, AugmentationConfig
   
   pipeline = AugmentationPipeline(
       retriever=your_retriever,
       encoder=your_embedder,
       reranker=your_reranker,  # optionnel
       cfg=AugmentationConfig()
   )
   
   response = await pipeline.augment("Votre question")
   ```

📁 STRUCTURE RECOMMANDÉE
   AgConnect/
   ├── bulletins_json/          # Vos fichiers JSON
   │   ├── BAD25092.json
   │   ├── PREV_OCT.json
   │   └── ...
   ├── rag/
   │   ├── augmentation_pipeline.py
   │   ├── embedder.py
   │   ├── retriever.py
   │   ├── reranker.py
   │   └── setup_with_real_data.py  # Ce fichier
   └── venv/

🚀 DÉMARRAGE RAPIDE
   1. Placer vos bulletins JSON dans bulletins_json/
   2. Exécuter: python rag/complete_working_example.py
   3. Adapter progressivement avec vos composants
"""
    
    print(guide)


# =========================================================================
# Main
# =========================================================================

async def main():
    """Point d'entrée principal."""
    print("\n" + "=" * 70)
    print("SETUP PIPELINE RAG AVEC DONNÉES RÉELLES")
    print("=" * 70)
    
    # Afficher le guide
    print_integration_guide()
    
    # Setup et test
    try:
        pipeline, documents = await setup_complete_pipeline()
        
        print(f"\n✅ Pipeline prête avec {len(documents)} documents indexés")
        
        # Test
        await test_pipeline(pipeline)
        
        print("\n" + "=" * 70)
        print("✨ SETUP TERMINÉ AVEC SUCCÈS")
        print("=" * 70)
        print("\n💡 Prochaines étapes:")
        print("   1. Remplacer MockEmbedder par un vrai modèle")
        print("   2. Utiliser Milvus/Qdrant pour l'index")
        print("   3. Ajouter un vrai cross-encoder pour le reranking")
        print("   4. Charger vos bulletins depuis bulletins_json/")
        
    except Exception as e:
        print(f"\n✗ Erreur durant le setup: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())