# rag/complete_working_example.py
"""
Exemple complet et fonctionnel de la pipeline RAG avec données de test.
Ce script crée un mini-système RAG opérationnel avec des bulletins météo simulés.
"""

from __future__ import annotations
import asyncio
import json
import hashlib
import numpy as np
from typing import Any, Dict, List, Tuple
from pathlib import Path
import sys

# --- Ensure project root is importable (robust for direct execution) ---
# en haut de rag/utils/mockExample.py
import sys
from pathlib import Path

# remonte jusqu'à la racine du projet (2 niveaux si script est rag/utils)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.contextaugmentedv9 import (
    AugmentationPipeline,
    AugmentationConfig
)


# =========================================================================
# 1. DONNÉES DE TEST - Bulletins Météo Simulés
# =========================================================================

SAMPLE_BULLETINS = [
    {
        "id": "BAD25092_p1",
        "source": "bulletins_json/BAD25092.json",
        "text": (
            "Les hauteurs de pluie décadaires enregistrées du 11 au 20 septembre 2025 "
            "ont varié de 0,0 mm à Dori à 121,9 mm à Bobo-Dioulasso. "
            "Les cumuls saisonniers du 01 avril au 20 septembre 2025 ont fluctué entre "
            "374,6 mm à Korsimoro et 1208,0 mm à Bama."
        ),
        "meta": {
            "date": "2025-09-20",
            "region": "Burkina Faso",
            "type": "bulletin_agro_decadaire"
        }
    },
    {
        "id": "BAD25092_p2",
        "source": "bulletins_json/BAD25092.json",
        "text": (
            "Les précipitations ont été généralement faibles sur l'ensemble du territoire "
            "avec des cumuls inférieurs à 50 mm dans la plupart des stations. "
            "La situation pluviométrique reste déficitaire dans les régions du Sahel et du Nord."
        ),
        "meta": {
            "date": "2025-09-20",
            "region": "Burkina Faso",
            "type": "bulletin_agro_decadaire"
        }
    },
    {
        "id": "PREV_OCT_001",
        "source": "bulletins_json/PREV_OCT.json",
        "text": (
            "Prévisions pour octobre 2025: Les modèles indiquent une reprise des pluies "
            "sur la partie sud du pays avec des cumuls attendus entre 80 et 150 mm. "
            "Les températures maximales devraient se situer entre 32°C et 35°C."
        ),
        "meta": {
            "date": "2025-09-25",
            "region": "Burkina Faso",
            "type": "prevision"
        }
    },
    {
        "id": "TEMP_SEPT_001",
        "source": "bulletins_json/TEMP_SEPT.json",
        "text": (
            "Les températures moyennes de septembre 2025 ont oscillé entre 25°C et 30°C. "
            "Les nuits ont été plus fraîches avec des minimales autour de 22°C. "
            "Aucune vague de chaleur n'a été observée durant ce mois."
        ),
        "meta": {
            "date": "2025-09-30",
            "region": "Burkina Faso",
            "type": "observation"
        }
    },
    {
        "id": "CUMUL_AVRIL_001",
        "source": "bulletins_json/CUMUL_SAISON.json",
        "text": (
            "Bilan pluviométrique saisonnier (avril-septembre 2025): "
            "Les cumuls varient de 374,6 mm (déficitaire) à 1208,0 mm (excédentaire). "
            "La moyenne nationale s'établit à 687 mm, légèrement en dessous de la normale."
        ),
        "meta": {
            "date": "2025-09-30",
            "region": "Burkina Faso",
            "type": "bilan"
        }
    },
    {
        "id": "VENT_SEPT_001",
        "source": "bulletins_json/VENT_SEPT.json",
        "text": (
            "Les vents de septembre ont soufflé principalement du secteur sud-ouest "
            "avec une vitesse moyenne de 15 km/h. Des rafales jusqu'à 45 km/h ont été "
            "enregistrées lors des passages pluvieux."
        ),
        "meta": {
            "date": "2025-09-30",
            "region": "Burkina Faso",
            "type": "observation"
        }
    }
]


# =========================================================================
# 2. MOCK EMBEDDER - Embeddings Déterministes
# =========================================================================

class MockEmbedder:
    """Embedder simple qui génère des embeddings déterministes."""
    
    def __init__(self, dim: int = 128):
        self.dim = dim
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Génère des embeddings basés sur hash du texte."""
        embeddings = []
        for text in texts:
            # Hash du texte pour générer un embedding déterministe
            h = hashlib.sha256(text.encode('utf-8')).digest()
            vec = np.frombuffer(h[:self.dim * 4], dtype=np.float32)[:self.dim]
            # Normalisation L2
            norm = np.linalg.norm(vec) + 1e-12
            vec = vec / norm
            embeddings.append(vec.tolist())
        return embeddings
    
    async def encode_async(self, texts: List[str]) -> List[List[float]]:
        """Version async."""
        return self.encode(texts)


# =========================================================================
# 3. MOCK RETRIEVER - Recherche Simple par Mots-Clés
# =========================================================================

class MockRetriever:
    """Retriever simple basé sur matching de mots-clés."""
    
    def __init__(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        self.embedder = MockEmbedder()
        
        # Pré-calculer les embeddings
        self.doc_embeddings = {
            doc['id']: self.embedder.encode([doc['text']])[0]
            for doc in documents
        }
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calcule la similarité cosine."""
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
    
    async def retrieve(self, query: str, top_k: int = 50, **kwargs) -> List[Dict[str, Any]]:
        """Récupère les documents pertinents."""
        # Encoder la query
        query_emb = self.embedder.encode([query])[0]
        
        # Calculer les scores
        scored_docs = []
        for doc in self.documents:
            doc_emb = self.doc_embeddings[doc['id']]
            score = self._cosine_similarity(query_emb, doc_emb)
            
            # Bonus pour matching de mots-clés
            query_words = set(query.lower().split())
            doc_words = set(doc['text'].lower().split())
            keyword_bonus = len(query_words & doc_words) * 0.1
            
            final_score = score + keyword_bonus
            
            scored_docs.append({
                'id': doc['id'],
                'doc_id': doc['id'],
                'text': doc['text'],
                'source': doc['source'],
                'meta': doc['meta'],
                'score': final_score
            })
        
        # Trier et retourner top_k
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        return scored_docs[:top_k]


# =========================================================================
# 4. MOCK CROSS-ENCODER - Reranking Basique
# =========================================================================

class MockCrossEncoder:
    """Cross-encoder simple pour le reranking."""
    
    def predict_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Calcule des scores de pertinence query-document."""
        scores = []
        for query, doc in pairs:
            # Score basé sur overlap de mots + longueur
            query_words = set(query.lower().split())
            doc_words = set(doc.lower().split())
            
            overlap = len(query_words & doc_words)
            total = len(query_words | doc_words)
            
            if total == 0:
                scores.append(0.0)
            else:
                # Score = jaccard + bonus pour docs plus longs
                jaccard = overlap / total
                length_bonus = min(len(doc.split()) / 100, 0.3)
                scores.append(jaccard + length_bonus)
        
        return scores


# =========================================================================
# 5. MOCK RERANKER - Wrapper pour Cross-Encoder
# =========================================================================

class MockReranker:
    """Reranker qui utilise le cross-encoder."""
    
    def __init__(self):
        self.cross_encoder = MockCrossEncoder()
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int = 20
    ) -> List[Dict[str, Any]]:
        """Rerank synchrone."""
        pairs = [(query, c.get('text', '')) for c in candidates]
        scores = self.cross_encoder.predict_batch(pairs)
        
        for doc, score in zip(candidates, scores):
            doc['cross_score'] = score
            doc['combined_score'] = 0.6 * score + 0.4 * doc.get('score', 0.0)
        
        ranked = sorted(candidates, key=lambda x: x['combined_score'], reverse=True)
        return ranked[:top_n]
    
    async def rerank_async(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int = 20,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Rerank asynchrone."""
        return self.rerank(query, candidates, top_n)


# =========================================================================
# 6. FONCTION PRINCIPALE - Setup et Test
# =========================================================================

async def main():
    print("=" * 70)
    print("EXEMPLE COMPLET FONCTIONNEL - PIPELINE RAG")
    print("=" * 70)
    
    # 1. Initialiser les composants
    print("\n📦 Initialisation des composants...")
    embedder = MockEmbedder(dim=128)
    retriever = MockRetriever(SAMPLE_BULLETINS)
    reranker = MockReranker()
    
    print(f"  ✓ Embedder: {embedder.dim} dimensions")
    print(f"  ✓ Retriever: {len(SAMPLE_BULLETINS)} documents indexés")
    print(f"  ✓ Reranker: Cross-encoder activé")
    
    # 2. Créer la pipeline avec configuration optimale
    print("\n⚙️  Configuration de la pipeline...")
    config = AugmentationConfig(
        top_k=10,
        rerank_top_n=5,
        max_snippets_per_doc=2,
        snippet_max_tokens=150,
        min_snippet_score=0.1,
        
        # Scoring
        retrieval_weight=0.3,
        rerank_weight=0.5,
        semantic_weight=0.2,
        diversity_penalty=0.1,
        
        # Performance
        concurrency=4,
        batch_size=16,
        enable_caching=True,
        cache_ttl_s=3600.0,
        
        # Quality
        enable_deduplication=True,
        dedup_threshold=0.85,
        enable_quality_filter=True,
        min_text_length=20,
        
        # Cross-encoder
        use_cross_encoder=True,
        
        # Timeouts
        timeout_s=10.0,
        retrieval_timeout_s=3.0,
        rerank_timeout_s=2.0,
        encode_timeout_s=2.0,
        
        # Monitoring
        enable_metrics=True,
        log_slow_queries=True,
        slow_query_threshold_s=2.0
    )
    
    pipeline = AugmentationPipeline(
        retriever=retriever,
        encoder=embedder,
        reranker=reranker,
        cfg=config
    )
    
    print("  ✓ Pipeline configurée")
    
    # 3. Tester avec différentes requêtes
    test_queries = [
        "Quelle est la tendance pluviométrique pour la période 11-20 septembre 2025 ?",
        "Quelles sont les prévisions de pluie pour octobre 2025 ?",
        "Quelle est la température moyenne en septembre 2025 ?",
        "Quel est le cumul de précipitations depuis avril 2025 ?",
        "Quelle est la situation des vents en septembre ?",
    ]
    
    print(f"\n🔍 Test avec {len(test_queries)} requêtes...\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 70}")
        print(f"REQUÊTE {i}/{len(test_queries)}")
        print(f"{'=' * 70}")
        print(f"❓ {query}\n")
        
        # Exécuter la pipeline
        response = await pipeline.augment(query)
        
        # Afficher les résultats
        result = response.to_dict()
        
        print(f"📊 RÉSULTATS:")
        print(f"  ✓ Snippets retournés: {len(result['context']['snippets'])}")
        print(f"  ✓ Total tokens: {result['context']['total_tokens']}")
        print(f"  ✓ Temps total: {result['diagnostics']['timings']['total_time_s']}s")
        print(f"  ✓ Cache hit rate: {result['diagnostics']['cache'].get('hit_rate', 0) * 100:.1f}%")
        
        # Afficher les snippets
        if result['context']['snippets']:
            print(f"\n📄 SNIPPETS TROUVÉS:")
            for j, snippet in enumerate(result['context']['snippets'][:3], 1):
                print(f"\n  {j}. Document: {snippet['doc_id']}")
                print(f"     Source: {snippet['citation']}")
                print(f"     Score: {snippet['rerank_score']:.3f}")
                print(f"     Texte: {snippet['text'][:150]}...")
        else:
            print("\n  ⚠️  Aucun snippet trouvé")
        
        # Diagnostics détaillés
        print(f"\n⚙️  DIAGNOSTICS:")
        for step_name, step_data in result['diagnostics']['steps'].items():
            print(f"  - {step_name}: {step_data.get('count', 'N/A')} items en {step_data.get('time_s', 0):.3f}s")
        
        # Warnings/Errors
        if result['diagnostics'].get('warnings'):
            print(f"\n  ⚠️  Warnings: {len(result['diagnostics']['warnings'])}")
            for w in result['diagnostics']['warnings']:
                print(f"      - {w}")
        
        if result['diagnostics'].get('errors'):
            print(f"\n  ✗ Errors: {len(result['diagnostics']['errors'])}")
            for e in result['diagnostics']['errors']:
                print(f"      - {e}")
    
    # 4. Afficher les métriques globales
    print(f"\n\n{'=' * 70}")
    print("MÉTRIQUES GLOBALES DE LA PIPELINE")
    print(f"{'=' * 70}")
    
    #metrics = pipeline.get_metrics()
   # print(f"\n📈 Performance:")
    #print(f"  Total queries: {metrics['total_queries']}")
    #print(f"  Success rate: {metrics['success_rate'] * 100:.1f}%")
    #print(f"  Avg latency: {metrics['avg_latency_s']:.3f}s")
    #print(f"  P95 latency: {metrics['p95_latency_s']:.3f}s")
    
    print(f"\n💾 Cache:")
    cache_info = result['diagnostics']['cache']
    print(f"  Size: {cache_info['cache_size']} entries")
    print(f"  Hit rate: {cache_info.get('hit_rate', 0) * 100:.1f}%")
    print(f"  Hits: {cache_info['cache_hits']}")
    print(f"  Misses: {cache_info['cache_misses']}")
    
    # 5. Sauvegarder un exemple de réponse complète
    print(f"\n\n{'=' * 70}")
    print("EXEMPLE DE RÉPONSE COMPLÈTE (JSON)")
    print(f"{'=' * 70}\n")
    
    # Re-run une requête pour avoir une réponse complète
    final_response = await pipeline.augment(test_queries[0])
    final_json = final_response.to_dict()
    
    print(json.dumps(final_json, indent=2, ensure_ascii=False))
    
    # Sauvegarder dans un fichier
    output_file = "example_augmented_response.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Réponse sauvegardée dans: {output_file}")
    
    print(f"\n{'=' * 70}")
    print("✨ DÉMONSTRATION COMPLÈTE TERMINÉE")
    print(f"{'=' * 70}\n")


# =========================================================================
# Point d'entrée
# =========================================================================

if __name__ == "__main__":
    asyncio.run(main())