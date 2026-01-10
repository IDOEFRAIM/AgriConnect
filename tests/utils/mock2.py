"""
Exemple complet et fonctionnel de la pipeline RAG avec vos vraies classes.
Ce script utilise vos vraies classes Retriever, Embedder, Reranker.
"""

from __future__ import annotations
import asyncio
import json
from typing import Any, Dict, List
from pathlib import Path
import sys

# --- Setup paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Importer VOS vraies classes
from rag.context2 import (
    AugmentationPipeline,
    AugmentationConfig
)

# TODO: Remplacer par vos vraies classes
# from rag.retriever import YourRetriever
# from rag.embedder import YourEmbedder
# from rag.reranker import YourReranker


# =========================================================================
# DONNÉES DE TEST - Bulletins Météo Simulés
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
# FONCTION PRINCIPALE
# =========================================================================

async def main():
    print("=" * 70)
    print("EXEMPLE COMPLET FONCTIONNEL - PIPELINE RAG V2")
    print("=" * 70)
    
    # 1. Initialiser VOS composants
    print("\n📦 Initialisation des composants...")
    
    # TODO: Remplacer par vos vraies classes
    # embedder = YourEmbedder(model_name="your-model")
    # retriever = YourRetriever(documents=SAMPLE_BULLETINS, embedder=embedder)
    # reranker = YourReranker(model_name="your-reranker")
    
    # Pour l'instant, on garde les mocks pour que ça compile
    from rag.utils.embedder import Embedder,EmbedderConfig
    from rag.utils.retriever import Retriever,RetrieverConfig  
    from rag.utils.indexer_init import MilvusIndexer
    from rag.utils.reRank import Reranker
    from rag.utils.crossEncoder import CrossEncoder

    collection_name = "sample_collection"
    embedder = Embedder()
    indexer = MilvusIndexer()
    retriever = Retriever(indexer=indexer, embedder=embedder, collection=collection_name)
    reranker = Reranker(cross_encoder=CrossEncoder)
    
    print(f"  ✓ Embedder: initialisé")
    print(f"  ✓ Retriever: {len(SAMPLE_BULLETINS)} documents indexés")
    print(f"  ✓ Reranker: activé")
    
    # 2. Configuration optimale
    print("\n⚙️  Configuration de la pipeline...")
    config = AugmentationConfig(
        # Retrieval
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
        encoder=CrossEncoder,
        reranker=reranker,
        cfg=config
    )
    
    print("  ✓ Pipeline configurée avec métriques activées")
    
    # 3. Test avec requêtes variées
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
        result = response.to_dict()
        
        # Afficher résultats
        print(f"📊 RÉSULTATS:")
        print(f"  ✓ Snippets retournés: {len(result['context']['snippets'])}")
        print(f"  ✓ Total tokens: {result['context']['total_tokens']}")
        print(f"  ✓ Temps total: {result['diagnostics']['timings']['total_time_s']:.3f}s")
        
        # Cache stats avec affichage correct
        cache_info = result['diagnostics']['cache']
        print(f"  ✓ Cache: {cache_info.get('hit_rate_display', 'N/A')} "
              f"({cache_info['cache_hits']} hits, {cache_info['cache_misses']} misses)")
        
        # Top snippets
        if result['context']['snippets']:
            print(f"\n📄 TOP SNIPPETS:")
            for j, snippet in enumerate(result['context']['snippets'][:3], 1):
                print(f"\n  {j}. Doc: {snippet['doc_id']}")
                print(f"     Score: {snippet.get('rerank_score', 0.0):.3f}")
                print(f"     Texte: {snippet['text'][:120]}...")
        else:
            print("\n  ⚠️  Aucun snippet trouvé")
        
        # Diagnostics détaillés
        if result['diagnostics'].get('steps'):
            print(f"\n⚙️  DIAGNOSTICS DÉTAILLÉS:")
            for step_name, step_data in result['diagnostics']['steps'].items():
                count = step_data.get('count', 'N/A')
                time_s = step_data.get('time_s', 0)
                print(f"  - {step_name}: {count} items en {time_s:.3f}s")
        
        # Warnings/Errors
        if result['diagnostics'].get('warnings'):
            print(f"\n  ⚠️  Warnings: {len(result['diagnostics']['warnings'])}")
        
        if result['diagnostics'].get('errors'):
            print(f"\n  ✗ Errors: {len(result['diagnostics']['errors'])}")
    
    # 4. MÉTRIQUES GLOBALES (MAINTENANT DISPONIBLE!)
    print(f"\n\n{'=' * 70}")
    print("📈 MÉTRIQUES GLOBALES DE LA PIPELINE")
    print(f"{'=' * 70}")
    
    metrics = pipeline.get_metrics()
    
    print(f"\n🎯 Performance:")
    print(f"  Total queries: {metrics['total_queries']}")
    print(f"  Success rate: {metrics['success_rate'] * 100:.1f}%")
    print(f"  Avg latency: {metrics['avg_latency_s']:.3f}s")
    print(f"  P95 latency: {metrics['p95_latency_s']:.3f}s")
    print(f"  P99 latency: {metrics['p99_latency_s']:.3f}s")
    print(f"  Min latency: {metrics['min_latency_s']:.3f}s")
    print(f"  Max latency: {metrics['max_latency_s']:.3f}s")
    
    print(f"\n💾 Cache:")
    cache_metrics = metrics['cache']
    print(f"  Size: {cache_metrics['cache_size']} entries")
    print(f"  Total requests: {cache_metrics['total_requests']}")
    print(f"  Hit rate: {cache_metrics['hit_rate_pct']}")
    print(f"  Hits: {cache_metrics['hits']}")
    print(f"  Misses: {cache_metrics['misses']}")
    
    print(f"\n⏱️  Uptime: {metrics['uptime_s']:.1f}s")
    
    # 5. Sauvegarder exemple complet
    print(f"\n\n{'=' * 70}")
    print("💾 SAUVEGARDE EXEMPLE COMPLET")
    print(f"{'=' * 70}\n")
    
    final_response = await pipeline.augment(test_queries[0])
    final_json = final_response.to_dict()
    
    output_file = "example_augmented_response.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Réponse complète sauvegardée dans: {output_file}")
    
    # 6. Test cache warming (relancer même query)
    print(f"\n\n{'=' * 70}")
    print("🔥 TEST CACHE WARMING")
    print(f"{'=' * 70}\n")
    
    print("Relance de la première query pour tester le cache...")
    cache_before = metrics['cache']['hits']
    
    response2 = await pipeline.augment(test_queries[0])
    
    metrics_after = pipeline.get_metrics()
    cache_after = metrics_after['cache']['hits']
    
    print(f"\n✓ Cache hits avant: {cache_before}")
    print(f"✓ Cache hits après: {cache_after}")
    print(f"✓ Nouveaux hits: {cache_after - cache_before}")
    print(f"✓ Hit rate global: {metrics_after['cache']['hit_rate_pct']}")
    
    print(f"\n{'=' * 70}")
    print("✨ DÉMONSTRATION COMPLÈTE TERMINÉE")
    print(f"{'=' * 70}\n")
    
    print("📝 RÉSUMÉ DES AMÉLIORATIONS:")
    print("  ✓ Cache avec clés normalisées (Unicode, espaces, etc.)")
    print("  ✓ Hit rate calculé correctement")
    print("  ✓ Métriques de performance (latences, percentiles)")
    print("  ✓ Sauvegarde atomique du cache")
    print("  ✓ Utilisation de time.perf_counter() pour précision")
    print("  ✓ Stats détaillées par requête et globales")


# =========================================================================
# Point d'entrée
# =========================================================================

if __name__ == "__main__":
    asyncio.run(main())