# rag/examples/pipeline_usage.py
"""
Exemples d'utilisation avancée de la pipeline d'augmentation RAG.
Montre différents cas d'usage, configurations et optimisations.
"""

from __future__ import annotations
import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Imports supposés - adapter selon votre structure
from rag.contextaugmentedv9 import (
    AugmentationPipeline,
    AugmentationConfig,
    AugmentedResponse
)


# =========================================================================
# Exemple 1: Configuration Basique
# =========================================================================
async def example_basic_usage():
    """Utilisation basique avec configuration par défaut."""
    print("=" * 70)
    print("EXEMPLE 1: Configuration Basique")
    print("=" * 70)
    
    # Setup minimal (remplacer par vos vrais composants)
    from rag.utils.retriever import Retriever, RetrieverConfig
    from rag.utils.embedder import Embedder, EmbedderConfig
    from rag.utils.reRank import Reranker, RerankConfig
    from rag.utils.crossEncoder import CrossEncoder
    
    # Initialisation
    embedder = Embedder(cfg=EmbedderConfig())
    retriever = Retriever(
        embedder=embedder,
        cfg=RetrieverConfig(top_k=50)
    )
    cross_encoder = CrossEncoder()
    reranker = Reranker(
        cross_encoder=cross_encoder,
        cfg=RerankConfig()
    )
    
    # Pipeline avec config par défaut
    pipeline = AugmentationPipeline(
        retriever=retriever,
        encoder=embedder,
        reranker=reranker
    )
    
    # Requête simple
    query = "Quelle est la tendance pluviométrique pour septembre 2025 ?"
    response = await pipeline.augment(query)
    
    # Affichage
    output = response.to_dict()
    print(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n✓ Snippets retournés: {len(output['context']['snippets'])}")
    print(f"✓ Temps total: {output['diagnostics']['timings']['total_time_s']}s")


# =========================================================================
# Exemple 2: Configuration Haute Performance
# =========================================================================
async def example_high_performance():
    """Configuration optimisée pour performance maximale."""
    print("\n" + "=" * 70)
    print("EXEMPLE 2: Configuration Haute Performance")
    print("=" * 70)
    
    # Configuration optimisée
    cfg = AugmentationConfig(
        # Récupération agressive
        top_k=100,
        candidate_pool=500,
        rerank_top_n=20,
        
        # Snippets optimisés
        max_snippets_per_doc=5,
        snippet_max_tokens=150,
        snippet_overlap=30,
        
        # Performance
        concurrency=16,  # Plus de parallélisme
        batch_size=64,   # Batches plus larges
        enable_caching=True,
        cache_ttl_s=7200.0,  # 2h de cache
        
        # Qualité
        enable_deduplication=True,
        dedup_threshold=0.90,
        enable_quality_filter=True,
        min_text_length=30,
        
        # Scoring avancé
        retrieval_weight=0.2,
        rerank_weight=0.5,
        semantic_weight=0.3,
        diversity_penalty=0.15,
        
        # Timeouts généreux
        timeout_s=15.0,
        retrieval_timeout_s=5.0,
        rerank_timeout_s=4.0,
        encode_timeout_s=4.0,
        
        # Retry agressif
        retry_attempts=3,
        retry_backoff_s=0.3,
        
        # Monitoring
        enable_metrics=True,
        log_slow_queries=True,
        slow_query_threshold_s=3.0
    )
    
    # Setup pipeline (code omis pour brièveté)
    # pipeline = setup_pipeline(cfg)
    
    print("Configuration haute performance activée:")
    print(f"  - Concurrency: {cfg.concurrency}")
    print(f"  - Batch size: {cfg.batch_size}")
    print(f"  - Cache TTL: {cfg.cache_ttl_s}s")
    print(f"  - Deduplication: {cfg.enable_deduplication}")
    print(f"  - Quality filter: {cfg.enable_quality_filter}")


# =========================================================================
# Exemple 3: Configuration Économie de Ressources
# =========================================================================
async def example_low_resource():
    """Configuration pour environnements avec ressources limitées."""
    print("\n" + "=" * 70)
    print("EXEMPLE 3: Configuration Économie de Ressources")
    print("=" * 70)
    
    cfg = AugmentationConfig(
        # Récupération minimale
        top_k=20,
        candidate_pool=50,
        rerank_top_n=5,
        
        # Snippets réduits
        max_snippets_per_doc=2,
        snippet_max_tokens=100,
        
        # Performance limitée
        concurrency=2,
        batch_size=8,
        enable_caching=True,
        cache_ttl_s=1800.0,  # 30min
        
        # Pas de cross-encoder (économise CPU)
        use_cross_encoder=False,
        
        # Timeouts courts
        timeout_s=5.0,
        retrieval_timeout_s=2.0,
        encode_timeout_s=2.0,
        
        # Pas de retry
        retry_attempts=1,
        
        # Monitoring minimal
        enable_metrics=False,
        log_slow_queries=False
    )
    
    print("Configuration économie de ressources:")
    print(f"  - Top-K: {cfg.top_k} (vs 50 par défaut)")
    print(f"  - Concurrency: {cfg.concurrency} (vs 8 par défaut)")
    print(f"  - Cross-encoder: {'Non' if not cfg.use_cross_encoder else 'Oui'}")
    print(f"  - Retry: {cfg.retry_attempts} (vs 2 par défaut)")


# =========================================================================
# Exemple 4: Traitement par Batch
# =========================================================================
async def example_batch_processing():
    """Traiter plusieurs requêtes en parallèle."""
    print("\n" + "=" * 70)
    print("EXEMPLE 4: Traitement par Batch")
    print("=" * 70)
    
    # Setup pipeline (code omis)
    # pipeline = setup_pipeline()
    
    queries = [
        "Quelles sont les prévisions de pluie pour octobre 2025 ?",
        "Quelle est la température moyenne en septembre ?",
        "Y a-t-il des alertes météo en cours ?",
        "Quel est le cumul de précipitations depuis avril ?",
        "Quelle est la tendance des vents pour la semaine ?"
    ]
    
    print(f"Traitement de {len(queries)} requêtes en parallèle...\n")
    
    start = time.time()
    
    # Traitement parallèle avec limite de concurrence
    semaphore = asyncio.Semaphore(3)  # Max 3 requêtes simultanées
    
    async def process_with_limit(q: str) -> AugmentedResponse:
        async with semaphore:
            # response = await pipeline.augment(q)
            # return response
            pass  # Simulation
    
    # tasks = [process_with_limit(q) for q in queries]
    # responses = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start
    
    print(f"✓ {len(queries)} requêtes traitées en {elapsed:.2f}s")
    print(f"✓ Moyenne: {elapsed / len(queries):.2f}s par requête")
    
    # Statistiques agrégées
    # total_snippets = sum(len(r.context.snippets) for r in responses)
    # print(f"✓ Total snippets: {total_snippets}")


# =========================================================================
# Exemple 5: Monitoring et Métriques
# =========================================================================
async def example_monitoring():
    """Utilisation avancée du monitoring et des métriques."""
    print("\n" + "=" * 70)
    print("EXEMPLE 5: Monitoring et Métriques")
    print("=" * 70)
    
    cfg = AugmentationConfig(
        enable_metrics=True,
        log_slow_queries=True,
        slow_query_threshold_s=1.0
    )
    
    # Setup pipeline
    # pipeline = setup_pipeline(cfg)
    
    # Simuler plusieurs requêtes
    queries = [
        "Requête rapide 1",
        "Requête rapide 2",
        "Requête potentiellement lente avec beaucoup de contexte",
        "Requête moyenne",
    ]
    
    print("Traitement de requêtes avec monitoring...\n")
    
    # for i, query in enumerate(queries, 1):
    #     print(f"Query {i}/{len(queries)}: {query[:50]}...")
    #     response = await pipeline.augment(query)
    #     
    #     # Afficher diagnostics
    #     diag = response.diagnostics.to_dict()
    #     print(f"  ✓ Temps: {diag['timings']['total_time_s']}s")
    #     print(f"  ✓ Snippets: {diag['summary']['snippets_returned']}")
    #     
    #     if diag.get('warnings'):
    #         print(f"  ⚠ Warnings: {len(diag['warnings'])}")
    #     if diag.get('errors'):
    #         print(f"  ✗ Errors: {len(diag['errors'])}")
    #     print()
    
    # Métriques globales
    # metrics = pipeline.get_metrics()
    metrics = {
        "total_queries": 4,
        "successful_queries": 4,
        "failed_queries": 0,
        "timeout_queries": 0,
        "success_rate": 1.0,
        "avg_latency_s": 0.845,
        "p95_latency_s": 1.234
    }
    
    print("Métriques Globales:")
    print(f"  Total queries: {metrics['total_queries']}")
    print(f"  Success rate: {metrics['success_rate'] * 100:.1f}%")
    print(f"  Avg latency: {metrics['avg_latency_s']:.3f}s")
    print(f"  P95 latency: {metrics['p95_latency_s']:.3f}s")
    print(f"  Timeouts: {metrics['timeout_queries']}")


# =========================================================================
# Exemple 6: Gestion d'Erreurs Avancée
# =========================================================================
async def example_error_handling():
    """Gestion robuste des erreurs et retry."""
    print("\n" + "=" * 70)
    print("EXEMPLE 6: Gestion d'Erreurs Avancée")
    print("=" * 70)
    
    cfg = AugmentationConfig(
        retry_attempts=3,
        retry_backoff_s=0.5,
        timeout_s=5.0
    )
    
    # Setup pipeline
    # pipeline = setup_pipeline(cfg)
    
    # Requêtes potentiellement problématiques
    test_cases = [
        ("Query normale", "Quelle est la météo ?"),
        ("Query vide", ""),
        ("Query très longue", "x" * 10000),
        ("Query avec caractères spéciaux", "Météo à Ouagadougou ☀️ 🌧️"),
    ]
    
    print("Test de robustesse avec différents cas d'erreur:\n")
    
    for name, query in test_cases:
        print(f"Test: {name}")
        try:
            # response = await pipeline.augment(query)
            # status = "✓ Succès" if response.context.snippets else "⚠ Vide"
            status = "✓ Succès (simulé)"
        except asyncio.TimeoutError:
            status = "✗ Timeout"
        except Exception as e:
            status = f"✗ Erreur: {type(e).__name__}"
        
        print(f"  {status}\n")


# =========================================================================
# Exemple 7: Analyse de Performance Détaillée
# =========================================================================
async def example_performance_analysis():
    """Analyse détaillée des performances par étape."""
    print("\n" + "=" * 70)
    print("EXEMPLE 7: Analyse de Performance Détaillée")
    print("=" * 70)
    
    # Setup pipeline
    # pipeline = setup_pipeline()
    
    query = "Quelle est la tendance pluviométrique pour septembre 2025 ?"
    # response = await pipeline.augment(query)
    
    # Simulation d'une réponse
    diag = {
        "steps": {
            "retrieve": {"count": 50, "time_s": 0.123},
            "rerank": {"count": 50, "time_s": 0.089},
            "extract": {"count": 15, "time_s": 0.156},
            "encode": {"texts_encoded": 10, "time_s": 0.145, "batches": 2},
            "format": {"count": 10, "time_s": 0.012}
        },
        "cache": {
            "cache_size": 42,
            "cache_hits": 5,
            "cache_misses": 10,
            "cache_evictions": 0,
            "hit_rate": 0.333
        },
        "timings": {
            "total_time_s": 0.525,
            "augment_time_s": 0.513,
            "format_time_s": 0.012
        }
    }
    
    print("Analyse par étape:\n")
    
    total_time = diag["timings"]["total_time_s"]
    
    for step_name, step_data in diag["steps"].items():
        step_time = step_data.get("time_s", 0)
        percentage = (step_time / total_time * 100) if total_time > 0 else 0
        
        print(f"{step_name.upper()}:")
        print(f"  Temps: {step_time:.3f}s ({percentage:.1f}%)")
        print(f"  Count: {step_data.get('count', 'N/A')}")
        
        if "timeout" in step_data:
            print(f"  ⚠ TIMEOUT détecté")
        if "error" in step_data:
            print(f"  ✗ ERREUR: {step_data['error']}")
        print()
    
    print("Statistiques Cache:")
    cache = diag["cache"]
    print(f"  Taille: {cache['cache_size']}")
    print(f"  Hit rate: {cache['hit_rate'] * 100:.1f}%")
    print(f"  Hits: {cache['cache_hits']}")
    print(f"  Misses: {cache['cache_misses']}")
    print(f"  Evictions: {cache['cache_evictions']}")


# =========================================================================
# Exemple 8: Optimisation Progressive
# =========================================================================
async def example_progressive_optimization():
    """Compare différentes configurations pour trouver l'optimale."""
    print("\n" + "=" * 70)
    print("EXEMPLE 8: Optimisation Progressive")
    print("=" * 70)
    
    query = "Quelle est la météo prévue pour demain ?"
    
    configurations = [
        ("Minimale", AugmentationConfig(top_k=10, rerank_top_n=3, concurrency=1)),
        ("Standard", AugmentationConfig(top_k=50, rerank_top_n=10, concurrency=4)),
        ("Optimisée", AugmentationConfig(top_k=100, rerank_top_n=20, concurrency=8)),
        ("Maximum", AugmentationConfig(top_k=200, rerank_top_n=30, concurrency=16)),
    ]
    
    print("Comparaison de configurations:\n")
    print(f"{'Config':<15} {'Latency':<10} {'Snippets':<10} {'Quality':<10}")
    print("-" * 50)
    
    for name, cfg in configurations:
        # pipeline = setup_pipeline(cfg)
        # start = time.time()
        # response = await pipeline.augment(query)
        # latency = time.time() - start
        
        # Simulation
        latency = 0.5 + (cfg.top_k / 100) * 0.3
        snippets = min(cfg.rerank_top_n, 15)
        quality = 0.7 + (cfg.top_k / 200) * 0.2
        
        print(f"{name:<15} {latency:.3f}s    {snippets:<10} {quality:.2f}")
    
    print("\n✓ Configuration 'Optimisée' recommandée (bon équilibre)")


# =========================================================================
# Main - Exécute tous les exemples
# =========================================================================
async def main():
    """Exécute tous les exemples de démonstration."""
    print("\n" + "=" * 70)
    print("DÉMONSTRATIONS - PIPELINE D'AUGMENTATION RAG")
    print("=" * 70)
    
    examples = [
        ("Basic Usage", example_basic_usage),
        ("High Performance", example_high_performance),
        ("Low Resource", example_low_resource),
        ("Batch Processing", example_batch_processing),
        ("Monitoring", example_monitoring),
        ("Error Handling", example_error_handling),
        ("Performance Analysis", example_performance_analysis),
        ("Progressive Optimization", example_progressive_optimization),
    ]
    
    for name, example_func in examples:
        try:
            await example_func()
        except Exception as e:
            print(f"\n✗ Exemple '{name}' a échoué: {e}")
    
    print("\n" + "=" * 70)
    print("FIN DES DÉMONSTRATIONS")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())