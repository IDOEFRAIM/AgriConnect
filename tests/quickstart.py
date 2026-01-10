# rag/quick_start.py
"""
🚀 QUICK START - Pipeline RAG en 5 Minutes
===========================================

Ce script vous permet de tester la pipeline RAG immédiatement,
sans configuration complexe.

Usage:
    python rag/quick_start.py

Ou avec vos propres questions:
    python rag/quick_start.py "Votre question ici"
"""

from __future__ import annotations
import asyncio
import sys
import json
from pathlib import Path

# Setup path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# =========================================================================
# Import des composants
# =========================================================================

try:
    from rag.utils.mockExample import (
        MockEmbedder,
        MockRetriever,
        MockReranker,
        SAMPLE_BULLETINS
    )
    from rag.contextaugmentedv9 import (
        AugmentationPipeline,
        AugmentationConfig
    )
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("\nAssurez-vous que les fichiers suivants existent:")
    print("  - rag/complete_working_example.py")
    print("  - rag/augmentation_pipeline.py")
    sys.exit(1)


# =========================================================================
# Configuration par défaut
# =========================================================================

DEFAULT_CONFIG = AugmentationConfig(
    top_k=10,
    rerank_top_n=5,
    max_snippets_per_doc=2,
    use_cross_encoder=True,
    enable_caching=True,
    enable_metrics=True,
    concurrency=4
)


# =========================================================================
# Questions prédéfinies
# =========================================================================

SAMPLE_QUESTIONS = [
    "Quelle est la tendance pluviométrique pour septembre 2025 ?",
    "Quelles sont les prévisions de pluie pour octobre ?",
    "Quelle est la température moyenne en septembre ?",
    "Quel est le cumul de précipitations depuis avril ?",
    "Quelle est la situation des vents ?",
]


# =========================================================================
# Fonction principale
# =========================================================================

async def quick_start(custom_query: str = None):
    """
    Démarrage rapide de la pipeline RAG.
    
    Args:
        custom_query: Question personnalisée (optionnel)
    """
    print("\n" + "🚀" * 35)
    print("QUICK START - PIPELINE RAG")
    print("🚀" * 35)
    
    # 1. Initialisation rapide
    print("\n⚡ Initialisation en cours...", end=" ", flush=True)
    
    embedder = MockEmbedder(dim=128)
    retriever = MockRetriever(SAMPLE_BULLETINS)
    reranker = MockReranker()
    
    pipeline = AugmentationPipeline(
        retriever=retriever,
        encoder=embedder,
        reranker=reranker,
        cfg=DEFAULT_CONFIG
    )
    
    print("✅ Fait!")
    print(f"   📚 {len(SAMPLE_BULLETINS)} documents chargés")
    
    # 2. Choisir la ou les questions
    if custom_query:
        queries = [custom_query]
        print(f"\n❓ Question personnalisée: {custom_query}")
    else:
        print("\n📝 Questions d'exemple disponibles:")
        for i, q in enumerate(SAMPLE_QUESTIONS, 1):
            print(f"   {i}. {q}")
        
        print("\n💡 Utilisation: python rag/quick_start.py \"Votre question\"")
        print("   Ou testez toutes les questions avec Enter...", end=" ", flush=True)
        
        # Demander à l'utilisateur
        if sys.stdin.isatty():  # Si en mode interactif
            try:
                choice = input()
                if choice.strip().isdigit():
                    idx = int(choice.strip()) - 1
                    if 0 <= idx < len(SAMPLE_QUESTIONS):
                        queries = [SAMPLE_QUESTIONS[idx]]
                    else:
                        queries = SAMPLE_QUESTIONS
                elif choice.strip():
                    queries = [choice.strip()]
                else:
                    queries = SAMPLE_QUESTIONS
            except:
                queries = SAMPLE_QUESTIONS
        else:
            queries = SAMPLE_QUESTIONS[:2]  # Par défaut, tester 2 questions
    
    # 3. Exécuter les requêtes
    print(f"\n🔍 Traitement de {len(queries)} requête(s)...\n")
    
    results = []
    for i, query in enumerate(queries, 1):
        print(f"{'─' * 70}")
        print(f"REQUÊTE {i}/{len(queries)}")
        print(f"{'─' * 70}")
        print(f"❓ {query}\n")
        
        # Exécuter
        response = await pipeline.augment(query)
        result = response.to_dict()
        results.append(result)
        
        # Afficher résumé
        print(f"⏱️  Temps: {result['diagnostics']['timings']['total_time_s']:.3f}s")
        print(f"📊 Snippets: {len(result['context']['snippets'])}")
        print(f"📝 Tokens: {result['context']['total_tokens']}")
        
        # Top 3 snippets
        if result['context']['snippets']:
            print(f"\n📄 Top 3 Snippets:\n")
            for j, snippet in enumerate(result['context']['snippets'][:3], 1):
                print(f"   {j}. Score: {snippet['rerank_score']:.3f}")
                print(f"      Doc: {snippet['doc_id']}")
                print(f"      Texte: {snippet['text'][:100]}...")
                print()
        else:
            print("⚠️  Aucun snippet pertinent trouvé\n")
    
    # 4. Métriques globales
    print(f"{'═' * 70}")
    print("📊 MÉTRIQUES GLOBALES")
    print(f"{'═' * 70}\n")
    
    #metrics = pipeline.get_metrics()
   # print(f"✅ Success rate: {metrics['success_rate'] * 100:.1f}%")
    #print(f"⏱️  Latence moyenne: {metrics['avg_latency_s']:.3f}s")
    #print(f"⚡ P95 latence: {metrics['p95_latency_s']:.3f}s")
    
    if len(results) > 0:
        cache = results[-1]['diagnostics']['cache']
        print(f"\n💾 Cache:")
        print(f"   Taille: {cache['cache_size']} entrées")
        print(f"   Hit rate: {cache.get('hit_rate', 0) * 100:.1f}%")
    
    # 5. Sauvegarder le dernier résultat
    if results:
        output_file = "quick_start_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results[-1], f, indent=2, ensure_ascii=False)
        print(f"\n💾 Dernier résultat sauvegardé: {output_file}")
    
    # 6. Suggestions
    print(f"\n{'═' * 70}")
    print("✨ PROCHAINES ÉTAPES")
    print(f"{'═' * 70}\n")
    print("1️⃣  Ajouter vos propres bulletins dans bulletins_json/")
    print("2️⃣  Tester avec: python rag/complete_working_example.py")
    print("3️⃣  Configuration avancée: voir rag/pipe_usage.py")
    print("4️⃣  Benchmarks: voir rag/benchmark/performance_tests.py")
    print("\n💡 Documentation complète dans les fichiers artifact!")
    
    print(f"\n{'🎉' * 35}")
    print("QUICK START TERMINÉ AVEC SUCCÈS!")
    print("🎉" * 35 + "\n")


# =========================================================================
# Mode interactif
# =========================================================================

async def interactive_mode():
    """Mode interactif pour poser plusieurs questions."""
    print("\n" + "💬" * 35)
    print("MODE INTERACTIF")
    print("💬" * 35)
    print("\nPosez vos questions (Ctrl+C ou 'quit' pour sortir)\n")
    
    # Setup pipeline
    embedder = MockEmbedder(dim=128)
    retriever = MockRetriever(SAMPLE_BULLETINS)
    reranker = MockReranker()
    
    pipeline = AugmentationPipeline(
        retriever=retriever,
        encoder=embedder,
        reranker=reranker,
        cfg=DEFAULT_CONFIG
    )
    
    print("✅ Pipeline prête!\n")
    
    while True:
        try:
            query = input("❓ Votre question: ").strip()
            
            if not query or query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Au revoir!")
                break
            
            print("\n🔍 Recherche en cours...", end=" ", flush=True)
            response = await pipeline.augment(query)
            result = response.to_dict()
            print("✅")
            
            # Afficher résultats
            print(f"\n📊 Trouvé {len(result['context']['snippets'])} snippets:")
            print(f"⏱️  Temps: {result['diagnostics']['timings']['total_time_s']:.3f}s\n")
            
            for i, snippet in enumerate(result['context']['snippets'][:3], 1):
                print(f"{i}. [{snippet['rerank_score']:.3f}] {snippet['text'][:150]}...\n")
            
            print()
        
        except KeyboardInterrupt:
            print("\n\n👋 Au revoir!")
            break
        except Exception as e:
            print(f"\n❌ Erreur: {e}\n")


# =========================================================================
# Main
# =========================================================================

async def main():
    """Point d'entrée principal."""
    # Vérifier les arguments
    if len(sys.argv) > 1:
        # Mode avec question en argument
        if sys.argv[1] in ['-i', '--interactive']:
            await interactive_mode()
        else:
            custom_query = ' '.join(sys.argv[1:])
            await quick_start(custom_query)
    else:
        # Mode par défaut
        await quick_start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Au revoir!")
    except Exception as e:
        print(f"\n❌ Erreur fatale: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)