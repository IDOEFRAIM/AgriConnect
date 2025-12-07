# rag/augretr.py
from __future__ import annotations
import asyncio
import logging
import json
import sys
from pathlib import Path
from dataclasses import asdict
from typing import Optional

# ensure project root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("augretr")

# -------------------------
# Import project components
# -------------------------
# Retriever (try common locations / names)
Retriever = None
RetrieverConfig = None
try:
    from rag.utils.retriever import Retriever, RetrieverConfig  # preferred
except Exception:
    try:
        from rag.utils.retriever import Retriever, RetrieverConfig  # alternate
    except Exception:
        _logger.error("Impossible d'importer rag.retriever. Vérifie rag/retriever.py et rag/__init__.py")
        Retriever = None
        RetrieverConfig = None

# Embedder (try common names)
embedder: Optional[object] = None
try:
    from rag.utils.embedder import Embedder  # common name
    embedder = Embedder()
except Exception:
    try:
        from rag.utils.embedder import DummyEmbedder
        embedder = DummyEmbedder(dim=128)
    except Exception:
        embedder = None

# Reranker (try common names)
reranker: Optional[object] = None
try:
    # try common module name
    from rag.utils.reRank import Reranker, RerankConfig
    try:
        reranker = Reranker()
    except Exception:
        reranker = None
except Exception:
    try:
        from rag.utils.reRank import Reranker, RerankConfig
        try:
            reranker = Reranker()
        except Exception:
            reranker = None
    except Exception:
        reranker = None

# Pipeline import (correct module name)
try:
    from test.augmentationPipeline import AugmentPipeline, PipelineConfig, AugmentResult
except Exception:
    try:
        # fallback if file named differently
        from test.augmentationPipeline import AugmentPipeline, PipelineConfig, AugmentResult
    except Exception:
        _logger.error("Impossible d'importer AugmentPipeline depuis rag.augment_pipeline. Vérifie le fichier et ses exports.")
        raise

# -------------------------
# Main
# -------------------------
async def main():
    # Validate embedder
    if embedder is None:
        _logger.error("Aucun embedder trouvé. Crée ou exporte une classe Embedder dans rag/embedder.py")
        return

    # Validate retriever class availability
    if Retriever is None:
        _logger.error("Aucune classe Retriever disponible. Corrige l'import de rag.retriever avant de continuer.")
        return

    # Instantiate retriever (adapt to your constructor)
    try:
        try:
            # common constructor with config
            if RetrieverConfig is not None:
                retr_cfg = RetrieverConfig(top_k=50, candidate_pool=200)
                retriever = Retriever(embedder=embedder, indexer=getattr(Retriever, "default_indexer", None), cfg=retr_cfg, collection="default")
            else:
                retriever = Retriever(embedder=embedder)
        except TypeError:
            # fallback: try simpler constructor
            retriever = Retriever(embedder=embedder)
    except Exception:
        _logger.exception("Impossible d'instancier Retriever automatiquement. Adapte ce script à la signature de ton Retriever.")
        return

    # Instantiate pipeline
    pipeline = AugmentPipeline(
        retriever=retriever,
        encoder=embedder,
        reranker=reranker,
        cfg=PipelineConfig(index_batch_size=64, index_concurrency=2),
        augmenter_cfg=None,
    )

    # Optional: index existing JSON files (bulletins_json)
    try:
        # prefer build_and_index_from_folder if available, else try index_records via load_folder
        if hasattr(pipeline, "build_and_index_from_folder"):
            idx_summary = await pipeline.build_and_index_from_folder("bulletins_json")
        else:
            docs = pipeline.load_folder("bulletins_json")
            idx_summary = await pipeline.index_records(docs)
        _logger.info("Index summary: %s", idx_summary)
    except Exception as e:
        _logger.warning("Index skipped or failed: %s", e)

    # Run a test query
    query = "Quelle est la tendance pluviométrique pour la période 11-20 septembre 2025 ?"
    try:
        res: AugmentResult = await pipeline.augment_query(query, top_k=10, timeout=30.0, use_cross_encoder=True)
    except Exception:
        _logger.exception("Erreur lors de l'appel augment_query")
        return

    # Print results
    print("\n=== AUGMENTED CONTEXT ===")
    print("Query:", res.context.query)
    print("Snippets returned:", len(res.context.snippets))
    for s in res.context.snippets:
        print(f"- {s.source} | retrieval={s.retrieval_score:.3f} rerank={s.rerank_score:.3f} | {s.text[:200]}")
    print("\nDiagnostics:")
    print(json.dumps(res.diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # Run the async main safely. If running inside an already running loop (e.g., some IDEs),
    # use asyncio.run only when safe; here we assume script executed from a normal Python process.
    try:
        asyncio.run(main())
    except RuntimeError as e:
        # If event loop already running, run main in the existing loop
        _logger.warning("Event loop already running; scheduling main() in current loop.")
        loop = asyncio.get_event_loop()
        task = loop.create_task(main())
        # Wait briefly for completion (best-effort)
        try:
            loop.run_until_complete(task)
        except Exception:
            _logger.exception("Failed to run main in existing loop.")