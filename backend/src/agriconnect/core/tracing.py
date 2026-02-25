"""
Tracing — Intégration LangSmith pour AgriConnect.
===================================================

Ce module fournit :
1. `init_tracing()`   → Initialise le client LangSmith + valide la connexion.
2. `trace_agent()`    → Décorateur pour tracer n'importe quelle fonction agent.
3. `get_ls_client()`  → Retourne le client LangSmith singleton.
4. `create_feedback()` → Enregistre un feedback (score) sur un run.
5. `TracingCallbackHandler` → Handler LangChain injectable dans les .invoke().

Usage dans un agent :
    from backend.core.tracing import trace_agent

    @trace_agent(name="sentinelle.analyze", tags=["weather", "alerts"])
    def my_function(state):
        ...

Usage pour tracer un .invoke() avec métadonnées custom :
    from backend.core.tracing import get_tracing_config
    result = graph.invoke(state, get_tracing_config(run_name="council_session"))
"""

import functools
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional

from agriconnect.core.settings import settings

logger = logging.getLogger(__name__)

# ── Singleton client ──────────────────────────────────────────────────
_ls_client = None


def get_ls_client():
    """
    Retourne le client LangSmith (singleton, lazy-init).
    Retourne None si LangSmith n'est pas configuré.
    """
    global _ls_client
    if _ls_client is not None:
        return _ls_client

    if not settings.langsmith_enabled:
        logger.debug("LangSmith désactivé — pas de client initialisé.")
        return None

    try:
        from langsmith import Client
        _ls_client = Client()
        logger.info("✅ LangSmith client initialisé (projet: %s)", settings.LANGCHAIN_PROJECT)
        return _ls_client
    except Exception as e:
        logger.warning("⚠️  LangSmith client init échouée: %s", e)
        return None


def init_tracing() -> bool:
    """
    Initialise et valide la connexion LangSmith.
    Retourne True si tout est OK, False sinon.
    Appelé au démarrage de l'app (main.py).
    """
    if not settings.langsmith_enabled:
        logger.info("🔕 LangSmith tracing désactivé (LANGCHAIN_TRACING_V2 != true)")
        return False

    client = get_ls_client()
    if client is None:
        return False

    # Vérification de santé
    try:
        # Tente de lister les projets pour valider la clé API
        projects = list(client.list_projects(limit=1))
        logger.info(
            "✅ LangSmith connecté — endpoint: %s, projet: %s",
            settings.LANGCHAIN_ENDPOINT,
            settings.LANGCHAIN_PROJECT,
        )
        return True
    except Exception as e:
        logger.warning("⚠️  LangSmith health check échoué: %s", e)
        return False


# ── Décorateur de traçabilité ─────────────────────────────────────────

def trace_agent(
    name: Optional[str] = None,
    run_type: str = "chain",
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Décorateur pour tracer une fonction avec LangSmith.

    @trace_agent(name="sentinelle.analyze", tags=["weather"])
    def analyze_node(self, state):
        ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not settings.langsmith_enabled:
                return func(*args, **kwargs)

            try:
                from langsmith.run_helpers import traceable
                traced_fn = traceable(
                    name=name or func.__qualname__,
                    run_type=run_type,
                    tags=tags or [],
                    metadata=metadata or {},
                )(func)
                return traced_fn(*args, **kwargs)
            except ImportError:
                return func(*args, **kwargs)
            except Exception as e:
                logger.debug("Tracing error (non-blocking): %s", e)
                return func(*args, **kwargs)

        return wrapper
    return decorator


@contextmanager
def trace_span(name: str, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None):
    """
    Context manager pour créer un span de traçage autour d'un bloc.

    Usage:
        with trace_span("council_synthesis", tags=["parallel"]):
            results = do_parallel_work()
    """
    start = time.perf_counter()
    _metadata = {**(metadata or {}), "start_time": time.time()}

    try:
        yield _metadata
    finally:
        elapsed = time.perf_counter() - start
        _metadata["duration_seconds"] = round(elapsed, 3)
        logger.debug("Span '%s' completed in %.3fs", name, elapsed)


# ── Config injectable dans .invoke() ──────────────────────────────────

def get_tracing_config(
    run_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Retourne un dict de config injectable dans graph.invoke(state, config).

    Usage:
        result = self.graph.invoke(state, get_tracing_config(
            run_name="solo_sentinelle",
            tags=["weather", "bobo-dioulasso"],
            metadata={"user_level": "debutant"}
        ))
    """
    config: Dict[str, Any] = {"recursion_limit": 50}

    if not settings.langsmith_enabled:
        return config

    if run_name:
        config["run_name"] = run_name
    if tags:
        config["tags"] = tags
    if metadata:
        config["metadata"] = metadata

    return config


# ── Feedback programmatique ───────────────────────────────────────────

def create_feedback(
    run_id: str,
    key: str,
    score: float,
    comment: Optional[str] = None,
) -> bool:
    """
    Enregistre un feedback sur un run LangSmith.

    Args:
        run_id:  ID du run (récupéré via les callbacks).
        key:     Clé du feedback (ex: "relevance", "hallucination", "safety").
        score:   Score entre 0.0 et 1.0.
        comment: Commentaire optionnel.

    Returns:
        True si le feedback a été créé, False sinon.
    """
    client = get_ls_client()
    if client is None:
        return False

    try:
        client.create_feedback(
            run_id=run_id,
            key=key,
            score=score,
            comment=comment,
        )
        return True
    except Exception as e:
        logger.warning("Feedback creation failed: %s", e)
        return False


# ── Dataset helper ────────────────────────────────────────────────────

def get_or_create_dataset(name: str, description: str = "") -> Optional[Any]:
    """
    Récupère ou crée un dataset LangSmith.
    Utilisé par les scripts d'évaluation.
    """
    client = get_ls_client()
    if client is None:
        return None

    try:
        # Chercher le dataset existant
        datasets = list(client.list_datasets(dataset_name=name))
        if datasets:
            return datasets[0]
        # Créer sinon
        return client.create_dataset(dataset_name=name, description=description)
    except Exception as e:
        logger.warning("Dataset get/create failed for '%s': %s", name, e)
        return None
