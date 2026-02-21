"""
Orchestrator — Flux LangGraph AgriConnect.

- MessageResponseFlow : Flux principal (message utilisateur → réponse)
- GlobalAgriState     : État partagé entre les nœuds du graphe
"""

# Imports lazy pour éviter de charger LangGraph au startup
# Usage: from backend.orchestrator.message_flow import MessageResponseFlow

__all__ = [
    "MessageResponseFlow",
    "GlobalAgriState",
]