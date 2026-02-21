"""
Agents — Agents IA spécialisés AgriConnect.

Chaque agent est un sous-graphe LangGraph autonome,
appelé par l'orchestrateur (MessageResponseFlow / ReportFlow).
"""

from .sentinelle import ClimateSentinel
from .formation import FormationCoach
from .market import MarketCoach
from .marketplace import MarketplaceAgent
from .soil import AgriSoilAgent
from .plant_doctor import PlantHealthDoctor

# Les agents ci-dessous dépendent de modules optionnels (services.google, etc.)
# Ils sont importés à la demande pour ne pas bloquer le startup.
# Usage:  from backend.agents.watcher import WatcherAgent
# Usage:  from backend.agents.voice import VoiceAgent

__all__ = [
    "ClimateSentinel",
    "FormationCoach",
    "MarketCoach",
    "MarketplaceAgent",
    "AgriSoilAgent",
    "PlantHealthDoctor",
]