"""
Services — couche métier AgriConnect.

Structure :
  - db_handler.py     : Accès base de données (SQLAlchemy ORM)
  - models.py         : Modèles SQLAlchemy (source unique de vérité)
  - voice_engine.py   : Azure TTS / STT (service layer)
  - voice.py          : Re-export de VoiceEngine (compat)
  - llm_clients.py    : Clients LLM (Groq / ChatGroq)
  - memory/           : Mémoire 3 niveaux (profil, épisodique, optimiseur)
  - broadcaster.py    : Diffusion d'alertes multi-canal
  - external_apis/    : Intégrations APIs externes
  - data_collection/  : Collecteurs de données
  - scraper/          : Système de scraping
  - scheduling/       : Orchestration temporelle
  - utils/            : Utilitaires transverses
"""

from .db_handler import AgriDatabase
from .llm_clients import get_groq_client
from .voice_engine import VoiceEngine

__all__ = [
    "AgriDatabase",
    "get_groq_client",
    "VoiceEngine",
]
