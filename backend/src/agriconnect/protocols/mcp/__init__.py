"""
MCP Servers — Sous-package "Système Nerveux" AgriConnect 2.0
=============================================================

Expose les données et services internes comme des ressources/outils MCP standardisés.

Serveurs :
  - mcp_db      : Profils agriculteurs (Resources)       → agri://profile/{user_id}
  - mcp_rag     : Base de connaissances agronomiques (Tools) → search_agronomy_docs()
  - mcp_weather : Données météo et alertes (Tools)       → get_weather(), get_alerts()
  - mcp_context : Context Optimizer en MCP Host (Resources + Tools)

Avantage : Les agents ne font plus d'appels directs SQL/API.
Si on change de base, de fournisseur météo ou de vector DB, les agents ne changent PAS.
"""

from .mcp_db import MCPDatabaseServer
from .mcp_rag import MCPRagServer
from .mcp_weather import MCPWeatherServer
from .mcp_context import MCPContextServer

__all__ = [
    "MCPDatabaseServer",
    "MCPRagServer", 
    "MCPWeatherServer",
    "MCPContextServer",
]
