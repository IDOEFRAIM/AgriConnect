"""
MCP Database Server — Accès standardisé aux profils agriculteurs.
=================================================================

AVANT (couplage direct) :
    session = SessionLocal()
    profile = session.query(UserFarmProfileModel).filter_by(user_id=uid).first()

APRÈS (découplé via MCP) :
    result = mcp_db.read_resource("agri://profile/{user_id}")

AVANTAGE : Si on migre de PostgreSQL vers CockroachDB ou Neon,
on ne touche QUE ce fichier. Les agents ne changent pas.
"""

import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

logger = logging.getLogger("MCP.Database")


class MCPDatabaseServer:
    """
    Serveur MCP exposant les tables PostgreSQL comme des Ressources.
    
    Ressources exposées :
      - agri://profile/{user_id}           → Profil ferme JSONB
      - agri://episodes/{user_id}          → Épisodes mémoire
      - agri://marketplace/products/{zone} → Produits marketplace par zone
    
    Outils exposés :
      - update_profile(user_id, patch)     → Mise à jour MERGE du profil
      - search_episodes(user_id, query)    → Recherche épisodique filtrée
    """

    def __init__(self, session_factory):
        self._session_factory = session_factory
        self._resources = {}
        self._tools = {}
        self._register_resources()
        self._register_tools()
        logger.info("🔌 MCP Database Server initialisé")

    # ═══════════════════════════════════════════════════════════
    # REGISTRATION
    # ═══════════════════════════════════════════════════════════

    def _register_resources(self):
        """Enregistre les ressources MCP disponibles."""
        self._resources = {
            "agri://profile/{user_id}": {
                "name": "Profil Ferme",
                "description": "Fiche ferme structurée JSONB de l'agriculteur",
                "mime_type": "application/json",
                "handler": self._read_profile,
            },
            "agri://episodes/{user_id}": {
                "name": "Mémoire Épisodique",
                "description": "Résumés d'interactions significatives",
                "mime_type": "application/json",
                "handler": self._read_episodes,
            },
            "agri://marketplace/products/{zone}": {
                "name": "Produits Marketplace",
                "description": "Produits disponibles dans une zone",
                "mime_type": "application/json",
                "handler": self._read_marketplace_products,
            },
        }

    def _register_tools(self):
        """Enregistre les outils MCP (lecture/écriture sécurisée)."""
        self._tools = {
            "update_profile": {
                "name": "update_profile",
                "description": "Mise à jour incrémentale (MERGE) du profil ferme",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "patch": {"type": "object", "description": "Champs à merger dans profile_data"},
                    },
                    "required": ["user_id", "patch"],
                },
                "handler": self._update_profile,
            },
            "search_episodes": {
                "name": "search_episodes",
                "description": "Recherche épisodique filtrée par culture, zone, catégorie",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "crop": {"type": "string"},
                        "zone": {"type": "string"},
                        "category": {"type": "string"},
                        "limit": {"type": "integer", "default": 5},
                    },
                    "required": ["user_id"],
                },
                "handler": self._search_episodes,
            },
        }

    # ═══════════════════════════════════════════════════════════
    # INTERFACE MCP STANDARDISÉE
    # ═══════════════════════════════════════════════════════════

    def list_resources(self) -> List[Dict[str, Any]]:
        """Liste toutes les ressources MCP disponibles."""
        return [
            {
                "uri": uri,
                "name": info["name"],
                "description": info["description"],
                "mimeType": info["mime_type"],
            }
            for uri, info in self._resources.items()
        ]

    def list_tools(self) -> List[Dict[str, Any]]:
        """Liste tous les outils MCP disponibles."""
        return [
            {
                "name": info["name"],
                "description": info["description"],
                "inputSchema": info["input_schema"],
            }
            for info in self._tools.values()
        ]

    def read_resource(self, uri: str, params: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Lit une ressource MCP par URI.
        
        Args:
            uri: URI de la ressource (ex: "agri://profile/user_123")
            params: Paramètres extraits de l'URI
            
        Returns:
            {"contents": [...], "status": "ok"} ou {"error": "..."}
        """
        # Résolution de l'URI vers le handler
        handler = self._resolve_resource(uri)
        if not handler:
            return {"error": f"Ressource inconnue: {uri}", "status": "not_found"}

        try:
            data = handler(params or self._extract_params(uri))
            return {
                "contents": [{"uri": uri, "mimeType": "application/json", "text": json.dumps(data, ensure_ascii=False)}],
                "status": "ok",
            }
        except Exception as e:
            logger.error("MCP read_resource error (%s): %s", uri, e)
            return {"error": str(e), "status": "error"}

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Appelle un outil MCP.
        
        Args:
            name: Nom de l'outil (ex: "update_profile")
            arguments: Arguments de l'outil
            
        Returns:
            {"content": [...], "status": "ok"} ou {"error": "..."}
        """
        tool = self._tools.get(name)
        if not tool:
            return {"error": f"Outil inconnu: {name}", "status": "not_found"}

        try:
            result = tool["handler"](arguments)
            return {
                "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
                "status": "ok",
            }
        except Exception as e:
            logger.error("MCP call_tool error (%s): %s", name, e)
            return {"error": str(e), "status": "error"}

    # ═══════════════════════════════════════════════════════════
    # HANDLERS INTERNES
    # ═══════════════════════════════════════════════════════════

    def _read_profile(self, params: Dict[str, str]) -> Dict[str, Any]:
        """Lit le profil ferme d'un utilisateur."""
        from backend.src.agriconnect.services.memory.user_profile import UserFarmProfileModel

        user_id = params.get("user_id")
        if not user_id:
            raise ValueError("user_id requis")

        session = self._session_factory()
        try:
            profile = session.query(UserFarmProfileModel).filter_by(user_id=user_id).first()
            if not profile:
                return {"user_id": user_id, "profile_data": {}, "exists": False}
            return {
                "user_id": user_id,
                "profile_data": profile.profile_data or {},
                "version": profile.version,
                "updated_at": profile.updated_at.isoformat() if profile.updated_at else None,
                "exists": True,
            }
        finally:
            session.close()

    def _read_episodes(self, params: Dict[str, str]) -> List[Dict[str, Any]]:
        """Lit les épisodes mémoire d'un utilisateur."""
        from backend.src.agriconnect.services.memory.episodic_memory import EpisodicMemoryModel

        user_id = params.get("user_id")
        if not user_id:
            raise ValueError("user_id requis")

        limit = int(params.get("limit", 10))

        session = self._session_factory()
        try:
            episodes = (
                session.query(EpisodicMemoryModel)
                .filter_by(user_id=user_id)
                .order_by(EpisodicMemoryModel.relevance_score.desc(), EpisodicMemoryModel.created_at.desc())
                .limit(limit)
                .all()
            )
            return [ep.to_dict() for ep in episodes]
        finally:
            session.close()

    def _read_marketplace_products(self, params: Dict[str, str]) -> List[Dict[str, Any]]:
        """Lit les produits marketplace d'une zone."""
        zone = params.get("zone")

        session = self._session_factory()
        try:
            # Requête SQL adaptée à la table marketplace existante
            from sqlalchemy import text
            result = session.execute(
                text("""
                    SELECT p.id, p.name, p.price, p.unit, p.quantity, p.zone_id, p.status
                    FROM products p 
                    WHERE (:zone IS NULL OR p.zone_id = :zone)
                    AND p.status = 'AVAILABLE'
                    ORDER BY p.created_at DESC
                    LIMIT 50
                """),
                {"zone": zone},
            )
            return [dict(row._mapping) for row in result]
        except Exception as e:
            logger.warning("Marketplace products read failed: %s", e)
            return []
        finally:
            session.close()

    def _update_profile(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Mise à jour incrémentale du profil (MERGE, pas REPLACE)."""
        from backend.src.agriconnect.services.memory.user_profile import UserFarmProfileModel

        user_id = arguments["user_id"]
        patch = arguments["patch"]

        session = self._session_factory()
        try:
            profile = session.query(UserFarmProfileModel).filter_by(user_id=user_id).first()
            if not profile:
                # Auto-création du profil
                profile = UserFarmProfileModel(user_id=user_id, profile_data=patch)
                session.add(profile)
            else:
                # Merge JSONB
                current = profile.profile_data or {}
                for key, value in patch.items():
                    if isinstance(value, dict) and isinstance(current.get(key), dict):
                        current[key].update(value)
                    elif isinstance(value, list) and isinstance(current.get(key), list):
                        current[key].extend(value)
                    else:
                        current[key] = value
                profile.profile_data = current

            session.commit()
            return {"user_id": user_id, "status": "updated", "profile_data": profile.profile_data}
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()

    def _search_episodes(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recherche épisodique filtrée."""
        from backend.src.agriconnect.services.memory.episodic_memory import EpisodicMemoryModel

        user_id = arguments["user_id"]
        crop = arguments.get("crop")
        zone = arguments.get("zone")
        category = arguments.get("category")
        limit = arguments.get("limit", 5)

        session = self._session_factory()
        try:
            query = session.query(EpisodicMemoryModel).filter_by(user_id=user_id)
            if crop:
                query = query.filter(EpisodicMemoryModel.crop.ilike(f"%{crop}%"))
            if zone:
                query = query.filter(EpisodicMemoryModel.zone.ilike(f"%{zone}%"))
            if category:
                query = query.filter_by(category=category)

            episodes = (
                query.order_by(
                    EpisodicMemoryModel.relevance_score.desc(),
                    EpisodicMemoryModel.created_at.desc(),
                )
                .limit(limit)
                .all()
            )
            return [ep.to_dict() for ep in episodes]
        finally:
            session.close()

    # ═══════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════

    def _resolve_resource(self, uri: str):
        """Résout un URI vers son handler."""
        for pattern, info in self._resources.items():
            # Matching simple (ex: agri://profile/user_123)
            if self._uri_matches(pattern, uri):
                return info["handler"]
        return None

    def _uri_matches(self, pattern: str, uri: str) -> bool:
        """Vérifie si un URI correspond à un pattern."""
        # Conversion du pattern MCP en regex simple
        import re
        regex = pattern.replace("{user_id}", r"[^/]+").replace("{zone}", r"[^/]+")
        return bool(re.match(regex, uri))

    def _extract_params(self, uri: str) -> Dict[str, str]:
        """Extrait les paramètres d'un URI MCP."""
        parts = uri.replace("agri://", "").split("/")
        params = {}
        if len(parts) >= 2:
            if parts[0] == "profile":
                params["user_id"] = parts[1]
            elif parts[0] == "episodes":
                params["user_id"] = parts[1]
            elif parts[0] == "marketplace" and len(parts) >= 3:
                params["zone"] = parts[2]
        return params
