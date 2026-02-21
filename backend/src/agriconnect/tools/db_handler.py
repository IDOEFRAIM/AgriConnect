"""
DB Handler (tools) — Thin wrapper over services.db_handler.AgriDatabase.

This module provides a simplified interface for tools that need DB access.
All heavy lifting delegates to the centralized AgriDatabase.
"""

import logging
from typing import Any, Dict, List, Optional

from backend.src.agriconnect.services.db_handler import AgriDatabase
from backend.src.agriconnect.core.settings import settings
import backend.src.agriconnect.core.database as _core_db

logger = logging.getLogger(__name__)

# ── Singleton lazy-init ──────────────────────────────────────

_db_instance: Optional[AgriDatabase] = None


def get_db() -> Optional[AgriDatabase]:
    """
    Retourne l'instance AgriDatabase singleton.
    Utilise le pool centralisé si disponible, sinon crée un engine propre.
    """
    global _db_instance
    if _db_instance is not None:
        return _db_instance

    try:
        if _core_db._engine and _core_db._SessionLocal:
            _db_instance = AgriDatabase(
                engine=_core_db._engine,
                session_factory=_core_db._SessionLocal,
            )
        elif settings.DATABASE_URL:
            _db_instance = AgriDatabase(db_url=settings.DATABASE_URL)
        else:
            logger.warning("DATABASE_URL not set — DB unavailable for tools")
            return None
    except Exception as exc:
        logger.error("Failed to init AgriDatabase for tools: %s", exc)
        return None

    return _db_instance


# ── Convenience functions for tools ──────────────────────────

def save_weather(zone_id: str, **kwargs) -> Optional[Dict[str, Any]]:
    """Proxy : enregistre des données météo."""
    db = get_db()
    if db is None:
        return None
    return db.save_weather(zone_id=zone_id, **kwargs)


def get_market_prices(product: str, zone_id: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Proxy : derniers prix marché."""
    db = get_db()
    if db is None:
        return []
    return db.get_latest_market_prices(product, zone_id, limit)


def create_alert(alert_type: str, severity: str, message: str, zone_id: str) -> Optional[Dict[str, Any]]:
    """Proxy : crée une alerte."""
    db = get_db()
    if db is None:
        return None
    return db.create_alert(alert_type, severity, message, zone_id)


def register_surplus(product_name: str, quantity_kg: float, **kwargs) -> Optional[Dict[str, Any]]:
    """Proxy : enregistre une offre de surplus."""
    db = get_db()
    if db is None:
        return None
    return db.save_surplus_offer(product_name=product_name, quantity_kg=quantity_kg, **kwargs)
