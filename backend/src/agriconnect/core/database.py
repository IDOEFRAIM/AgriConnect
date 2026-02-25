"""
Database — Connexion centralisée PostgreSQL (SQLAlchemy async-ready).

Usage:
    from backend.core.database import get_db, db_engine
"""

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from agriconnect.core.settings import settings

logger = logging.getLogger(__name__)

# ---------- Engine (créé une seule fois au démarrage) ----------

_engine = None
_SessionLocal = None


def init_db() -> None:
    """Initialise le moteur et la session factory. Appelé au startup FastAPI."""
    global _engine, _SessionLocal
    if not settings.DATABASE_URL:
        logger.warning("DATABASE_URL non configurée — base de données désactivée.")
        return
    _engine = create_engine(
        settings.DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
    )
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    logger.info("✅ Database engine initialisé.")


def close_db() -> None:
    """Ferme proprement le pool de connexions. Appelé au shutdown FastAPI."""
    global _engine
    if _engine:
        _engine.dispose()
        logger.info("🔒 Database engine fermé.")


def get_db() -> Generator[Session, None, None]:
    """
    Dependency FastAPI :
        @router.get("/items")
        def get_items(db: Session = Depends(get_db)):
            ...
    """
    if _SessionLocal is None:
        raise RuntimeError("Database non initialisée. Appelez init_db() d'abord.")
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_connection() -> bool:
    """Vérifie que la base est accessible."""
    if _engine is None:
        return False
    try:
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.warning("DB health check échoué: %s", e)
        return False
