"""
AgriConnect Backend — Point d'entrée FastAPI.

Responsabilités :
  1. Configurer le logging
  2. Créer l'app FastAPI avec métadonnées
  3. Ajouter middlewares (CORS, error handler)
  4. Brancher le lifecycle (startup → init DB ; shutdown → close DB)
  5. Inclure les routes
"""

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.src.agriconnect.api.routes import router

from backend.src.agriconnect.core.settings import settings
from backend.src.agriconnect.core.logger import setup_logging
from backend.src.agriconnect.core.database import init_db, close_db


# ── Lifecycle ────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / Shutdown hooks."""
    setup_logging()
    logger = logging.getLogger("AgriConnect")
    logger.info("🚀 Starting AgriConnect v%s …", settings.APP_VERSION)

    # Init DB pool
    init_db()

    yield  # ← app is running

    # Shutdown
    close_db()
    logger.info("🛑 AgriConnect stopped.")


# ── App Factory ──────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    description="Agricultural Multi-Agent Assistant — Event-Driven Architecture",
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# ── Middlewares ───────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all pour les erreurs non gérées → JSON propre."""
    logging.getLogger("AgriConnect").error(
        "Unhandled error on %s %s: %s",
        request.method, request.url.path, exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."},
    )


# ── Routes ───────────────────────────────────────────────────

app.include_router(router)


# ── Standalone runner ────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
