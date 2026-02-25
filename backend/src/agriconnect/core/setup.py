import logging
from typing import Any, Dict, Optional

# Configuration et Core
from agriconnect.core.settings import settings
import agriconnect.core.database as _core_db
from agriconnect.services.db_handler import AgriDatabase
from agriconnect.rag.components import get_groq_sdk
from agriconnect.core.tracing import init_tracing

# Protocoles
from agriconnect.protocols.mcp import (
    MCPDatabaseServer, MCPRagServer, MCPWeatherServer, MCPContextServer
)
from agriconnect.protocols.a2a import A2ADiscovery
from agriconnect.protocols.ag_ui import WhatsAppRenderer, WebRenderer, SMSRenderer

# Mémoire 3 niveaux
from agriconnect.services.memory import (
    UserFarmProfile, ProfileExtractor, EpisodicMemory, ContextOptimizer
)

logger = logging.getLogger(__name__)

class AgriContext:
    """
    Système Nerveux Central AgriConnect.
    Gère le cycle de vie des ressources (DB, MCP, A2A, Mémoire).
    """

    def __init__(self, llm_client=None):
        self.llm = llm_client or get_groq_sdk()
        
        # Ressources Core
        self.db: Optional[AgriDatabase] = None
        self.session_factory: Any = None
        self.memory: Optional[ContextOptimizer] = None
        
        # Protocoles & Tracing
        self.mcp: Dict[str, Any] = {}
        self.a2a: Optional[A2ADiscovery] = None
        self.renderers: Dict[str, Any] = {}
        self.tracing_enabled: bool = False

    def bootstrap(self):
        """Lance l'initialisation dans l'ordre de dépendance strict."""
        self._init_tracing()
        self._init_db_and_memory()
        self._init_mcp_servers()
        self._init_a2a_discovery()
        self._init_renderers()
        logger.info("🚀 AgriContext bootstrap terminé.")
        return self

    def _init_tracing(self):
        """Initialise LangSmith avant tout le reste."""
        try:
            self.tracing_enabled = init_tracing()
        except Exception as e:
            logger.warning(f"🔭 Tracing non disponible: {e}")

    def _init_db_and_memory(self):
        """Initialise la persistence et la mémoire épisodique."""
        # Setup Database
        if _core_db._engine and _core_db._SessionLocal:
            self.db = AgriDatabase(engine=_core_db._engine, session_factory=_core_db._SessionLocal)
            self.session_factory = _core_db._SessionLocal
        elif settings.DATABASE_URL:
            self.db = AgriDatabase(db_url=settings.DATABASE_URL)
            logger.warning("⚠️ DB: Fallback URL utilisé.")

        # Setup Mémoire (Dépend de la session DB)
        if self.session_factory:
            try:
                profile = UserFarmProfile(self.session_factory)
                episodic = EpisodicMemory(self.session_factory, llm_client=self.llm)
                extractor = ProfileExtractor(self.llm, profile)
                self.memory = ContextOptimizer(profile, episodic, extractor)
                logger.info("🧠 Mémoire 3 niveaux activée")
            except Exception as e:
                logger.error(f"❌ Erreur Mémoire: {e}")

    def _init_mcp_servers(self):
        """Initialise les serveurs MCP (Système de Tools)."""
        try:
            self.mcp = {
                "db": MCPDatabaseServer(self.session_factory) if self.session_factory else None,
                "rag": MCPRagServer(),
                "weather": MCPWeatherServer(llm_client=self.llm),
                "context": None
            }
            # Initialisation du serveur de contexte (dépend de memory)
            if self.memory:
                self.mcp["context"] = MCPContextServer(context_optimizer=self.memory)
            elif self.session_factory:
                self.mcp["context"] = MCPContextServer(session_factory=self.session_factory, llm_client=self.llm)
            
            logger.info("🔌 MCP Servers configurés")
        except Exception as e:
            logger.error(f"❌ Erreur MCP: {e}")

    def _init_a2a_discovery(self):
        """Initialise le protocole Agent-to-Agent."""
        try:
            self.a2a = A2ADiscovery()
            self.a2a.register_internal_agents()
            logger.info("📡 A2A Discovery activé")
        except Exception as e:
            logger.error(f"❌ Erreur A2A: {e}")

    def _init_renderers(self):
        """Initialise les moteurs de rendu AG-UI."""
        self.renderers = {
            "whatsapp": WhatsAppRenderer(),
            "web": WebRenderer(),
            "sms": SMSRenderer(),
        }