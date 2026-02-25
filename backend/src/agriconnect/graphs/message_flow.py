"""
Message Response Flow — Orchestrateur multi-experts AgriBot
============================================================

PHILOSOPHIE : "Salle de Crise" (pas "Standardiste")
-----------------------------------------------------
Tous les experts travaillent EN PARALLÈLE (fan-out / fan-in).
Le temps de réponse = max(expert_i), PAS sum(expert_i).

PATTERN :
  ANALYZE → route → [experts en parallèle] → SYNTHESIZE → TTS → PERSIST → END

  - Fan-out  : Le nœud PARALLEL_EXPERTS lance les experts activés simultanément.
  - Fan-in   : Le nœud SYNTHESIZE attend tous les résultats (via Reducer).
  - Reducer  : expert_responses: Annotated[List[ExpertResponse], operator.add]
               Chaque expert APPEND sa réponse ; pas de conflit d'écriture.
"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Literal

from langgraph.graph import END, StateGraph

from agriconnect.services.persistence import AgriPersister
from agriconnect.rag.components import get_groq_sdk
from agriconnect.core.tracing import get_tracing_config, init_tracing, trace_span
from agriconnect.core.setup import AgriContext
# Imports de vos structures et graphs.nodes
from .state import GlobalAgriState
from agriconnect.graphs.nodes.sentinelle import ClimateSentinel
from agriconnect.graphs.nodes.formation import FormationCoach
from agriconnect.graphs.nodes.market import MarketCoach
from agriconnect.graphs.nodes.marketplace import MarketplaceAgent

# Services
from agriconnect.services.voice import VoiceEngine
from agriconnect.services.db_handler import AgriDatabase
from agriconnect.core.settings import settings
import agriconnect.core.database as _core_db

# Mémoire 3 niveaux (Profil + Épisodique + Optimiseur)
from agriconnect.services.memory import (
    UserFarmProfile,
    ProfileExtractor,
    EpisodicMemory,
    ContextOptimizer,
)
from agriconnect.graphs.message_flow_helpers import ExpertInvoker
from agriconnect.graphs.message_flow_router import MessageRouter
from agriconnect.graphs.message_flow_parallel import ParallelExecutor

# ═══ Protocoles AgriConnect 2.0 (MCP + A2A + AG-UI) ═══
from agriconnect.protocols.mcp import MCPDatabaseServer, MCPRagServer, MCPWeatherServer, MCPContextServer
from agriconnect.protocols.a2a import A2ADiscovery
from agriconnect.protocols.ag_ui import AgriResponse, WhatsAppRenderer, WebRenderer, SMSRenderer

logger = logging.getLogger(__name__)


class MessageResponseFlow:
    """
    Orchestrateur ('Le Conseil') AgriConnect — version parallèle.

    Améliorations vs v1 séquentielle :
    1. Fan-out  : les experts s'exécutent en parallèle (ThreadPool).
    2. Fan-in   : un nœud SYNTHESIZE fusionne les réponses.
    3. Reducers : expert_responses est une liste additive (pas d'écrasement).
    4. Latence  : max(t_expert) au lieu de sum(t_expert).
    """


    def __init__(self, llm_client=None):
        self.llm = llm_client if llm_client is not None else get_groq_sdk()
        self.ctx = AgriContext(llm_client=self.llm).bootstrap()
        self._init_db_and_memory()
        self._init_protocols()
        self._init_experts()
        self._init_services()
        self._init_tracing()
        self.graph = self.build_graph()

    def _init_db_and_memory(self):
        # ── DB & Mémoire 3 niveaux ────────────────────────────────────

        if _core_db._engine and _core_db._SessionLocal:
            self.db = AgriDatabase(engine=_core_db._engine, session_factory=_core_db._SessionLocal)
            self.session_factory = _core_db._SessionLocal
        elif settings.DATABASE_URL:
            self.db = AgriDatabase(db_url=settings.DATABASE_URL)
            self.session_factory = None
            logger.warning("⚠️  DB: fallback engine propre (core/database.py non initialisé)")
        else:
            self.db = None
            self.session_factory = None

        self.memory = None
        if self.session_factory:
            try:
                _profile = UserFarmProfile(self.session_factory)
                _episodic = EpisodicMemory(self.session_factory, llm_client=self.llm)
                _extractor = ProfileExtractor(self.llm, _profile)
                self.memory = ContextOptimizer(_profile, _episodic, _extractor)
                logger.info("🧠 Mémoire 3 niveaux activée")
            except Exception as e:
                logger.warning("⚠️  Mémoire désactivée: %s", e)

    def _init_protocols(self):
        # ═══ Protocoles AgriConnect 2.0 (MCP + A2A + AG-UI) ═══
        # 1. MCP Servers (Système Nerveux)
        self.mcp_db = None
        self.mcp_rag = None
        self.mcp_weather = None
        self.mcp_context = None
        try:
            if self.session_factory:
                self.mcp_db = MCPDatabaseServer(self.session_factory)
            self.mcp_rag = MCPRagServer()
            self.mcp_weather = MCPWeatherServer(llm_client=self.llm)
            if self.memory:
                self.mcp_context = MCPContextServer(context_optimizer=self.memory)
            elif self.session_factory:
                self.mcp_context = MCPContextServer(session_factory=self.session_factory, llm_client=self.llm)
            logger.info("🔌 MCP Servers activés")
        except Exception as e:
            logger.warning("⚠️  MCP Servers error: %s", e)

        # 2. A2A Discovery
        self.a2a = None
        try:
            self.a2a = A2ADiscovery()
            self.a2a.register_internal_agents()
        except Exception as e:
            logger.warning("⚠️  A2A error: %s", e)

        # 3. AG-UI Renderers
        self.renderers = {
            "whatsapp": WhatsAppRenderer(),
            "web": WebRenderer(),
            "sms": SMSRenderer(),
        }

    def _init_experts(self):
        # ── Le Conseil des Experts (Injection de dépendances Protocolaires) ──
        self.sentinelle = ClimateSentinel(llm_client=self.llm)
        # Formation utilise MCP RAG + Context
        self.formation = FormationCoach(
            llm_client=self.llm,
            mcp_rag=self.mcp_rag,
            mcp_context=self.mcp_context
        )
        self.market = MarketCoach(llm_client=self.llm)
        # Marketplace injecte A2A et MCP DB
        self.marketplace = MarketplaceAgent(
            llm_client=self.llm,
            mcp_db=self.mcp_db,
            a2a=self.a2a
        )
        # Workflows compilés
        self.wf_sentinelle = self.sentinelle.build()
        self.wf_formation = self.formation.build()
        self.wf_market = self.market.build()
        self.wf_marketplace = self.marketplace.build()

        # Small helper to move invocation boilerplate out of this file
        self.invoker = ExpertInvoker({
            "sentinelle": self.wf_sentinelle,
            "formation": self.wf_formation,
            "market": self.wf_market,
            "marketplace": self.wf_marketplace,
        })
        # Encapsulated parallel executor to reduce complexity in this class
        self.parallel_executor = ParallelExecutor(self.invoker)

    def _init_services(self):
        # ── Services ──────────────────────────────────────────────────
        azure_key = settings.AZURE_SPEECH_KEY
        azure_fallback = settings.AZURE_SPEECH_KEY_2 or None
        azure_region = settings.AZURE_REGION
        self.voice = VoiceEngine(
            api_key=azure_key,
            region=azure_region,
            fallback_key=azure_fallback,
            storage_dir=settings.AUDIO_OUTPUT_DIR,
        ) if azure_key else None

    def _init_tracing(self):
        # ── LangSmith tracing (auto si .env configuré) ────────────
        self._tracing_ok = init_tracing()
        if self._tracing_ok:
            logger.info("🔭 LangSmith tracing actif pour l'orchestrateur")

    # ==================================================================
    # 1. ANALYSE DES BESOINS (The Chairman)
    # ==================================================================

    def analyze_needs(self, state: GlobalAgriState) -> Dict[str, Any]:
            query = state.get("requete_utilisateur", "")
            
            # 1. Récupération dynamique des fiches métiers via A2A
            # On demande au registre : "Donne-moi les capacités de tous les experts enregistrés"
            agent_cards = self.ctx.a2a.get_all_agent_cards() 
            
            # 2. Construction du catalogue pour le LLM
            expert_catalog = "\n".join([
                f"- {name}: {card.description}" 
                for name, card in agent_cards.items()
            ])

            system_prompt = (
                "Tu es le 'Cerveau Central' d'AgriConnect.\n"
                "Analyse la requête de l'agriculteur en fonction des experts DISPONIBLES ci-dessous :\n\n"
                f"{expert_catalog}\n\n" # <-- Injection dynamique !
                
                "DIRECTIVES :\n"
                "1. Détecte les ARNAQUES (demandes de fonds/codes) -> intent: REJECT\n"
                "2. CLASSIFICATION :\n"
                "   - 'CHAT' : Simple politesse.\n"
                "   - 'SOLO' : Un seul expert peut répondre.\n"
                "   - 'COUNCIL' : Plusieurs experts requis (ex: market + marketplace).\n"
                "   - 'REJECT' : Hors-sujet ou arnaque.\n\n"
                "Retourne JSON strict :\n"
                '{"intent": "REJECT"|"CHAT"|"SOLO"|"COUNCIL", '
                '"selected_experts": ["nom_expert_1", "nom_expert_2"], '
                '"reason": "si reject"}'
            )

            try:
                # Appel LLM (Llama-3.1-8b est parfait pour ce routage rapide)
                response = self.llm.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "system", "content": system_prompt},
                            {"role": "user", "content": query}],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                analysis = json.loads(response.choices[0].message.content)
                
                # On stocke les experts sélectionnés dans le state
                return {
                    "needs": analysis,
                    "execution_path": ["analyze"],
                }
            except Exception as e:
                logger.error(f"Routing Error: {e}")
                return {"needs": {"intent": "REJECT", "reason": "error"}, "execution_path": ["error"]}


    def execute_rejection(self, state: GlobalAgriState) -> Dict[str, Any]:
        """Prépare une réponse pédagogique en cas de message hors-sujet ou suspect."""
        reason = state.get("needs", {}).get("reason", "")
        
        # Message par défaut pour le Sahel
        msg = (
            "Désolé, je suis AgriBot et je ne peux répondre qu'aux questions sur l'agriculture, "
            "l'élevage et les prix du marché. Comment puis-je vous aider pour vos cultures ?"
        )
        
        # Si c'est une arnaque détectée
        if "arnaque" in reason.lower() or "argent" in reason.lower():
            msg = "Attention : Pour votre sécurité, ne partagez jamais de codes secrets ou de transferts d'argent via ce chat."

        return {
            "final_response": msg,
            "execution_path": ["rejection_node"],
        }
    
    # ==================================================================
    # 2. ROUTAGE DYNAMIQUE
    # ==================================================================
    def route_flow(self, state: GlobalAgriState):
        needs = state.get("needs", {})
        intent = needs.get("intent")
        selected = needs.get("selected_experts", [])

        if intent == "REJECT":
            return "REJECT"
        if intent == "CHAT":
            return "EXECUTE_CHAT"
        
        # Si on a plus d'un expert, on lance le Fan-out (PARALLEL)
        if intent == "COUNCIL" or len(selected) > 1:
            return "PARALLEL_EXPERTS"
        
        # Sinon, on route vers le nœud SOLO correspondant
        if len(selected) == 1:
            expert_name = selected[0]
            # Mapping dynamique vers tes nœuds SOLO
            return f"SOLO_{expert_name.upper()}"
        
        return "REJECT"
    # ==================================================================
    # 3. APPELS EXPERTS (Wrappers thread-safe)
    # ==================================================================

    # Expert invocation helpers moved to message_flow_helpers.ExpertInvoker

    # ==================================================================
    # 4. NŒUDS DU GRAPHE
    # ==================================================================
    # ── Nœud SOLO UNIVERSEL (Remplace tous tes execute_solo_...) ─────────────────

    def execute_solo_agent(self, state: GlobalAgriState) -> Dict[str, Any]:
        """
        Nœud dynamique qui appelle n'importe quel expert via le protocole A2A.
        """
        # 1. On récupère l'expert sélectionné par le routeur
        needs = state.get("needs", {})
        selected_experts = needs.get("selected_experts", [])
        
        if not selected_experts:
            return {"final_response": "Je n'ai pas trouvé d'expert pour vous répondre.", "execution_path": ["solo_error"]}

        agent_name = selected_experts[0] # En mode SOLO, on prend le premier
        query = state["requete_utilisateur"]

        # 2. Préparation du contexte minimal pour l'agent
        # On passe tout le state, l'A2A filtrera ce dont l'agent a besoin
        agent_context = {
            "zone_id": state.get("zone_id", "Bobo-Dioulasso"),
            "crop": state.get("crop", "Céréales"),
            "user_level": state.get("user_level", "debutant"),
            "user_phone": state.get("user_phone", "")
        }

        try:
            # 3. Appel Dynamique via le registre A2A (Inspiré de ton orchestrateur)
            # On ne fait plus self._call_sentinelle, mais un appel générique
            logger.info(f"📡 Appel A2A vers l'expert : {agent_name}")
            
            res = self.ctx.a2a.call_agent(
                agent_name=agent_name, 
                query=query, 
                context=agent_context
            )

            return {
                "final_response": res.get("response", ""),
                "execution_path": [f"solo_a2a_{agent_name}"],
            }

        except Exception as e:
            logger.error(f"❌ Échec de l'appel A2A pour {agent_name}: {e}")
            return {
                "final_response": "Désolé, l'expert est indisponible pour le moment.",
                "execution_path": ["solo_a2a_failed"]
            }

    # ── FAN-OUT : experts en parallèle (ThreadPool) ─────────────────
    def parallel_experts(self, state: GlobalAgriState) -> Dict[str, Any]:
        """
        Exécute en parallèle tous les experts sélectionnés par le routeur
        en utilisant le registre A2A.
        """
        needs = state.get("needs", {})
        selected_experts = needs.get("selected_experts", [])
        query = state["requete_utilisateur"]
        
        # Contexte partagé pour tous les experts
        context = {
            "zone_id": state.get("zone_id", "Bobo-Dioulasso"),
            "crop": state.get("crop", "Céréales"),
            "user_level": state.get("user_level", "debutant")
        }

        

        def _safe_call(name):
            try:
                # On utilise l'A2A du contexte pour appeler n'importe quel agent
                res = self.ctx.a2a.call_agent(name, query, context)
                return {
                    "expert": name,
                    "response": res.get("response", ""),
                    "has_alerts": res.get("has_alerts", False),
                    "is_lead": name == selected_experts[0] # Le 1er est le leader par défaut
                }
            except Exception as e:
                logger.error(f"Error calling {name}: {e}")
                return {"expert": name, "response": "", "has_alerts": False, "is_lead": False}

        # Exécution parallèle réelle
        with ThreadPoolExecutor(max_workers=len(selected_experts)) as executor:
            expert_responses = list(executor.map(_safe_call, selected_experts))

        # Nettoyage des réponses vides
        expert_responses = [r for r in expert_responses if r["response"]]

        return {
            "expert_responses": expert_responses,
            "execution_path": ["parallel_experts_a2a"],
        }
   # ==================================================================
    # 5. NETTOYAGE TTS
    # ==================================================================

    def clean_for_tts(self, text: str) -> str:
        """Supprime le Markdown et le HTML avant la synthèse vocale."""
        text = re.sub(r"[*_`#]", "", text)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # ==================================================================
    # 6. GÉNÉRATION AUDIO (TTS)
    # ==================================================================

    def generate_audio(self, state: GlobalAgriState) -> Dict[str, Any]:
        """Convertit final_response en fichier .wav via Azure TTS."""
        text = state.get("final_response", "")
        tts_text = self.clean_for_tts(text)

        if not tts_text or not self.voice:
            logger.warning("TTS skipped: no text or voice engine not configured")
            return {"audio_url": None}

        try:
            audio_path = self.voice.generate_audio(tts_text)
            logger.info("Audio generated: %s", audio_path)
            return {"audio_url": audio_path}
        except Exception as e:
            logger.warning("TTS error: %s", e)
            return {"audio_url": None}

    # ==================================================================
    # 7. PERSISTANCE (DB)
    # ==================================================================

    def persist(self, state: GlobalAgriState) -> Dict[str, Any]:
        """
        persistence de l'état final dans la base de données.
        """
        # Initialisation du persister avec les objets du contexte
        persister = AgriPersister(db=self.ctx.db, memory=self.ctx.memory)
        
        # Une seule ligne pour tout gérer
        persister.save_all(state)
        
        return {}
    # ==================================================================
    # BUILDER — Graphe LangGraph
    # ==================================================================
    def build_graph(self):
        """
        Graphe Dynamique A2A avec Fan-out / Fan-in.
        Structure simplifiée : les experts ne sont plus codés en dur.
        """
        workflow = StateGraph(GlobalAgriState)

        # 1. Nœuds de Traitement
        workflow.add_node("ANALYZE", self.analyze_needs)
        workflow.add_node("EXECUTE_CHAT", self.execute_chat)
        
        # Nœud Universel : invoque n'importe quel expert via A2A
        workflow.add_node("SOLO_AGENT", self.execute_solo_agent) 

        # Nœuds Parallèles
        workflow.add_node("PARALLEL_EXPERTS", self.parallel_experts)
        workflow.add_node("SYNTHESIZE", self.synthesize_results)

        # Nœuds de Sortie
        workflow.add_node("GENERATE_AUDIO", self.generate_audio)
        workflow.add_node("PERSIST", self.persist)
        workflow.add_node("REJECT", self.execute_rejection)

        # 2. Entrée
        workflow.set_entry_point("ANALYZE")

        # 3. Routage Conditionnel (Simplifié)
        # Le routeur renvoie maintenant des types d'exécution, pas des noms d'experts.
        workflow.add_conditional_edges(
            "ANALYZE",
            self.route_flow,
            {
                "EXECUTE_CHAT": "EXECUTE_CHAT",
                "SOLO_AGENT": "SOLO_AGENT", # Un seul nœud pour tous les experts
                "PARALLEL_EXPERTS": "PARALLEL_EXPERTS",
                "REJECT": "REJECT",
            },
        )

        # 4. Connexions des Flux
        workflow.add_edge("PARALLEL_EXPERTS", "SYNTHESIZE")

        # Tous les chemins de réponse convergent vers la suite du pipeline
        for node in ["EXECUTE_CHAT", "SOLO_AGENT", "SYNTHESIZE", "REJECT"]:
            workflow.add_edge(node, "GENERATE_AUDIO")

        workflow.add_edge("GENERATE_AUDIO", "PERSIST")
        workflow.add_edge("PERSIST", END)

        return workflow.compile()
# ======================================================================
# TESTS RAPIDES
# ======================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bot = MessageResponseFlow()

    print("\n--- TEST 1: CHAT ---")
    state1 = {
        "requete_utilisateur": "Bonjour AgriBot, comment ça va ?",
        "zone_id": "Bobo",
        "crop": "Maïs",
    }
    print(bot.run(state1)["final_response"])

    print("\n--- TEST 2: SOLO FORMATION ---")
    state2 = {
        "requete_utilisateur": "Explique-moi comment faire un compost simple.",
        "zone_id": "Bobo",
        "crop": "Maïs",
    }
    print(bot.run(state2)["final_response"])

    print("\n--- TEST 3: CONSEIL PARALLÈLE ---")
    state3 = {
        "requete_utilisateur": "Mes feuilles jaunissent et les prix du maïs tombent, je dois traiter ou vendre ?",
        "zone_id": "Sud-Ouest",
        "crop": "Maïs",
    }
    result = bot.run(state3)
    print(result["final_response"])
    print(f"Execution path: {result.get('execution_path')}")
    print(f"Expert responses: {len(result.get('expert_responses', []))}")
