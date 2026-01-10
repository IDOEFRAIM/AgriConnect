import logging
from typing import TypedDict, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOllama

# --- IMPORTATION DES OUTILS MÉTIERS ---
from tools.health.base_health import HealthDoctorTool

logger = logging.getLogger("Agent.HealthSahel")

# ==============================================================================
# 1. DÉFINITION DE L'ÉTAT (STATE)
# ==============================================================================
class AgentState(TypedDict):
    user_query: str
    culture_config: Dict[str, Any]
    diagnosis_raw: Optional[Dict[str, Any]]
    technical_advice_text: str
    final_response: str
    status: str

# ==============================================================================
# 2. L'AGENT DE SANTÉ VÉGÉTALE (DOCTEUR DES PLANTES)
# ==============================================================================
class PlantHealthDoctor:

    def __init__(self,ollama_host: str = "http://localhost:11434",llm_client=None,OLLAMA_MODEL:Optional[str] = "mistral" ):
        self.doctor = HealthDoctorTool() 
        self.model_name = OLLAMA_MODEL
        self.llm_client = self._initialize_llm(OLLAMA_MODEL, llm_client, ollama_host)

    def _initialize_llm(self, model_name, llm_client, ollama_host: str):
        try:
            return llm_client if llm_client else ChatOllama(model=model_name, base_url=ollama_host, temperature=0.1) # adapte la temperature

        except Exception as e:
            logger.error(f"Échec connexion LLM: {e}")
            return None

    def _identify_symptoms_semantically(self, user_text: str) -> str:
        """
        Traduit les descriptions vagues en mots-clés techniques pour aider l'outil.
        Ex: 'belles fleurs violettes qui tuent mon mil' -> 'WONGO STRIGA'
        """
        if not self.llm_client: return user_text
        
        prompt = (
            "Tu es un phytopathologiste expert au Sahel. Analyse cette description : \n"
            f"'{user_text}'\n\n"
            "TÂCHE :\n"
            "Identifie les menaces potentielles même si l'utilisateur ne connait pas le nom.\n"
            "- Si ça ressemble au Striga (fleurs violettes, plante parasite, herbe sorcière), ajoute 'SUSPICION_WONGO STRIGA'.\n"
            "- Si ça parle de trous dans les feuilles ou de vers, ajoute 'CHENILLE LEGIONNAIRE'.\n"
            "- Si les feuilles jaunissent ou sèchent, ajoute 'SECHERESSE ou MALADIE FONGIQUE'.\n\n"
            "Réponds juste avec les mots-clés techniques détectés."
        )
        try:
            resp = self.llm_client.invoke([SystemMessage(content=prompt), HumanMessage(content="Analyse les symptômes.")])
            # On ajoute l'analyse à la requête originale pour garantir que l'outil attrape les mots-clés
            return f"{user_text} {resp.content.upper()}"
        except Exception as e:
            logger.error(f"Erreur semantic symptom detect: {e}")
            return user_text

    # --- NŒUD 1 : ANALYSE (LOGIQUE MÉTIER) ---
    def analyze_node(self, state: AgentState) -> AgentState:
        """Utilise le HealthDoctorTool pour identifier la menace."""
        config = state.get("culture_config", {})
        crop_name = config.get("crop_name", "Culture inconnue")
        query = state.get("user_query", "")
        
        # --- ENRICHISSEMENT SÉMANTIQUE ---
        # "Je vois des fleurs violettes" -> L'IA ajoute "WONGO STRIGA" -> L'outil déclenche l'alerte
        enhanced_query = self._identify_symptoms_semantically(query)

        # Appel à l'outil métier importé
        diag = self.doctor.diagnose_and_prescribe(
            crop=crop_name, 
            user_obs=enhanced_query
        )

        # Construction du rapport technique brut enrichi de supports visuels
        # Le nouvel outil retourne un dict avec 'diagnostique' si trouvé, sinon 'status': 'Inconnu'
        if "diagnostique" in diag:
            # On insère ici les tags visuels pour aider l'agriculteur à confirmer le diagnostic
            visual_aid = diag.get("diagramme_aide", "")
            prep_aid = self.doctor.get_biopesticide_tutorial("neem") # Exemple par défaut
            
            report = (
                f"PATHOLOGIE DÉTECTÉE : {diag['diagnostique']}\n"
                f"NIVEAU DE RISQUE : {diag.get('niveau_alerte')}\n"
                f"RECETTE BIO : {diag.get('prescription_bio')}\n"
                f"GUIDE DE PRÉPARATION : {prep_aid}\n"
                f"CONSEIL CHIMIQUE : {diag.get('conseil_chimique')}\n"
                f"MESURES PRÉVENTIVES : {diag.get('prevention')}"
            )
            status = "Trouvé"
        else:
            report = f"ERREUR : {diag.get('message', 'Symptômes non reconnus.')}"
            status = "Inconnu"

        return {
            **state,
            "diagnosis_raw": diag,
            "technical_advice_text": report,
            "status": status
        }

    # --- NŒUD 2 : FORMATAGE (LLM) ---
    def format_node(self, state: AgentState) -> AgentState:
        """Rend le diagnostic humain, bienveillant et structuré."""
        if self.llm_client is None or state["status"] != "Trouvé":
            return {**state, "final_response": state["technical_advice_text"]}

        system_prompt = (
            "Tu es le **Guérisseur des Plantes d'AgriConnect**. Ton but est de sauver la récolte ET la santé du paysan.\n"
            "Ton ennemi juré est 'Le Wongo' (Striga) et l'abus de chimie.\n\n"
            "**TON SERMENT :**\n"
            "'Je ne proposerai jamais un poison si un remède naturel existe.'\n\n"
            "**DIRECTIVES MÉDICALES :**\n"
            "1. **BIO D'ABORD :** Ta première ordonnance est TOUJOURS locale (Feuilles de Neem, Piment, Cendres, Ail). C'est gratut et sain.\n"
            "2. **CHIMIE EN DERNIER RECOURS :** Si l'attaque est critique, propose la chimie mais avec des **Avertissements de Sécurité EXTRÊMES** (Gants, masques).\n"
            "3. **DIAGNOSTIC WONGO :** Si c'est le Striga, dis 'Le problème est dans le sol, pas sur la feuille'. Ordonne l'arrachage immédiat avant la floraison.\n\n"
            "**STRUCTURE DE L'ORDONNANCE :**\n"
            "- 🔍 LE NOM DU MAL : Ce que la plante a attrapé.\n"
            "- 🌿 LE REMÈDE DE GRAND-MÈRE (Bio) : La recette exacte.\n"
            "- 🧪 LE REMÈDE CHOC (Chimique) : Seulement si nécessaire (+ Précautions).\n"
            "- 🛡️ LE VACCIN (Prévention) : Comment éviter que ça revienne."
        )

        try:
            msg = self.llm_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Rapport Technique :\n{state['technical_advice_text']}")
            ])
            return {**state, "final_response": msg.content, "status": "COMPLETED"}
        except Exception:
            return {**state, "final_response": state["technical_advice_text"], "status": "FALLBACK"}

    # --- CONSTRUCTION DU WORKFLOW ---
    def get_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("analyze", self.analyze_node)
        workflow.add_node("format", self.format_node)
        
        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "format")
        workflow.add_edge("format", END)
        
        return workflow.compile()