import logging
from typing import TypedDict, Dict, Any, Optional, List
from datetime import datetime

# --- Importations LangChain & LangGraph ---
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOllama

from tools.crop.base_crop import BurkinaCropTool

# --- Configuration du Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CropAgent.Burkina")

# ==============================================================================
# 1. DÉFINITION DE L'ÉTAT (ROBUSTE)
# ==============================================================================
class AgentState(TypedDict):
    user_query: str
    crop_name: str
    location_zone: str  # Nord, Centre, Sud
    surface_ha: float
    technical_data: Optional[Dict[str, Any]]
    final_response: str
    errors: List[str]
    blocking_alert: Optional[str]  # Ajout pour gérer l'alerte "Saison courte"

# ==============================================================================
# 2. L'AGENT PRODUCTION EXPERT (Vision Terrain)
# ==============================================================================
class ProductionExpert:
    def __init__(self, model_name="llama3:8b", ollama_host="http://localhost:11434",llm_client=None):
        # Initialisation de ton Tool
        self.agro_tool = BurkinaCropTool()
        self.model_name = model_name
        
        # Initialisation sécurisée d'Ollama
        try:
            self.llm = llm_client if llm_client else ChatOllama(model=self.model_name, base_url=ollama_host, temperature=0.1) # adapte la temperature
        except Exception as e:
            logger.error(f"Ollama inaccessible : {e}")
            self.llm = None

    # --- NOEUD 1 : VALIDATION ET CALCULS ---
    def process_technical_node(self, state: AgentState):
        """Valide les entrées et extrait les données de BurkinaCropTool."""
        logger.info("Analyse technique en cours...")
        errors = []
        
        crop = state.get("crop_name", "").lower()
        zone = state.get("location_zone", "Centre")
        surface = state.get("surface_ha", 1.0)
        
        # Validation minimale
        if not crop:
            errors.append("Le nom de la culture est manquant.")
        
        if errors:
            return {"errors": errors}

        try:
            # Récupération des données du Tool
            sheet = self.agro_tool.get_technical_sheet(crop, zone)
            inputs = self.agro_tool.calculate_inputs(crop, surface)
            
            # --- FEATURE REQUISE : AVEUGLEMENT CLIMATIQUE ---
            # Simulation : On récupère la prévision saisonnière (mockée ici)
            # Dans le futur : self.meteo_tool.get_seasonal_forecast(zone)
            seasonal_forecast_days = 90  # Ex: La pluie s'arrêtera dans 90 jours
            crop_cycle_days = 110 # Ex: Ce maïs met 110 jours
            
            blocking_alert = None
            risk = f"Saison prévue : {seasonal_forecast_days} jours."

            if crop_cycle_days > seasonal_forecast_days:
                blocking_alert = (
                    f"⛔ **INTERDICTION DE SEMER** : La variété '{crop}' demande {crop_cycle_days} jours, "
                    f"mais la pluie s'arrêtera dans {seasonal_forecast_days} jours.\n"
                    f"👉 **ACTION OBLIGATOIRE** : Privilégier une variété à cycle court (ex: Sorgho hâtif)."
                )
            
            return {
                "technical_data": {
                    "sheet": sheet,
                    "inputs": inputs,
                    "risk": risk
                },
                "blocking_alert": blocking_alert,
                "errors": []
            }
        except Exception as e:
            logger.error(f"Erreur Tool : {e}")
            return {"errors": [f"Erreur lors de l'accès aux fiches techniques : {str(e)}"]}

    # --- NOEUD 2 : GÉNÉRATION LLM (AVEC FALLBACK) ---
    def expert_response_node(self, state: AgentState):
        """Utilise Ollama pour rendre le conseil humain et chaleureux."""
        
        # GESTION PRIORITAIRE DE L'ALERTE BLOQUANTE
        if state.get("blocking_alert"):
            return {"final_response": state["blocking_alert"]}

        # S'il y a des erreurs, on les affiche proprement sans appeler le LLM
        if state.get("errors"):
            return {"final_response": f"❌ Désolé, j'ai rencontré des problèmes : {', '.join(state['errors'])}"}

        data = state["technical_data"]
        
        # Si Ollama est mort, on renvoie une réponse formatée "brute" (Scalabilité/Robustesse)
        if not self.llm:
            return {"final_response": self._get_fallback_ui(data)}

        # Prompt Expert
        system_prompt = (
            "Tu es **l'Expert Production d'AgriConnect**, un vieux sage de l'agriculture burkinabè qui a vu toutes les saisons.\n"
            "TA MISSION : Empêcher le producteur de faire une erreur fatale (semer trop tard, mauvaise variété).\n\n"
            "**PHILOSOPHIE :**\n"
            "- 'Mieux vaut ne pas semer que de tout perdre.'\n"
            "- La technique est un moyen, pas une fin.\n\n"
            "**RÈGLES D'ACTION :**\n"
            "1. **CALENDRIER D'ABORD :** Si la saison est courte, INTERDIS les variétés lentes. Impose les **semences améliorées** (INERA).\n"
            "2. **ÉCONOMIE D'INTANTS :** Sois précis sur les doses (sacs et charrettes). Le gaspillage est un péché.\n"
            "3. **L'EAU C'EST LA VIE :** Si la pluie est rare, ordonne les techniques de **Zaï** ou **Demi-Lunes** immédiatement.\n"
            "4. **PEDAGOGIE :** Explique comme à ton petit frère. Utilise des images simples.\n\n"
            "STRUCTURE :\n"
            "- 🎯 VERDICT : Semer ou Attendre ?\n"
            "- 🌾 LA VARIÉTÉ : Laquelle choisir et pourquoi ?\n"
            "- 🎒 LE SAC À DOS (Besoins) : NPK, Urée, Fumure (Quantités exactes).\n"
            "- ✋ INTERDICTIONS : Ce qu'il ne faut surtout pas faire."
        )
        
        try:
            # Appel LLM
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Données techniques : {str(data)}")
            ]
            response = self.llm.invoke(messages)
            return {"final_response": response.content}
        except Exception as e:
            logger.error(f"Erreur LLM : {e}")
            return {"final_response": self._get_fallback_ui(data)}

    def _get_fallback_ui(self, data: Dict[str, Any]) -> str:
        """Génère une réponse textuelle structurée sans LLM."""
        sheet = data.get("sheet", "Non disponible")
        inputs = data.get("inputs", {})
        risk = data.get("risk", "Non évalué")
        
        return (
            f"📋 **FICHE TECHNIQUE (Mode Hors-Ligne)**\n\n"
            f"{sheet}\n\n"
            f"🚜 **BESOINS INTRANTS (pour {inputs.get('surface_ha', 0)} ha)**\n"
            f"- NPK : {inputs.get('NPK_sacs_50kg', 0)} sacs\n"
            f"- Urée : {inputs.get('Uree_sacs_50kg', 0)} sacs\n"
            f"- Fumure : {inputs.get('Fumure_organique_tonnes', 0)} tonnes\n\n"
            f"🌦️ **RISQUE CLIMATIQUE** : {risk}"
        )

        
        human_prompt = f"""
        FICHE TECHNIQUE : {data['sheet']}
        BESOINS CALCULÉS : {data['inputs']}
        ALERTE RISQUE : {data['risk']}
        QUESTION INITIALE : {state['user_query']}
        """

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ])
            return {"final_response": response.content}
        except Exception as e:
            logger.error(f"Erreur LLM : {e}")
            return {"final_response": self._get_fallback_ui(data)}

    def _get_fallback_ui(self, data: Dict) -> str:
        """Interface de secours si le LLM crash."""
        inputs = data['inputs']
        return (
            f"📢 **CONSEIL TECHNIQUE (Mode Secours)**\n\n"
            f"{data['sheet']}\n\n"
            f"📦 **BESOINS POUR VOTRE SURFACE :**\n"
            f"- NPK : {inputs.get('NPK_sacs_50kg')} sacs\n"
            f"- Urée : {inputs.get('Uree_sacs_50kg')} sacs\n"
            f"- Fumure : {inputs.get('Fumure_organique_tonnes')} tonnes\n\n"
            f"{data['risk']}"
        )

    # --- CONSTRUCTION DU WORKFLOW ---
    def build(self):
        workflow = StateGraph(AgentState)
        
        workflow.add_node("logic", self.process_technical_node)
        workflow.add_node("expert", self.expert_response_node)
        
        workflow.set_entry_point("logic")
        workflow.add_edge("logic", "expert")
        workflow.add_edge("expert", END)
        
        return workflow.compile()

# ==============================================================================
# 3. EXEMPLE D'UTILISATION
# ==============================================================================
if __name__ == "__main__":
    # 1. Initialisation de l'agent
    agent_app = ProductionExpert().build()

    # 2. Données d'entrée (Scalable : on peut ajouter des champs facilement)
    inputs = {
        "user_query": "Combien de sacs d'engrais faut-il pour 2 hectares de sorgho au Nord ?",
        "crop_name": "sorgho",
        "location_zone": "Nord",
        "surface_ha": 2.0,
        "errors": []
    }

    # 3. Execution
    print("\n--- TRAITEMENT EN COURS ---\n")
    result = agent_app.invoke(inputs)
    
    print("\n--- RÉPONSE FINALE ---")
    print(result["final_response"])