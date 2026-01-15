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
# 1. DÉFINITION DE L'ÉTAT
# ==============================================================================
class AgentState(TypedDict):
    user_query: str
    crop_name: str
    location_zone: str  # Nord, Centre, Sud
    surface_ha: float
    technical_data: Optional[Dict[str, Any]]
    final_response: str
    errors: List[str]
    blocking_alert: Optional[str]

# ==============================================================================
# 2. L'AGENT PRODUCTION EXPERT
# ==============================================================================
class ProductionExpert:
    def __init__(self, model_name="llama3:8b", ollama_host="http://localhost:11434", llm_client=None):
        self.agro_tool = BurkinaCropTool()
        self.model_name = model_name
        
        try:
            self.llm = llm_client if llm_client else ChatOllama(
                model=self.model_name, 
                base_url=ollama_host, 
                temperature=0.1
            )
        except Exception as e:
            logger.error(f"Ollama inaccessible : {e}")
            self.llm = None

    # --- NOEUD 1 : VALIDATION ET CALCULS ---
    def process_technical_node(self, state: AgentState) -> Dict[str, Any]:
        """Analyse les données agronomiques et vérifie la viabilité climatique."""
        logger.info("Analyse technique en cours...")
        errors = []
        
        crop = state.get("crop_name", "").lower()
        zone = state.get("location_zone", "Centre")
        surface = state.get("surface_ha", 1.0)
        
        if not crop:
            errors.append("Le nom de la culture est manquant.")
        
        if errors:
            return {"errors": errors}

        try:
            # Récupération des données métiers
            sheet = self.agro_tool.get_technical_sheet(crop, zone)
            inputs = self.agro_tool.calculate_inputs(crop, surface)
            
            # --- LOGIQUE D'ALERTE CLIMATIQUE ---
            seasonal_forecast_days = 90  # Simulation prévision
            crop_cycle_days = 110        # Simulation cycle culture
            
            blocking_alert = None
            if crop_cycle_days > seasonal_forecast_days:
                blocking_alert = (
                    f"⛔ **ALERTE : SAISON TROP COURTE**\n"
                    f"La variété '{crop}' nécessite {crop_cycle_days} jours, mais les pluies s'arrêteront "
                    f"dans environ {seasonal_forecast_days} jours. Semer maintenant est trop risqué.\n"
                    f"👉 **CONSEIL** : Utilisez une variété hâtive (cycle de 70-80 jours)."
                )
            
            return {
                "technical_data": {
                    "sheet": sheet,
                    "inputs": inputs,
                    "risk": f"Pluies prévues : {seasonal_forecast_days} jours."
                },
                "blocking_alert": blocking_alert,
                "errors": []
            }
        except Exception as e:
            logger.error(f"Erreur Tool : {e}")
            return {"errors": [f"Accès fiches techniques impossible : {str(e)}"]}

    # --- NOEUD 2 : GÉNÉRATION DE LA RÉPONSE ---
    def expert_response_node(self, state: AgentState) -> Dict[str, Any]:
        """Génère le conseil final via LLM ou interface de secours."""
        
        # 1. Cas de blocage (Climat ou Erreurs)
        if state.get("blocking_alert"):
            return {"final_response": state["blocking_alert"]}
        
        if state.get("errors"):
            return {"final_response": f"❌ Erreur : {', '.join(state['errors'])}"}

        data = state.get("technical_data")
        if not data:
            return {"final_response": "❌ Données techniques indisponibles."}

        # 2. Si LLM indisponible -> Fallback UI
        if not self.llm:
            return {"final_response": self._get_fallback_ui(data)}

        # 3. Prompting Expert
        system_prompt = (
            "Tu es l'Expert Production d'AgriConnect (Burkina Faso).\n"
            "Ton rôle est d'être direct, protecteur et technique.\n"
            "Structure : 🎯 Verdict | 🌾 Variété | 🎒 Intrants | ✋ Interdictions."
        )
        
        human_content = (
            f"Fiche : {data['sheet']}\n"
            f"Besoins : {data['inputs']}\n"
            f"Risque : {data['risk']}\n"
            f"Question : {state['user_query']}"
        )

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_content)
            ])
            return {"final_response": response.content}
        except Exception:
            return {"final_response": self._get_fallback_ui(data)}

    def _get_fallback_ui(self, data: Dict[str, Any]) -> str:
        """Interface de secours structurée (sans IA)."""
        sheet = data.get("sheet", "N/A")
        inputs = data.get("inputs", {})
        
        return (
            f"📋 **CONSEIL TECHNIQUE (Mode Secours)**\n\n"
            f"✅ **Fiche :** {sheet}\n\n"
            f"🚜 **BESOINS CALCULÉS :**\n"
            f"- NPK : {inputs.get('NPK_sacs_50kg', 0)} sacs\n"
            f"- Urée : {inputs.get('Uree_sacs_50kg', 0)} sacs\n"
            f"- Fumure : {inputs.get('Fumure_organique_tonnes', 0)} tonnes\n\n"
            f"🌦️ **RISQUE :** {data.get('risk')}"
        )

    def build(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("logic", self.process_technical_node)
        workflow.add_node("expert", self.expert_response_node)
        workflow.set_entry_point("logic")
        workflow.add_edge("logic", "expert")
        workflow.add_edge("expert", END)
        return workflow.compile()