import logging
from typing import TypedDict, Dict, Any, Optional, List
from datetime import datetime

# --- Importations LangChain & LangGraph ---
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOllama

from Tools.crop.base_crop import BurkinaCropTool

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

# ==============================================================================
# 2. L'AGENT CROP MANAGEMENT
# ==============================================================================
class BurkinaCropAgent:
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
            
            # Simulation d'un calcul de risque climatique (ex: 90 jours restants)
            risk = self.agro_tool.check_climate_risk(crop, 90)

            return {
                "technical_data": {
                    "sheet": sheet,
                    "inputs": inputs,
                    "risk": risk
                },
                "errors": []
            }
        except Exception as e:
            logger.error(f"Erreur Tool : {e}")
            return {"errors": [f"Erreur lors de l'accès aux fiches techniques : {str(e)}"]}

    # --- NOEUD 2 : GÉNÉRATION LLM (AVEC FALLBACK) ---
    def expert_response_node(self, state: AgentState):
        """Utilise Ollama pour rendre le conseil humain et chaleureux."""
        
        # S'il y a des erreurs, on les affiche proprement sans appeler le LLM
        if state.get("errors"):
            return {"final_response": f"❌ Désolé, j'ai rencontré des problèmes : {', '.join(state['errors'])}"}

        data = state["technical_data"]
        
        # Si Ollama est mort, on renvoie une réponse formatée "brute" (Scalabilité/Robustesse)
        if not self.llm:
            return {"final_response": self._get_fallback_ui(data)}

        # Prompt Expert
        system_prompt = (
            "Tu es un expert agronome burkinabè. Reformule les données techniques suivantes "
            "pour un producteur. Sois très clair sur les doses de sacs d'engrais. "
            "Encourage l'utilisation du Zaï ou des cordons pierreux si mentionné. "
            "Garde un ton fraternel et utilise des emojis 🌾."
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
    agent_app = BurkinaCropAgent().build()

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