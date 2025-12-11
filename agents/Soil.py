from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional, Callable
import logging

# NOUVELLES IMPORTATIONS pour le LLM
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_community.chat_models import ChatOllama 

# Import du toolkit Sol
from Tools.Soils.base_soil import SoilDoctorTool

logger = logging.getLogger("agent.soil_management")


# ======================================================================
# 1. DÉFINITION DE L'ÉTAT (CORRIGÉE)
# ======================================================================
class AgentState(TypedDict):
    zone_id: str
    user_query: str
    soil_config: Dict[str, Any]
    
    technical_advice_raw: Optional[str] # Sortie brute de l'outil
    final_response: str
    status: str


# ======================================================================
# 2. SERVICE DE GESTION DES SOLS
# ======================================================================
class SoilManagementService:
    def __init__(self, llm_client: Optional[Runnable] = None):
        self.name = "SoilManagementService"
        self.pedologist = SoilDoctorTool()
        self.llm_client = llm_client
        if not self.llm_client:
             logger.warning("Client ChatOllama non fourni. Les réponses seront brutes.")


    def analyze_node(self, state: AgentState) -> AgentState:
        """
        Nœud 1 : Exécute l'analyse pédologique et génère la sortie RAW.
        """
        query = state.get("user_query", "").lower()
        config = state.get("soil_config", {})

        # Normalisation robuste (inchangée)
        texture_input = (config.get("texture") or "").lower().strip()
        soil_condition = (config.get("history") or "normal").lower().strip()
        crop_name = (config.get("crop") or "maïs").lower().strip()
        organic_level = (config.get("organic_level") or "moyen").lower().strip()
        budget = (config.get("budget") or "moyen").lower().strip()
        
        if "sable" in query: texture_input = "sableux"
        elif "argile" in query: texture_input = "argileux"
        elif "gravier" in query or "cailloux" in query: texture_input = "gravillonnaire"
        elif "dur" in query and "pluie" in query: texture_input = "limoneux"
        if "pauvre" in query or "rien ne pousse" in query: soil_condition = "pauvre"


        logger.info(f"[{self.name}] Analyse Sol | Texture: {texture_input} | État: {soil_condition} | Culture: {crop_name}")

        response_parts = []
        has_advice = False

        # ============================================================
        # 1. TEXTURE & CONSERVATION (RAW)
        # ============================================================
        if texture_input:
            analysis = self.pedologist.analyze_texture(texture_input)
            if analysis["found"]:
                response_parts.append(f"**DIAGNOSTIC TEXTURE BRUT** : {analysis['profile']}")
                response_parts.append(f"- Rétention d'eau : {analysis['retention']}")
                response_parts.append(f"- Risques : {', '.join(analysis['risks'])}")
                response_parts.append(f"Action CES : {analysis['recommendation_ces']}")
                has_advice = True
        else:
             response_parts.append("MANQUE D'INFORMATION : Précisez la texture (sableux, argileux, etc.) pour une analyse complète.")

        # ============================================================
        # 2. FERTILITÉ (RAW)
        # ============================================================
        if "engrais" in query or "fumier" in query or "pauvre" in query or has_advice:
            org_advice = self.pedologist.recommend_organic_amendment(soil_condition)
            phos_advice = self.pedologist.recommend_phosphorus(
                 soil_type=texture_input, crop_name=crop_name, organic_level=organic_level, budget=budget
            )
            response_parts.append("\n**CONSEILS FERTILITÉ BRUTS :**")
            response_parts.append(f"Matière Organique : {org_advice}")
            response_parts.append(f"Phosphore : {phos_advice}")

        # ============================================================
        # 3. ACIDITÉ (pH) (RAW)
        # ============================================================
        if "acide" in query or "ph" in query:
            ph_val = config.get("ph")
            if ph_val:
                ph_analysis = self.pedologist.analyze_ph(float(ph_val))
                response_parts.append(f"\n**ANALYSE PH BRUTE ({ph_val})** : {ph_analysis['status']}")
                response_parts.append(f"Solution Recommandée : {ph_analysis['solution']}")
            else:
                 response_parts.append("MANQUE D'INFORMATION : PH non fourni. En cas de doute, la Dolomie est une solution de base.")

        # ============================================================
        # 4. CAS PAR DÉFAUT (RAW)
        # ============================================================
        if not response_parts:
             response_parts.append("**GUIDE DE L'AGENT SOL** : Veuillez poser une question sur la texture, la fertilité ou le pH de votre terre. (Ex: 'Mon sol est sableux et fatigué').")
             
        technical_advice_raw = "\n".join(response_parts)

        return {
            **state,
            "technical_advice_raw": technical_advice_raw,
            "status": "RAW_ANALYSIS_COMPLETE"
        }

    def llm_formatter_node(self, state: AgentState) -> AgentState:
        """
        Nœud 2 : Utilise ChatOllama pour transformer le conseil technique brut 
        en une réponse claire et professionnelle.
        """
        raw_advice = state.get("technical_advice_raw", "")
        user_query = state.get("user_query", "")
        
        if not self.llm_client:
            # Fallback
            return {**state, "final_response": raw_advice, "status": "FALLBACK_RAW_OUTPUT"}

        logger.info(f"[{self.name}] Début du formatage LLM avec ChatOllama.")

        # --- Définition du Prompt pour le LLM Léger ---
        system_prompt = (
            "Tu es le Docteur des Sols. Ton rôle est de fournir des conseils pédologiques "
            "avec un ton très professionnel, pédagogique et rassurant pour un agriculteur. "
            "Transforme le 'Conseil Technique Brut' en un plan d'action clair, structuré par : "
            "1. Diagnostic, 2. Améliorations, 3. Actions Immédiates. "
            "Utilise des emojis pour mettre en avant les points clés."
        )
        
        human_prompt = f"""
        **Question initiale de l'agriculteur :** "{user_query}"
        **Conseil Technique Brut :**
        ---
        {raw_advice}
        ---
        
        Rédige le conseil pédologique final.
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]

        # Appel du client ChatOllama
        try:
            response = self.llm_client.invoke(messages)
            final_response = response.content
        except Exception as e:
            logger.error(f"Erreur ChatOllama: {e}. Retour au conseil brut.")
            final_response = f"**Erreur de formatage.** Voici l'analyse technique brute :\n{raw_advice}"

        return {
            **state,
            "final_response": final_response,
            "status": "SUCCESS"
        }

    # ======================================================================
    # 3. WORKFLOW
    # ======================================================================
    def get_graph(self):
        """Construit et compile le Graph de l'Agent avec l'étape LLM."""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("manage_soil", self.analyze_node)       # Outil technique (génère la sortie RAW)
        workflow.add_node("format_llm_response", self.llm_formatter_node) # LLM léger (formatage)
        
        workflow.set_entry_point("manage_soil")
        workflow.add_edge("manage_soil", "format_llm_response")
        workflow.add_edge("format_llm_response", END)
        
        # Le flux est maintenant : Analyse (Outil) -> Formatage (LLM) -> Fin.
        
        return workflow.compile()


# ======================================================================
# 4. TEST AVEC if __main__ (CORRIGÉ)
# ======================================================================
if __name__ == "__main__":
    # --- 1. Initialisation du LLM Léger ---
    try:
        # NOTE: Remplacez 'mistral' par votre modèle Ollama léger si différent.
        ollama_client = ChatOllama(model="mistral", temperature=0.1)
        print("✅ ChatOllama initialisé avec Mistral.")
    except Exception:
        print("❌ ERREUR: Impossible de se connecter à ChatOllama. Le service utilisera la sortie brute.")
        ollama_client = None
    
    # --- 2. Initialisation du Service Agent ---
    service = SoilManagementService(llm_client=ollama_client)
    graph = service.get_graph()

    test_state: AgentState = {
        "zone_id": "zone-003",
        "user_query": "Mon champ est très sableux, que faire ? Il est fatigué et mon ph est de 6.8.",
        "soil_config": {
            "texture": "sableux",
            "ph": 6.8, 
            "history": "fatigué",
            "crop": "mil", 
            "organic_level": "faible",
            "budget": "faible"
        },
        "technical_advice_raw": None, # Clé correctement initialisée
        "final_response": "",
        "status": ""
    }

    print("\n--- Exécution du Graph ---")
    # Pour illustrer le concept de l'analyse pédologique
    
    
    # Exécution
    result = graph.invoke(test_state)
    
    print("\n=== Résultat du Docteur Sol (Formaté par LLM) ===")
    print(result["final_response"])
    print("\nStatus:", result["status"])