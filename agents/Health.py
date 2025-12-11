from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional, Callable

# NOUVELLES IMPORTATIONS pour le LLM
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_community.chat_models import ChatOllama 

import logging
from datetime import datetime
from Tools.health.base_health import HealthDoctorTool

# Configuration du logger
logger = logging.getLogger("agent.plant_health")

# ==============================================================================
# 1. DÉFINITION DE L'ÉTAT (INCHANGÉ)
# ==============================================================================
class AgentState(TypedDict):
    zone_id: str
    user_query: str
    culture_config: Dict[str, Any]
    
    diagnosis: Optional[Dict]
    technical_advice_raw: str         # <-- NOUVEAU: Stocke la sortie brute
    final_response: str
    status: str

# ==============================================================================
# 2. SERVICE DE SANTÉ VÉGÉTALE (MISE À JOUR)
# ==============================================================================
class HealthManagementService:
    def __init__(self, llm_client: Optional[Runnable] = None): # Accepte un client LLM
        self.name = "HealthManagementService"
        self.doctor = HealthDoctorTool()
        self.llm_client = llm_client
        if not self.llm_client:
            logger.warning("Client LLM non fourni. Les réponses ne seront pas formatées.")

    def analyze_node(self, state: AgentState) -> AgentState:
        """
        Nœud 1 : Détermine le diagnostic et génère la sortie brute.
        (Mise à jour pour stocker la sortie dans 'technical_advice_raw')
        """
        query = state.get("user_query", "").lower()
        config = state.get("culture_config", {})
        crop_name = config.get("crop_name", "Inconnue")
        
        logger.info(f"[{self.name}] Diagnostic Santé pour {crop_name} | Symptômes: {query[:30]}...")

        response_parts = []
        diag_result: Optional[Dict] = None
        
        # --- 1. DIAGNOSTIC ---
        if any(w in query for w in ["maladie", "insecte", "trou", "feuille", "bête", "manger", "jaune", "rouge", "tache"]):
            diag_result = self.doctor.diagnose(crop_name, query)
            
            if diag_result["found"]:
                # Affichage des informations brutes pour le LLM
                response_parts.append(f"**DIAGNOSTIC BRUT :** {diag_result['disease']}")
                response_parts.append(f"Certitude : {diag_result['confidence']}")
                response_parts.append(f"Gravité : {diag_result['severity']}")
                response_parts.append("Ordonnance :")
                response_parts.append(f" - Solution Bio : {diag_result['advice']['bio']}")
                response_parts.append(f" - Solution Chimique : {diag_result['advice']['chimique']}")
            else:
                response_parts.append(f"**ALERTE :** Je n'ai pas pu identifier ces symptômes sur le {crop_name}.")
                response_parts.append(diag_result["message"])
                response_parts.append("Pour un meilleur diagnostic, décrivez plus précisément.")

        # --- 2. PRÉVENTION ---
        elif "prévenir" in query or "protéger" in query:
             prevention = self.doctor.get_prevention_plan(crop_name)
             response_parts.append(f"**PLAN DE PRÉVENTION :** {prevention}")

        # --- 3. CAS PAR DÉFAUT ---
        else:
            response_parts.append(f"**CLINIQUE DES PLANTES ({crop_name})** : Décrivez-moi ce que vous observez :")
            response_parts.append("- 'Il y a des chenilles dans le cornet'")
            response_parts.append("- 'Les feuilles jaunissent'")
            response_parts.append("- 'Une herbe parasite étouffe mon mil'")

        technical_advice_raw = "\n".join(response_parts)
        
        return {
            **state,
            "diagnosis": diag_result,
            "technical_advice_raw": technical_advice_raw, # Stockage de la sortie brute
            "status": "DIAGNOSED_RAW"
        }

    def llm_formatter_node(self, state: AgentState) -> AgentState:
        """
        Nœud 2 : Utilise le LLM léger (ChatOllama) pour transformer le diagnostic brut 
        en une réponse professionnelle et claire.
        """
        technical_advice_raw = state.get("technical_advice_raw", "")
        user_query = state.get("user_query", "")
        crop_name = state.get("culture_config", {}).get("crop_name", "votre culture")

        if not self.llm_client:
            # Si le client LLM manque, on retourne le texte brut
            return {
                **state,
                "final_response": technical_advice_raw,
                "status": "FALLBACK_RAW_OUTPUT"
            }

        logger.info(f"[{self.name}] Début du formatage LLM avec ChatOllama.")

        # --- Définition du Prompt pour le LLM Léger ---
        system_prompt = (
            "Tu es un docteur en santé des plantes. Ton ton doit être empathique, "
            "rassurant, mais très professionnel. Lis le 'Conseil Technique Brut' "
            "et réponds à l'agriculteur en français. S'il y a un diagnostic, annonce-le clairement, "
            "puis donne les solutions. Utilise des emojis pour rendre le texte agréable."
        )
        
        human_prompt = f"""
        **Culture :** {crop_name}
        **Question initiale :** "{user_query}"
        **Conseil Technique Brut :**
        ---
        {technical_advice_raw}
        ---
        
        Rédige le conseil final pour l'agriculteur.
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
            final_response = f"**Erreur de formatage.** Voici le conseil technique brut :\n{technical_advice_raw}"

        return {
            **state,
            "final_response": final_response,
            "status": "SUCCESS"
        }

    # ==============================================================================
    # 3. WORKFLOW (MISE À JOUR)
    # ==============================================================================
    def get_graph(self):
        """Construit et compile le Graph de l'Agent avec l'étape LLM."""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("manage_health", self.analyze_node)       # Outil technique (génère la sortie RAW)
        workflow.add_node("format_llm_response", self.llm_formatter_node) # LLM léger (formatage)
        
        workflow.set_entry_point("manage_health")
        workflow.add_edge("manage_health", "format_llm_response")
        workflow.add_edge("format_llm_response", END)
        
        # Le flux est maintenant : Diagnostic (Outil) -> Formatage (LLM) -> Fin.
        
        return workflow.compile()


# ==============================================================================
# 4. EXEMPLE D'UTILISATION (Remplacement du __main__)
# ==============================================================================

if __name__ == "__main__":
    # --- 1. Initialisation du LLM Léger ---
    # NOTE: Assurez-vous qu'Ollama est démarré et que le modèle 'mistral' est installé.
    try:
        ollama_client = ChatOllama(model="mistral", temperature=0.1)
        print("✅ ChatOllama initialisé avec Mistral.")
    except Exception:
        print("❌ ERREUR: Impossible de se connecter à ChatOllama. Le service utilisera la sortie brute.")
        ollama_client = None

    # --- 2. Initialisation du Service Agent ---
    service = HealthManagementService(llm_client=ollama_client)
    graph = service.get_graph()

    test_state: AgentState = {
        "zone_id": "zone-002",
        "user_query": "Les feuilles de mon maïs sont déchiquetées",
        "culture_config": {"crop_name": "Maïs"},
        "diagnosis": None,
        "technical_advice_raw": "",
        "final_response": "",
        "status": ""
    }

    print("\n")
    print("\n--- Exécution du Graph ---")
    
    # Exécution
    result = graph.invoke(test_state)
    
    # Affichage
    print("\n=== Résultat du Phyto-Docteur Formaté ===")
    print(result["final_response"])
    print("\nDiagnosis structuré:", result.get("diagnosis", "N/A"))
    print("Status:", result["status"])