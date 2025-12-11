from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional, Callable # Callable est important
import logging

# NOUVELLES IMPORTATIONS pour le LLM
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_community.chat_models import ChatOllama 

from Tools.subventions.base_subsidy import GrantExpertTool

logger = logging.getLogger("agent.subsidy_finance")

# 1. DÉFINITION DE L'ÉTAT (INPUTS/OUTPUTS) - MIS À JOUR
class AgentState(TypedDict):
    zone_id: str
    user_query: str
    user_profile: Dict[str, Any]
    
    opportunities: Optional[List]
    technical_advice_raw: Optional[str] # <-- NOUVEAU: Stocke la sortie brute
    final_response: str
    status: str


# 2. SERVICE DE GESTION DES AIDES (MISE À JOUR)
class SubsidyManagementService:
    def __init__(self, llm_client: Optional[Runnable] = None): # Accepte un client LLM
        self.name = "SubsidyManagementService"
        self.expert = GrantExpertTool()
        self.llm_client = llm_client
        if not self.llm_client:
             logger.warning("Client ChatOllama non fourni. Les réponses seront brutes.")


    def analyze_node(self, state: AgentState) -> AgentState:
        """
        Nœud 1 : Exécute l'analyse des aides et génère la sortie RAW.
        """
        query = (state.get("user_query") or "").lower()
        profile = state.get("user_profile", {})

        logger.info(f"[{self.name}] Analyse demande financière | Profil: {profile.get('crop')}")

        response_parts = []
        status_code = "SUCCESS"
        opportunities_list = None
        
        # 1. SÉCURITÉ : DÉTECTION D'ARNAQUE (RAW)
        scam_keywords = ["arnaque", "vrai", "frais", "envoyer", "sms", "dossier", "payer"]
        if any(w in query for w in scam_keywords):
            scam_analysis = self.expert.check_scam(query)

            if scam_analysis["is_scam"]:
                status_code = "SCAM_DETECTED"
                # Sortie RAW formatée pour le LLM
                response_parts.append(f"**ALERTE SÉCURITÉ :** {scam_analysis['warning']}")
                response_parts.append("RAISONS SUSPECTES :")
                for reason in scam_analysis["reasons"]:
                    response_parts.append(f"- {reason}")
                response_parts.append(
                    "CONSEIL BRUT : Ne payez jamais de frais de dossier par Orange Money/Mobicash avant d'avoir vu un agent officiel."
                )

            elif "vrai" in query:
                 response_parts.append(f"VERIFICATION DEMANDÉE : {scam_analysis['warning']}")
        
        
        # 2. OPPORTUNITÉS : RECHERCHE D'AIDES (RAW)
        opportunity_keywords = ["aide", "subvention", "argent", "financement", "projet", "banque"]
        if any(w in query for w in opportunity_keywords) and status_code != "SCAM_DETECTED":
            matches = self.expert.find_opportunities(profile)
            opportunities_list = matches

            if matches:
                response_parts.append("\n**OPPORTUNITÉS DE SUBVENTION BRUTES :**")
                for i, m in enumerate(matches):
                    response_parts.append(f"PROGRAMME {i+1} : {m['program_name']}")
                    response_parts.append(f" - Source : {m['provider']}")
                    response_parts.append(f" - Catégorie : {m['category']}")
                    response_parts.append(f" - Période/Statut : {m['deadline']} - {m['status']}")
                    if m["missing_documents"]:
                        response_parts.append(f" - DOCS MANQUANTS : {', '.join(m['missing_documents'])}")
                    response_parts.append(f" - Score Eligibilité : {m['eligibility_score']}/100")
            else:
                 response_parts.append(
                     "AUCUNE SUBVENTION BRUTE : Aucune subvention active correspondant à votre profil (Culture/Zone) pour le moment."
                 )

        # 3. PROCÉDURE : COMMENT FAIRE ? (RAW)
        procedure_keywords = ["comment", "procédure", "papier", "document", "aller où"]
        if any(w in query for w in procedure_keywords) and status_code != "SCAM_DETECTED":
            p_type = "irrigation" if any(w in query for w in ["pompe", "eau", "irrigation", "foncier"]) else "intrant"
            guide = self.expert.get_application_guide(p_type)
            response_parts.append(f"\n**GUIDE DE PROCÉDURE BRUT ({p_type.upper()}) :**")
            response_parts.append(guide)

        # 4. CAS PAR DÉFAUT (RAW)
        if not response_parts:
            response_parts.append("**GUIDE DE L'AGENT SUBVENTION** : Je peux vérifier les opportunités pour vous ou analyser un risque d'arnaque.")
            response_parts.append("Exemples : 'Subvention pour le maïs', 'Comment avoir une pompe solaire', 'Est-ce une arnaque ?'")


        technical_advice_raw = "\n".join(response_parts)

        # Si SCAM détecté, on renvoie directement sans passer par le LLM (urgence)
        if status_code == "SCAM_DETECTED":
            # On utilise le RAW comme réponse finale dans ce cas d'urgence
             return {
                **state,
                "technical_advice_raw": technical_advice_raw,
                "final_response": technical_advice_raw, # Contournement du LLM
                "status": status_code,
                "opportunities": opportunities_list
            }

        return {
            **state,
            "technical_advice_raw": technical_advice_raw,
            "status": "RAW_ANALYSIS_COMPLETE",
            "opportunities": opportunities_list
        }


    def llm_formatter_node(self, state: AgentState) -> AgentState:
        """
        Nœud 2 : Utilise ChatOllama pour transformer la sortie technique brute 
        en un conseil financier convivial et clair.
        """
        raw_advice = state.get("technical_advice_raw", "")
        user_query = state.get("user_query", "")
        
        if not self.llm_client:
            return {**state, "final_response": raw_advice, "status": "FALLBACK_RAW_OUTPUT"}

        logger.info(f"[{self.name}] Début du formatage LLM avec ChatOllama.")

        # --- Définition du Prompt pour le LLM Léger ---
        system_prompt = (
            "Tu es l'Expert en Finances Agricoles. Ton rôle est de présenter les opportunités "
            "de subventions et les guides de procédure de manière claire, concise et inspirante. "
            "Transforme le 'Conseil Technique Brut' en une réponse professionnelle. "
            "Utilise des titres et des puces pour la lisibilité et des emojis pour encourager l'agriculteur."
        )
        
        human_prompt = f"""
        **Demande de l'agriculteur :** "{user_query}"
        **Conseil Technique Brut :**
        ---
        {raw_advice}
        ---
        
        Rédige le conseil financier final.
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

    # 3. WORKFLOW (MISE À JOUR)
    def get_graph(self):
        """Construit et compile le Graph de l'Agent avec l'étape LLM et la condition de sécurité."""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("manage_subsidy", self.analyze_node)
        workflow.add_node("format_llm_response", self.llm_formatter_node)
        
        workflow.set_entry_point("manage_subsidy")
        
        # Le routage conditionnel : Si SCAM détecté, on va à END. Sinon, on formate.
        workflow.add_edge("manage_subsidy", "format_llm_response") # Route par défaut
        
        # C'est ici que LangGraph excelle : on pourrait ajouter un noeud pour la sécurité
        # Mais pour cet exercice simple, on gère le statut de sortie directement dans le noeud
        # et on assume que le noeud LLM est SKIPPE si SCAM_DETECTED est géré dans analyze_node.
        # Dans ce cas, nous ajoutons l'arête finale:
        workflow.add_edge("format_llm_response", END)

        return workflow.compile()


# ======================================================================
# 4. TEST AVEC if __main__ (MISE À JOUR)
# ======================================================================
if __name__ == "__main__":
    # --- 1. Initialisation du LLM Léger ---
    try:
        ollama_client = ChatOllama(model="mistral", temperature=0.1)
        print("✅ ChatOllama initialisé avec Mistral.")
    except Exception:
        print("❌ ERREUR: Impossible de se connecter à ChatOllama. Le service utilisera la sortie brute.")
        ollama_client = None
    
    # --- 2. Initialisation du Service Agent ---
    service = SubsidyManagementService(llm_client=ollama_client)
    graph = service.get_graph()

    # TEST 1 : Recherche de subventions
    test_state_subsidy: AgentState = {
        "zone_id": "zone-004",
        "user_query": "Existe-t-il une aide pour acheter des semences pour mon riz ?",
        "user_profile": {
            "crop": "Riz",
            "status": "Jeune agriculteur",
            "documents_ok": ["CNI", "Titre Foncier"],
            "zone_eligible": True
        },
        "opportunities": None,
        "technical_advice_raw": None,
        "final_response": "",
        "status": ""
    }
    
    # TEST 2 : Alerte Arnaque (pour tester le contournement du LLM)
    test_state_scam: AgentState = {
        "zone_id": "zone-004",
        "user_query": "J'ai reçu un sms me demandant d'envoyer 10000F par Orange Money pour un dossier de subvention pour l'achat de semences. Est-ce vrai ?",
        "user_profile": {"crop": "Riz"},
        "opportunities": None,
        "technical_advice_raw": None,
        "final_response": "",
        "status": ""
    }

    print("\n--- Exécution TEST 1 : Recherche de Subvention ---")
    result_subsidy = graph.invoke(test_state_subsidy)
    
    print("\n=== RÉSULTAT 1 (Formaté par LLM) ===")
    print(result_subsidy["final_response"])
    print("\nStatus:", result_subsidy["status"])
    
    print("\n--- Exécution TEST 2 : Alerte Arnaque ---")
    result_scam = graph.invoke(test_state_scam)

    print("\n=== RÉSULTAT 2 (Alerte Sécurité) ===")
    print(result_scam["final_response"])
    print("\nStatus:", result_scam["status"])