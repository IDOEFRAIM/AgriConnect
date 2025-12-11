from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional, Callable # Callable est important

# NOUVELLES IMPORTATIONS pour le LLM
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_community.chat_models import ChatOllama 

import logging
from Tools.meteo.basis_tools import MeteoAnalysisToolkit, SahelAgroMath, SahelCropKnowledgeBase

logger = logging.getLogger("agent.meteo_analysis")


class AgentState(TypedDict):
    # ... (État inchangé)
    zone_id: str
    user_query: str
    meteo_data: Optional[Dict]
    culture_config: Dict[str, Any]

    agri_indicators: Optional[Dict]
    alerts: List[str]
    technical_advice_raw: str # <-- NOUVEAU: Pour stocker la sortie brute
    final_response: str
    status: str


class MeteoAgent:
    def __init__(self, llm_client: Optional[Runnable] = None): # Accepte un client LLM
        self.name = "MeteoAnalysisService"
        self.math = SahelAgroMath()
        self.crop_db = SahelCropKnowledgeBase()
        self.toolkit = MeteoAnalysisToolkit()
        self.llm_client = llm_client
        if not self.llm_client:
             logger.warning("Client ChatOllama non fourni. Les réponses seront brutes.")


    def analyze_node(self, state: AgentState) -> AgentState:
        """
        Nœud 1 : Exécute l'analyse agrométéo et génère la sortie RAW.
        """
        zone = state["zone_id"]
        data = state.get("meteo_data")
        config = state.get("culture_config", {})
        query = state.get("user_query", "").lower()

        logger.info(f"[{self.name}] Analyse météo pour {zone} | Culture: {config.get('crop_name')}")
        
        # ... (Logique de validation et de calcul des indicateurs ET0, GDD, Delta T... (inchangée))
        # 

        if not data or "current" not in data:
             # ... (Retour d'erreur inchangé)
             return {
                 **state,
                 "status": "ERROR",
                 "final_response": f"Données météo manquantes pour {zone}.",
                 "alerts": ["DONNÉES MANQUANTES"]
             }

        try:
             current = data["current"]
             crop_profile = self.crop_db.get_crop_config(config.get("crop_name", ""))

             # ✅ Calcul des indicateurs agro-météo (inchangé)
             et0 = self.math.calculate_hargreaves_et0(t_min=current["temp_min"], t_max=current["temp_max"], t_mean=(current["temp_min"] + current["temp_max"]) / 2)
             gdd = self.math.calculate_gdd(t_min=current["temp_min"], t_max=current["temp_max"], profile=crop_profile)
             delta_t_info = self.math.calculate_delta_t(temp_c=current["temp_max"], humidity_pct=current["humidity"])
             effective_rain = self.math.calculate_effective_rain(precip_mm=current.get("precip_mm", 0), soil="standard")

             indicators = {
                 "ET0": et0, "GDD": gdd, "Delta_T": delta_t_info["Delta_T"], 
                 "DeltaT_Advice": delta_t_info["Advice"], "Rain_Effective": effective_rain
             }

             # ✅ Analyse des risques (inchangée)
             alerts = []
             heat_stress = self.toolkit.check_heat_stress(t_max=current["temp_max"], crop_profile=crop_profile)
             if heat_stress: alerts.append(heat_stress)

             # ✅ Construction de la réponse RAW
             response_parts = []
             crop_name = config.get("crop_name", "Culture")
             
             # --- INTENTIONS ---
             if "semis" in query or "semer" in query:
                 sowing = self.toolkit.evaluate_sowing_conditions(rain_last_3_days=current.get("precip_mm", 0) * 3)
                 response_parts.append(f"**INTENTION SEMIS ({crop_name})** : {sowing['message']}")

             elif "traiter" in query or "pulvériser" in query:
                 spray = self.toolkit.evaluate_phytosanitary_conditions(
                     wind_speed=current["wind_speed_kmh"],
                     delta_t=indicators["Delta_T"],
                     rain_forecast_24h=0
                 )
                 response_parts.append(f"**INTENTION TRAITEMENT PHYTO** : {spray['message']}")

             else:
                 # ✅ Bulletin général RAW
                 response_parts.append(f"**BULLETIN AGROMÉTÉO BRUT** – {zone}")
                 response_parts.append(f"Culture : {crop_name} (Stade : {config.get('stage', 'N/A')})")
                 response_parts.append(f"T° max : {current['temp_max']}°C | Humidité : {current['humidity']}% | Vent : {current['wind_speed_kmh']} km/h")
                 response_parts.append("INDICATEURS CLÉS :")
                 response_parts.append(f"- Évapotranspiration (ET0) : {et0:.2f} mm/j")
                 response_parts.append(f"- Jours-Degrés (GDD) : {gdd:.2f}")
                 response_parts.append(f"- Delta T (Traitement) : {indicators['Delta_T']:.1f}°C ({delta_t_info['Advice']})")
                 response_parts.append(f"- Pluie efficace : {effective_rain:.1f} mm")

                 if alerts:
                     response_parts.append("⚠️ ALERTES DÉTECTÉES :")
                     for a in alerts:
                         response_parts.append(f"- {a}")
                 else:
                     response_parts.append("✅ CONSTAT : Aucun risque majeur détecté.")

             technical_advice_raw = "\n".join(response_parts)

             return {
                 **state,
                 "agri_indicators": indicators,
                 "alerts": alerts,
                 "technical_advice_raw": technical_advice_raw, # On stocke le RAW ici
                 "status": "RAW_ANALYSIS_COMPLETE"
             }

        except Exception as e:
            # ... (Gestion d'erreur inchangée)
            logger.error(f"Erreur interne : {e}", exc_info=True)
            return {
                **state,
                "status": "ERROR",
                "final_response": "Erreur technique lors de l'analyse météo.",
                "alerts": [str(e)]
            }


    def llm_formatter_node(self, state: AgentState) -> AgentState:
        """
        Nœud 2 : Utilise ChatOllama pour transformer l'analyse technique brute 
        en un bulletin convivial.
        """
        raw_advice = state.get("technical_advice_raw", "")
        user_query = state.get("user_query", "")
        
        if not self.llm_client:
            # Fallback : retourner la sortie brute si le LLM n'est pas là
            return {**state, "final_response": raw_advice, "status": "FALLBACK_RAW_OUTPUT"}

        logger.info(f"[{self.name}] Début du formatage LLM avec ChatOllama.")

        # --- Définition du Prompt pour le LLM Léger ---
        system_prompt = (
            "Tu es un météorologue agricole bienveillant et professionnel. "
            "Ton objectif est de présenter le 'Bulletin Technique Brut' de manière claire, "
            "en mettant l'accent sur les actions recommandées (Semis, Traitement, Irrigation). "
            "Utilise un format lisible (listes, gras) et des emojis pertinents. "
            "Ne change pas les chiffres des indicateurs."
        )
        
        human_prompt = f"""
        **Demande de l'agriculteur :** "{user_query}"
        **Bulletin Technique Brut :**
        ---
        {raw_advice}
        ---
        
        Rédige le bulletin agrométéo final pour l'agriculteur.
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

    # ==============================================================================
    # 3. WORKFLOW LANGGRAPH (MISE À JOUR)
    # ==============================================================================
    def get_graph(self):
        """Construit et compile le Graph de l'Agent avec l'étape LLM."""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("analyze_meteo", self.analyze_node)       # Outil technique (génère la sortie RAW)
        workflow.add_node("format_llm_response", self.llm_formatter_node) # LLM léger (formatage)
        
        workflow.set_entry_point("analyze_meteo")
        workflow.add_edge("analyze_meteo", "format_llm_response")
        workflow.add_edge("format_llm_response", END)
        
        return workflow.compile()
    
# ==============================================================================
# 4. EXEMPLE D'UTILISATION (Mise à jour du __main__)
# ==============================================================================
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
    agent = MeteoAgent(llm_client=ollama_client)
    graph = agent.get_graph()

    # Données météo de test
    dummy_state = {
        "zone_id": "Zone de Mopti",
        "user_query": "Puis-je semer le maïs ? Les conditions sont-elles bonnes pour traiter ?",
        "meteo_data": {
            "current": {
                "temp_max": 30,
                "temp_min": 15,
                "humidity": 60,
                "wind_speed_kmh": 10,
                "precip_mm": 5
            }
        },
        "culture_config": {
            "crop_name": "Maïs",
            "stage": "Développement"
        },
        "agri_indicators": None,
        "alerts": [],
        "technical_advice_raw": "", # Ajout de la clé
        "final_response": "",
        "status": ""
    }

    print("\n--- Exécution du Graph ---")
    
    # Exécution
    result = graph.invoke(dummy_state)

    print("\n==============================")
    print("✅ RÉSULTAT FINAL FORMATÉ PAR LLM")
    print("==============================\n")
    print(result["final_response"])
    print("\n------------------------------")
    print("Indicateurs calculés (pour debug) :")
    print(result["agri_indicators"])
    print("------------------------------")
    print("Status :", result["status"])
    # Note: Le 'technical_advice_raw' n'est pas affiché, car il est remplacé par le 'final_response' formaté.