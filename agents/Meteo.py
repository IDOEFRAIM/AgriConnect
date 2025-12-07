from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
import logging
from Tools.meteo.basis_tools import MeteoAnalysisToolkit

# Configuration du logger
logger = logging.getLogger("agent.meteo_analysis")


class AgentState(TypedDict):
    zone_id: str
    user_query: str                  # La question posée (ex: "Puis-je semer le maïs ?")
    meteo_data: Optional[Dict]       # Données brutes (Temp, Vent, Pluie...)
    culture_config: Dict[str, Any]   # Config culture (ex: {"crop_name": "Maïs", "stage": "semis"})
    
    # --- OUTPUTS (Produits par cet Agent) ---
    agri_indicators: Optional[Dict]  # Calculs (ET0, GDD, Delta T...)
    alerts: List[str]                # Liste d'avertissements
    final_response: str              # La réponse textuelle structurée
    status: str                      # SUCCESS / ERROR


class MeteoAgent:
    def __init__(self):
        self.name = "MeteoAnalysisService"
        # Initialisation de la boîte à outils sahélienne
        self.toolkit = MeteoAnalysisToolkit()

    def analyze_node(self, state: AgentState) -> AgentState:
        """
        Le cœur de l'Agent Météo :
        1. Valide les données d'entrée.
        2. Calcule les indicateurs agrométéorologiques.
        3. Génère les alertes de risques.
        4. Formule une réponse adaptée au contexte sahélien.
        """
        zone = state["zone_id"]
        data = state.get("meteo_data")
        config = state.get("culture_config", {})
        query = state.get("user_query", "").lower()
        
        logger.info(f"[{self.name}] Analyse en cours pour {zone} | Culture: {config.get('crop_name')}")

        # --- ÉTAPE 1 : VALIDATION ---
        if not data or "current" not in data:
            logger.error(f"[{self.name}] Pas de données météo fournies.")
            return {
                **state,
                "status": "ERROR",
                "final_response": f"Impossible d'analyser la météo pour {zone}. Données manquantes.",
                "alerts": ["DONNÉES MANQUANTES"]
            }

        try:
            # --- ÉTAPE 2 : CALCULS ---
            indicators = self.toolkit.calculate_indicators(data, config)
            
            # --- ÉTAPE 3 : RISQUES ---
            risk_alerts = self.toolkit.run_risk_analysis(indicators, data, config)
            
            # --- ÉTAPE 4 : RÉPONSE ---
            response_parts = []
            current = data["current"]
            crop_name = config.get("crop_name", "Culture")

            # A. Intention : SEMIS
            if "semis" in query or "semer" in query:
                sowing_advice = self.toolkit.risk.evaluate_sowing_conditions(
                    rain_last_3_days=current.get("precip_mm", 0) * 3,  # simplification
                    soil_type="sableux"  # par défaut au Sahel
                )
                response_parts.append(f"AVIS SEMIS {crop_name} : {sowing_advice['message']}")

            # B. Intention : TRAITEMENT
            elif "traiter" in query or "pulvériser" in query or "engrais" in query:
                spray_advice = self.toolkit.risk.evaluate_phytosanitary_conditions(
                    wind_speed=current["wind_speed_kmh"],
                    delta_t=indicators["Delta_T"],
                    rain_forecast_24h=0
                )
                response_parts.append(f"AVIS TRAITEMENT : {spray_advice['message']}")
                response_parts.append(f"Détails : Vent {current['wind_speed_kmh']} km/h | Delta T {indicators['Delta_T']} °C")

            # C. Intention : IRRIGATION
            elif "arroser" in query or "irrigation" in query or "eau" in query:
                bilan = indicators["Bilan_Hydrique_Jour"]
                if bilan < -2:
                    response_parts.append(f"AVIS IRRIGATION : Nécessaire. Bilan hydrique négatif ({bilan} mm).")
                    response_parts.append(f"L'évapotranspiration ({indicators['ETc']} mm) dépasse les apports de pluie.")
                else:
                    response_parts.append("AVIS IRRIGATION : Pas urgent. Le bilan hydrique est stable.")

            # D. Bulletin général
            else:
                response_parts.append(f"Bulletin Agrométéo : {zone}")
                response_parts.append(f"Culture : {crop_name} (Stade : {config.get('stage', 'N/A')})")
                response_parts.append(f"Conditions : T° {current['temp_max']} °C | Humidité {current['humidity']} % | Vent {current['wind_speed_kmh']} km/h")
                response_parts.append("Indicateurs clés :")
                response_parts.append(f"- ET0 (Évapotranspiration) : {indicators['ET0']} mm/jour")
                response_parts.append(f"- Delta T (Pulvérisation) : {indicators['Delta_T']} °C")
                
                if risk_alerts:
                    response_parts.append("Alertes vigilance :")
                    for alert in risk_alerts:
                        response_parts.append(f"- {alert}")
                else:
                    response_parts.append("Aucun risque majeur détecté pour la culture.")

            final_text = "\n".join(response_parts)

            logger.info(f"[{self.name}] Analyse terminée avec succès.")
            
            return {
                **state,
                "agri_indicators": indicators,
                "alerts": risk_alerts,
                "final_response": final_text,
                "status": "SUCCESS"
            }

        except Exception as e:
            logger.error(f"[{self.name}] Erreur interne de calcul : {e}", exc_info=True)
            return {
                **state,
                "status": "ERROR",
                "final_response": "Erreur technique lors des calculs agronomiques.",
                "alerts": [f"ERREUR CALCUL: {str(e)}"]
            }

    # ==========================================================================
    # 3. CONSTRUCTION DU GRAPHE
    # ==========================================================================
    def get_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("analyze_meteo", self.analyze_node)
        workflow.set_entry_point("analyze_meteo")
        workflow.add_edge("analyze_meteo", END)
        return workflow.compile()
    

if __name__ == "__main__":
    agent = MeteoAgent()
    graph = agent.get_graph()
    # pour les tests, nous avons mis un mockdata
    dummy_state = {
        "zone_id": "ZoneTest",
        "user_query": "Puis-je semer le maïs ?",
        "meteo_data": {
            "current": {
                "temp_max": 30,
                "temp_min": 15,
                "humidity": 60,
                "wind_speed_kmh": 10,
                "precip_mm": 5
            }
        },
        "culture_config": {"crop_name": "Maïs", "stage": "semis"},
        "agri_indicators": None,
        "alerts": [],
        "final_response": "",
        "status": ""
    }

    result = graph.invoke(dummy_state)
    print(result["final_response"])
