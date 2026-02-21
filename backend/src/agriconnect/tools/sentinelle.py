
import json
import logging
from typing import Dict, Any, List, Optional
import requests
import re
import math
from .shared_math import SahelAgroMath

logger = logging.getLogger("SentinelleTool")

class SentinelleTool:
    """
    Outil d'analyse avancée pour l'Agent Sentinelle.
    Calcul les métriques composées (Stress hydrique, Risques).
    Utilise SahelAgroMath pour la cohérence des formules.
    """
    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.model_planner = "llama-3.1-8b-instant"

    # ------------------------------------------------------------------ #
    # Utilitaires métiers                                                #
    # ------------------------------------------------------------------ #



    def _fetch_real_weather(self, location_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Récupère la météo réelle via OpenMeteo."""
        # 1. Résolution des coordonnées
        # Par défaut Bobo-Dioulasso si non trouvé
        lat = 11.1772
        lon = -4.2979
        
        region_name = location_profile.get("zone", "").lower() or location_profile.get("village", "").lower()
        
        # Mapping simple des grandes villes (à enrichir ou connecter à une DB géo)
        coords = {
            "bobo": (11.1772, -4.2979),
            "dedougou": (12.4634, -3.4607),
            "ouahigouya": (13.5828, -2.4216),
            "fada": (12.0627, 0.3578),
            "ouaga": (12.3714, -1.5197),
            "koudougou": (12.2526, -2.3627),
            "kaya": (13.0917, -1.0841),
            "dori": (14.0353, -0.0344),
            "nouna": (12.7296, -3.8631)
        }
        
        for key, (plat, plon) in coords.items():
            if key in region_name:
                lat, lon = plat, plon
                break
                
        # 2. Appel API OpenMeteo (Forecast)
        try:
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "current_weather": True,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,et0_fao_evapotranspiration",
                "timezone": "auto"
            }
            resp = requests.get(url, params=params, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            
            # 3. Formatage pour Sentinel
            current = data.get("current_weather", {})
            daily = data.get("daily", {})
            
            return {
                "temp_c": current.get("temperature"),
                "wind_kph": current.get("windspeed"),
                "condition_code": current.get("weathercode"),
                "precip_mm": daily.get("precipitation_sum", [0])[0] if daily.get("precipitation_sum") else 0.0,
                "t_max": daily.get("temperature_2m_max", [35])[0] if daily.get("temperature_2m_max") else 35.0,
                "t_min": daily.get("temperature_2m_min", [25])[0] if daily.get("temperature_2m_min") else 25.0,
                "et0": daily.get("et0_fao_evapotranspiration", [5])[0] if daily.get("et0_fao_evapotranspiration") else 5.0,
                "source": "OpenMeteo RealTime"
            }
            
        except Exception as e:
            logger.warning("Echec OpenMeteo: %s. Utilisation de valeurs par défaut.", e)
            return {
                "temp_c": 30.0, "precip_mm": 0.0, "et0": 5.0, "source": "Fallback (Error)"
            }


    def _moderate_request(self, query: str) -> Dict[str, Any]:
        """Détecte les tentatives d'arnaque ou demandes hors-sujet."""
        if not self.llm:
            return {"is_scam": False}

        prompt = (
            "Tu es le modérateur de sécurité d'AgriConnect. "
            "Analyse ce message pour détecter : arnaque financière, phishing, fausse nouvelle climatique alarmiste.\n"
            f"Message : {query}\n"
            "Format JSON : {'is_scam': boolean, 'reason': 'explication'}"
        )
        
        try:
            resp = self.llm.chat.completions.create(
                model=self.model_planner,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            logger.warning("Erreur modération: %s", e)
            return {"is_scam": False}

    def _build_context(self, nodes) -> str:
        """Construit le contexte texte à partir des nœuds retrouvés."""
        if not nodes:
            return ""
        return "\n\n".join([n.text for n in nodes])

    def _serialize_sources(self, nodes) -> List[Dict[str, Any]]:
        """Extrait les métadonnées des sources."""
        sources = []
        for i, n in enumerate(nodes, 1):
            meta = n.metadata or {}
            sources.append({
                "index": i,
                "title": meta.get("file_name", "Document inconnu"),
                "score": n.score if hasattr(n, "score") else 0.0
            })
        return sources

    def _format_location(self, loc: Dict[str, Any]) -> str:
        """Formate la localisation pour le prompt."""
        if not loc:
            return "Zone inconnue"
        parts = [loc.get("village"), loc.get("zone"), loc.get("country")]
        return ", ".join([p for p in parts if p])

    def _fallback_response(self, query, location, hazards, risk_summary, metrics, flood, sources) -> str:
        """Réponse de secours si le LLM échoue."""
        precip = metrics.get('precip_mm', 0)
        condition = metrics.get('condition_code', 'conditions variables')
        
        # ASSERTIF: On donne directement les actions, pas "veuillez consulter"
        if precip > 50:
            action = "Protégez vos cultures basses, risque d'inondation. Creusez des rigoles de drainage."
        elif precip == 0 and metrics.get('et0', 5) > 6:
            action = "Arrosez vos plants ce soir entre 20h-22h. L'évaporation est forte."
        else:
            action = "Surveillez l'humidité du sol. Paillez si la terre craquelle."
        
        return (
            f"📍 {location}\n"
            f"🌤️ Conditions: {condition}\n"
            f"💧 Pluie prévue: {precip}mm\n\n"
            f"🚜 ACTION IMMÉDIATE:\n{action}"
        )


    def _compute_metrics(self, weather: Dict[str, Any], satellite: Dict[str, Any]) -> Dict[str, Any]:
        math_tool = SahelAgroMath()
        
        t_min = float(weather.get("temperature_min_c", weather.get("t_min", 22.0)) or 22.0)
        t_max = float(weather.get("temperature_max_c", weather.get("t_max", 32.0)) or 32.0)
        precip = float(weather.get("forecast_precip_mm", weather.get("precip_mm", weather.get("precip", 0.0))) or 0.0)
        humidity = float(weather.get("humidity", weather.get("rh", 45.0)) or 45.0)
        wind = float(weather.get("wind_speed_kmh", weather.get("wind_speed", 8.0)) or 8.0)
        dry_days = int(weather.get("dry_days_ahead", 0) or 0)
        ndvi = float(satellite.get("ndvi_anomaly", 0.0) or 0.0)
        doy = weather.get("doy", 180) # Jour de l'année par défaut (juillet)
        lat = weather.get("lat", 12.0) # Latitude par défaut Burkina

        temp_avg = (t_min + t_max) / 2
        
        # Utilisation de la mathématique partagée pour ET0
        et0 = math_tool.calculate_hargreaves_et0(t_min, t_max, lat, doy)
        
        # Calcul Indice Humidité Sol simplifié
        # (Pluie + Humidité/10 - Evaporation) normalisé entre 0 et 1
        soil_moisture_idx = max(0.0, min(1.0, (precip + humidity / 10 - et0) / 10.0))
        
        heat_flag = t_max >= 38 or temp_avg >= 32

        return {
            "temp_min_c": round(t_min, 1),
            "temp_max_c": round(t_max, 1),
            "temp_avg_c": round(temp_avg, 1),
            "precip_mm": round(precip, 1),
            "et0_mm": round(et0, 1),
            "soil_moisture_idx": round(soil_moisture_idx, 2),
            "humidity_pct": round(humidity, 1),
            "wind_speed_kmh": round(wind, 1),
            "dry_days_ahead": dry_days,
            "ndvi_anomaly": round(ndvi, 2),
            "heat_stress": heat_flag,
        }

    def _assess_flood_risk(
        self,
        weather: Dict[str, Any],
        satellite: Dict[str, Any],
        location: Dict[str, Any],
    ) -> Dict[str, Any]:
        precip = float(
            weather.get("forecast_precip_mm", weather.get("precip_mm", weather.get("precip", 0.0))) or 0.0
        )
        flood_prob = float(satellite.get("flood_probability", 0.0) or 0.0)
        soil = float(weather.get("soil_moisture", 0.0) or 0.0)

        score = 0.0
        if precip >= 30:
            score += 2.5
        elif precip >= 20:
            score += 1.5
        elif precip >= 10:
            score += 1.0

        if flood_prob >= 0.6:
            score += 2.5
        elif flood_prob >= 0.4:
            score += 1.5

        if soil >= 0.7:
            score += 1.0

        if score >= 4:
            level = "CRITIQUE"
            message = "Risque d'inondation imminent : sécuriser les intrants, prévoir des drains."
        elif score >= 2.5:
            level = "ÉLEVÉ"
            message = "Sol saturé et pluie forte : ouvrir les rigoles, surveiller les parcelles basses."
        elif score >= 1.5:
            level = "MODÉRÉ"
            message = "Arrosage naturel important : privilégier la rétention d'eau."
        else:
            level = "FAIBLE"
            message = "Pas de signe d'inondation à court terme."

        return {
            "score": round(score, 2),
            "risk_level": level,
            "alert_message": message,
            "precip_indicator_mm": round(precip, 1),
            "flood_probability": round(flood_prob, 2),
            "soil_saturation": round(soil, 2),
            "location": location.get("village") or "non précisé",
        }

    def _derive_hazards(self, metrics: Dict[str, Any], flood: Dict[str, Any]) -> List[Dict[str, Any]]:
        hazards: List[Dict[str, Any]] = []

        # 0. ALERTE SÉCHERESSE SAHÉLIENNE (Priorité Haute en saison sèche)
        # Si pluie nulle et évaporation forte
        if metrics.get("precip_mm", 0.0) < 0.1 and metrics.get("et0_mm", 0.0) >= 4.0:
            hazards.append(
                {
                    "label": "Sécheresse / Harmattan critique",
                    "severity": "CRITIQUE" if metrics.get("et0_mm", 0.0) > 6.0 else "HAUTE",
                    "advice": "PAILLAGE OBLIGATOIRE : Couvrez le sol pour bloquer l'évaporation. IRRIGATION DE NUIT : Arrosez entre 22h et 5h pour maximiser l'absorption.",
                    "explanation": f"L'absence de pluie (0.0mm) est confirmée par l'air sec. L'Harmattan 'vole' {metrics['et0_mm']}mm d'eau à votre sol chaque jour.",
                }
            )

        if flood.get("risk_level") in {"ÉLEVÉ", "CRITIQUE"}:
            hazards.append(
                {
                    "label": "Risque d'inondation",
                    "severity": flood["risk_level"],
                    "advice": "ZAÏ ET CORDONS PIERREUX : Préparez vos parcelles avec des aménagements physiques pour freiner l'eau et favoriser l'infiltration.",
                    "explanation": f"Menace d'excès d'eau : {flood.get('alert_message', '')}",
                }
            )

        if metrics.get("heat_stress"):
            hazards.append(
                {
                    "label": "Vague de chaleur extrême",
                    "severity": "ÉLEVÉ",
                    "advice": "CONSERVATION DE L'EAU : N'apportez pas d'engrais minéraux (Urée) tant que le sol n'est pas bien refroidi. Créez des ombrages si possible.",
                    "explanation": f"La température de {metrics['temp_max_c']}°C brûle les feuilles et assèche la sève.",
                }
            )

        if metrics.get("soil_moisture_idx", 0) <= 0.2 and metrics.get("precip_mm", 0) < 5:
            # On ne l'ajoute que si l'alerte Sécheresse n'est pas déjà là pour éviter les doublons
            if not any(h['label'] == "Sécheresse / Harmattan critique" for h in hazards):
                hazards.append(
                    {
                        "label": "Stress hydrique du sol",
                        "severity": "MOYEN",
                        "advice": "Réduisez l'évaporation par un paillage épais et privilégiez les micro-doses d'eau au pied des plants.",
                        "explanation": f"L'indice d'humidité de {metrics['soil_moisture_idx']} indique un sol proche du point de flétrissement.",
                    }
                )

        if metrics.get("ndvi_anomaly", 0) <= -0.2:
            hazards.append(
                {
                    "label": "Stress végétatif détecté",
                    "severity": "ÉLEVÉ",
                    "advice": "Inspecter les cultures pour ravageurs ou carences et corriger rapidement.",
                    "explanation": f"Anomalie NDVI {metrics['ndvi_anomaly']}.",
                }
            )

        if metrics.get("wind_speed_kmh", 0) >= 35:
            hazards.append(
                {
                    "label": "Vent fort",
                    "severity": "ÉLEVÉ",
                    "advice": "Reporter les pulvérisations et arrimer les abris.",
                    "explanation": f"Vent prévu {metrics['wind_speed_kmh']} km/h.",
                }
            )

        return hazards

    def _plan_retrieval(
        self,
        query: str,
        risk_summary: str,
        hazards: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        fallback = {
            "optimized_query": query,
            "warnings": ["Planification RAG automatique indisponible."],
        }

        if not self.llm:
            return fallback

        hazard_text = "; ".join(
            f"{item['label']} (niveau {item['severity']})"
            for item in hazards[:3]
        ) or "pas de risque critique"

        planner_prompt = (
            "Prépare une requête documentaire pour un agent de veille climatique.\n"
            f"Question terrain: {query}\n"
            f"Risques détectés: {hazard_text}\n"
            f"Résumé: {risk_summary}\n"
            'Retourne un JSON {"optimized_query": "...", "warnings": ["..."]}.'
        )

        try:
            completion = self.llm.chat.completions.create(
                model=self.model_planner,
                messages=[{"role": "user", "content": planner_prompt}],
                temperature=0.15,
                max_tokens=320,
                response_format={"type": "json_object"},
            )
            content = completion.choices[0].message.content
            if not content:
                raise ValueError("Réponse vide du planificateur.")
            data = json.loads(content)
            return {
                "optimized_query": data.get("optimized_query") or query,
                "warnings": data.get("warnings", []),
            }
        except Exception as exc:
            logger.warning("Planification échouée: %s", exc)
            return fallback

    # ------------------------------------------------------------------ #
    # Utilitaires sécurité & fallback                                    #
    # ------------------------------------------------------------------ #

    def _extract_json_block(self, text: str) -> Dict[str, Any]:
        matches = re.findall(r"\{[\s\S]*?\}", text)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        return json.loads(text)

    def _moderate_request(self, query: str) -> Dict[str, Any]:
        if not self.llm:
            return {"is_scam": False, "reason": "LLM indisponible"}
        system_prompt = (
            "Tu es le garde-fou d'AgriConnect. Détecte les arnaques, demandes d'argent ou phishing. "
            'Réponds en JSON {"is_scam": boolean, "reason": "..."}.'
        )
        try:
            response = self.llm.chat.completions.create(
                model=self.model_planner,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                temperature=0.05,
                max_tokens=200,
            )
            content = (response.choices[0].message.content or "{}").strip() if response.choices else "{}"
            return self._extract_json_block(content)
        except Exception as exc:
            logger.warning("Contrôle anti-fraude indisponible : %s", exc)
            return {"is_scam": False, "reason": "Contrôle impossible"}
 
    def _fallback_response(
        self,
        query: str,
        location: str,
        hazards: List[Dict[str, Any]],
        risk_summary: str,
        metrics: Dict[str, Any],
        flood: Dict[str, Any],
        sources: List[Dict[str, Any]],
    ) -> str:
        lines = [
            "⚠️ Alerte en mode dégradé (LLM/RAG indisponible).",
            f"Question: {query}",
            f"Zone: {location}",
            "",
            "Évaluation rapide:",
            risk_summary or "Pas d'information sur les risques.",
        ]
        if flood:
            lines.append("")
            lines.append(f"Risque inondation: {flood.get('risk_level', 'FAIBLE')} - {flood.get('alert_message', '')}")
        if metrics:
            lines.append(
                f"Température max {metrics.get('temp_max_c', '?')} °C | Pluie prévue {metrics.get('precip_mm', '?')} mm "
                f"| ET0 {metrics.get('et0_mm', '?')} mm | Humidité sol {metrics.get('soil_moisture_idx', '?')}"
            )

        if hazards:
            lines.append("")
            lines.append("Actions recommandées:")
            for idx, hazard in enumerate(hazards, start=1):
                lines.append(f"{idx}. {hazard['advice']} ({hazard['label']})")
        else:
            lines.append("")
            lines.append("Actions recommandées: Surveillance régulière du champ.")

        if sources:
            ref = ", ".join(
                item.get("title") or item.get("filename") or f"Source {item['index']}"
                for item in sources
            )
            lines.append("")
            lines.append(f"Sources à consulter: {ref}")

        lines.append("")
        lines.append("Consultez un technicien agricole si la situation évolue.")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _build_context(self, nodes: List[Any]) -> str:
        chunks = []
        for idx, node in enumerate(nodes, start=1):
            metadata = node.node.metadata or {}
            title = metadata.get("title") or metadata.get("filename") or f"Source {idx}"
            chunks.append(f"[Source {idx} | {title}]\n{node.node.get_content().strip()}")
        return "\n\n".join(chunks)

    def _serialize_sources(self, nodes: List[Any]) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for idx, node in enumerate(nodes, start=1):
            metadata = node.node.metadata or {}
            payload.append(
                {
                    "index": idx,
                    "title": metadata.get("title"),
                    "filename": metadata.get("filename"),
                    "score": float(node.score) if node.score is not None else None,
                }
            )
        return payload

    def _format_location(self, profile: Dict[str, Any]) -> str:
        if not profile:
            return "Localisation non précisée"
        parts = []
        if name := profile.get("village"):
            parts.append(name)
        if zone := profile.get("zone"):
            parts.append(zone)
        if country := profile.get("country"):
            parts.append(country)
        return ", ".join(parts) or "Localisation non précisée"

    def _simulate_weather(self, location: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "precip_mm": 0.0,
            "forecast_precip_mm": 0.0,
            "wind_speed_kmh": 8.0,
            "dry_days_ahead": 2,
            "temperature_max_c": 34.0,
            "temperature_min_c": 22.0,
            "humidity": 45.0,
            "location_label": location.get("village") or "inconnu",
        }



