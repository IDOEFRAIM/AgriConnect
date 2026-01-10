from typing import Dict, Any, List
from datetime import datetime

class MeteoProcessor:
    """
    Transforme les données brutes JSON (Highcharts) en paragraphes descriptifs riches
    pour améliorer la qualité sémantique de l'embedding (RAG).
    """

    MONTHS_MAP = {
        0: "Janvier", 1: "Février", 2: "Mars", 3: "Avril", 4: "Mai", 5: "Juin",
        6: "Juillet", 7: "Août", 8: "Septembre", 9: "Octobre", 10: "Novembre", 11: "Décembre"
    }

    @staticmethod
    def process_highcharts_series(city: str, series_data: List[Dict[str, Any]]) -> str:
        """
        Convertit une série temporelle JSON en texte narratif.
        
        Args:
            city: Nom de la ville (ex: 'Bobo Dioulasso')
            series_data: Liste de points [{x: 0, y: 19.1}, {x: 1, y: 22}...] où x=mois
        """
        # Trier par mois (x)
        sorted_points = sorted(series_data, key=lambda p: p.get('x', 0))
        
        paragraphs = []
        summmary_min = float('inf')
        summary_max = float('-inf')
        month_min = ""
        month_max = ""
        
        # 1. Analyse statistique simple
        for pt in sorted_points:
            month_idx = pt.get('x')
            temp = pt.get('y')
            if temp is None: continue
            
            if temp < summmary_min:
                summmary_min = temp
                month_min = MeteoProcessor.MONTHS_MAP.get(month_idx, "Inconnu")
            if temp > summary_max:
                summary_max = temp
                month_max = MeteoProcessor.MONTHS_MAP.get(month_idx, "Inconnu")

        # 2. Génération Narrative
        intro = f"Climatologie de {city} : Analyse des températures."
        stat_summary = (f"Les températures varient de {summmary_min}°C (le minimum annuel en {month_min}) "
                        f"à {summary_max}°C (le maximum annuel en {month_max}).")
        
        # 3. Analyse saisonnière (Agronomie)
        season_details = []
        for pt in sorted_points:
            m_idx = pt.get('x', 0)
            month_name = MeteoProcessor.MONTHS_MAP.get(m_idx)
            val = pt.get('y')
            
            # Interprétation agronomique basique
            context = ""
            if val > 25 and m_idx in [2, 3, 4]: # Mars/Avril/Mai
                context = "(Période chaude, risque de stress hydrique pour les jeunes plants)."
            elif val < 20 and m_idx in [11, 0, 1]: # Dec/Jan/Fev
                context = "(Période fraîche, favorable aux cultures maraîchères mais croissance ralentie)."
            
            season_details.append(f"- {month_name}: {val}°C {context}")
            
        full_text = f"{intro}\n\n{stat_summary}\n\nDétails mensuels :\n" + "\n".join(season_details)
        return full_text

class MarketProcessor:
    """
    Transforme les données de marché brutes en texte d'analyse pour l'embedding.
    """
    
    @staticmethod
    def prices_to_text(prices_data: List[Dict[str, Any]]) -> str:
        # Implémentation future pour transformer le CSV/JSON SONAGESS
        return "Analyse des prix du marché non implémentée pour l'instant."
