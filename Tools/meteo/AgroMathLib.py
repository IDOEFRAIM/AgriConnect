from typing import Dict

class AgroMathLib:
    """
    Librairie statique de calculs agronomiques.
    Ne dépend d'aucun état, juste des entrées/sorties.
    """
    
    @staticmethod
    def calculate_gdd(t_max: float, t_min: float, t_base: float, t_cutoff: float = 30.0) -> float:
        """Calcule les Degrés-Jours de Croissance (Growing Degree Days)."""
        adjusted_t_max = min(t_max, t_cutoff)
        adjusted_t_min = min(t_min, t_cutoff)
        gdd = ((adjusted_t_max + adjusted_t_min) / 2) - t_base
        return round(max(0.0, gdd), 2)

    @staticmethod
    def calculate_et0(t_mean: float, wind_kmh: float, humidity: float) -> float:
        """
        Estimation simplifiée de l'évapotranspiration de référence (ET0).
        (Méthode simplifiée FAO ou Hargreaves pour l'exemple).
        """
        # Formule illustrative
        et0 = 0.0135 * (t_mean + 17.78) * (wind_kmh * 0.5) * (1 - humidity/100)
        # On force une valeur réaliste pour l'exemple si la formule renvoie n'importe quoi
        return round(max(3.0, et0), 2)

    @staticmethod
    def analyze_risk(t_max: float, humidity: float, culture_config: Dict) -> str:
        """Détermine le risque maladie selon les seuils de la culture."""
        seuil_temp = culture_config.get("risk_temp_threshold", 25)
        seuil_hum = culture_config.get("risk_humidity_threshold", 80)
        
        if t_max > seuil_temp and humidity > seuil_hum:
            return "RISQUE_FONGIQUE_ELEVÉ"
        return "RISQUE_NORMAL"