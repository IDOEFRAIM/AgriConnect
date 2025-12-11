import math
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class CropProfile:
    name: str
    t_base: float          # Température seuil de croissance
    t_max_optimal: float   # Température seuil de stress
    kc_ini: float          # Coefficient cultural début
    kc_mid: float          # Coefficient cultural mi-saison
    kc_end: float          # Coefficient cultural fin
    cycle_days: int        # Durée moyenne du cycle
    drought_sensitive: bool


# Base de connaissances cultures
class SahelCropKnowledgeBase:
    def __init__(self):
        self._crops = {
            "maïs": CropProfile("Maïs", 10, 35, 0.3, 1.2, 0.6, 90, True),
            "sorgho": CropProfile("Sorgho", 10, 40, 0.3, 1.1, 0.55, 110, False),
            "mil": CropProfile("Mil", 12, 42, 0.3, 1.0, 0.5, 100, False),
            "coton": CropProfile("Coton", 15, 38, 0.35, 1.2, 0.7, 150, True),
            "niébé": CropProfile("Niébé", 12, 36, 0.4, 1.0, 0.35, 70, False),
        }

    def get_crop_config(self, crop_name: str) -> CropProfile:
        name = crop_name.lower()
        for key, profile in self._crops.items():
            if key in name:
                return profile
        return CropProfile("Standard", 10, 35, 0.5, 1.0, 0.5, 100, False)


class SahelAgroMath:
    """
    Bibliothèque de calculs agrométéorologiques adaptés aux conditions tropicales sèches.
    """

    def __init__(self):
        self.GSC = 0.0820  # constante solaire

    def ra_extraterrestre(self, latitude_deg: float, doy: int = 180) -> float:
        """
        Calcule Ra (MJ/m^2/jour) pour une latitude donnée et un jour de l'année.
        """
        phi = math.radians(latitude_deg)
        dr = 1 + 0.033 * math.cos(2 * math.pi * doy / 365.0)
        delta = 0.409 * math.sin(2 * math.pi * doy / 365.0 - 1.39)
        x = -math.tan(phi) * math.tan(delta)
        x = max(-1.0, min(1.0, x))
        omega_s = math.acos(x)
        ra = (24 * 60 / math.pi) * self.GSC * dr * (
            omega_s * math.sin(phi) * math.sin(delta) +
            math.cos(phi) * math.cos(delta) * math.sin(omega_s)
        )
        return ra

    def calculate_hargreaves_et0(self, t_min: float, t_max: float, t_mean: float, lat: float = 12.3) -> float:
        """
        Calcule ET0 selon Hargreaves-Samani.
        """
        ra = self.ra_extraterrestre(lat)
        et0 = 0.0023 * 0.408 * ra * (t_mean + 17.8) * math.sqrt(max(0.1, t_max - t_min))
        return round(et0, 2)

    @staticmethod
    def calculate_gdd(t_min: float, t_max: float, profile: CropProfile) -> float:
        """
        Degrés-Jours de Croissance (GDD).
        """
        effective_t_max = min(t_max, profile.t_max_optimal)
        effective_t_min = max(t_min, profile.t_base)
        gdd = ((effective_t_max + effective_t_min) / 2) - profile.t_base
        return round(max(0, gdd), 2)

    @staticmethod
    def calculate_effective_rain(precip_mm: float, soil: str = "standard") -> float:
        """
        Pluie efficace selon type de sol.
        """
        coeffs = {
            "sableux": 0.45,
            "argileux": 0.65,
            "amenage": 0.85,
            "standard": 0.6
        }
        c = coeffs.get(soil, 0.6)
        if precip_mm < 5:
            return 0.0
        if precip_mm > 75:  # pertes par ruissellement
            precip_mm *= 0.7
        return round(c * precip_mm, 2)

    def wet_bulb_stull(self, temp_c: float, rh_pct: float) -> float:
        T = max(5.0, min(45.0, temp_c))
        RH = max(5.0, min(95.0, rh_pct))
        Tw = (
            T * math.atan(0.151977 * math.sqrt(RH + 8.313659))
            + math.atan(T + RH)
            - math.atan(RH - 1.676331)
            + 0.00391838 * (RH ** 1.5) * math.atan(0.023101 * RH)
            - 4.686035
        )
        return Tw

    def calculate_delta_t(self, temp_c: float, humidity_pct: float) -> dict:
        Tw = self.wet_bulb_stull(temp_c, humidity_pct)
        dT = round(temp_c - Tw, 1)
        if 2 <= dT <= 10:
            advice = "Conditions favorables à la pulvérisation."
        elif dT < 2:
            advice = "Trop humide, risque de ruissellement."
        else:
            advice = "Trop sec, risque d'évaporation."
        return {"Delta_T": dT, "Advice": advice}


class MeteoAnalysisToolkit:
    """
    Transforme les chiffres en conseils pour le paysan burkinabé.
    """

    @staticmethod
    def evaluate_sowing_conditions(rain_last_3_days: float, soil_type: str = "standard") -> dict:
        thresholds = {
            "sableux": 15.0,
            "argileux": 20.0,
            "limoneux": 18.0,
            "ferrugineux": 17.0,
            "amenage": 12.0,
            "standard": 18.0
        }
        threshold = thresholds.get(soil_type, thresholds["standard"])
        if rain_last_3_days >= threshold:
            return {"code": "SEMIS_FAVORABLE", "message": f"Humidité du sol favorable ({rain_last_3_days} mm). Semis recommandé.", "level": "INFO"}
        elif rain_last_3_days >= 5:
            return {"code": "SEMIS_RISQUE", "message": f"Humidité insuffisante ({rain_last_3_days} mm). Attendre > {threshold} mm.", "level": "WARNING"}
        else:
            return {"code": "SEMIS_IMPOSSIBLE", "message": "Sol trop sec pour semer.", "level": "CRITICAL"}

    def evaluate_phytosanitary_conditions(self, wind_speed: float, delta_t: float, rain_forecast_24h: float) -> dict:
        reasons = []
        is_safe = True
        level = "INFO"
        if wind_speed > 18:
            is_safe = False; level = "CRITICAL"; reasons.append(f"Vent trop fort ({wind_speed} km/h)")
        elif wind_speed > 12:
            level = "WARNING"; reasons.append(f"Vent modéré ({wind_speed} km/h)")
        if rain_forecast_24h > 3:
            is_safe = False; level = "CRITICAL"; reasons.append(f"Pluie prévue ({rain_forecast_24h} mm)")
        if delta_t < 2:
            is_safe = False; level = "WARNING"; reasons.append(f"Delta T trop bas ({delta_t}°C)")
        elif delta_t > 8:
            is_safe = False; level = "WARNING"; reasons.append(f"Delta T trop élevé ({delta_t}°C)")
        return {"can_spray": is_safe, "level": level, "message": "Conditions favorables." if is_safe else f"Ne pas traiter : {', '.join(reasons)}."}

    @staticmethod
    def check_heat_stress(t_max: float, crop_profile: CropProfile) -> Optional[str]:
        margin = t_max - crop_profile.t_max_optimal
        if margin > 3:
            return f"ALERTE CRITIQUE : T_max ({t_max}°C) dépasse de {margin:.1f}°C le seuil du {crop_profile.name}."
        elif margin > 1:
            msg = f"Alerte chaleur : T_max ({t_max}°C) dépasse légèrement le seuil du {crop_profile.name}."
            if crop_profile.drought_sensitive:
                msg += " Culture sensible à la sécheresse : impact aggravé."
            return msg
