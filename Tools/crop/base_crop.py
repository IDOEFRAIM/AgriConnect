from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime, timedelta

# --- ITINÉRAIRES TECHNIQUES SAHEL (Burkina Faso, zone soudano-sahélienne) ---

@dataclass
class CropTechProfile:
    name: str
    cycle_days: int
    seeding_density: str     # Espacement adapté aux sols tropicaux
    seeds_per_hole: str      # Nombre de graines par poquet
    sowing_depth_cm: int
    fertilizer_schedule: Dict[str, str] # Jour après semis -> Action

class SahelAgronomyDB:
    """Base de données des itinéraires techniques adaptés au Burkina Faso (Sahel tropical)."""
    
    DATA = {
        "maïs": CropTechProfile(
            name="Maïs (Variétés composites locales)",
            cycle_days=95,
            seeding_density="80cm x 40cm (sols ferrugineux tropicaux)",
            seeds_per_hole="2 graines par poquet",
            sowing_depth_cm=5,
            fertilizer_schedule={
                "15": "NPK (14-23-14 ou 15-15-15) : 200 kg/ha. Appliquer après un bon orage (>20 mm).",
                "45": "Urée 46% : 100 kg/ha au début de la montaison. Enfouir et refermer pour éviter volatilisation."
            }
        ),
        "sorgho": CropTechProfile(
            name="Sorgho (Blanc/Rouge, variétés locales)",
            cycle_days=110,
            seeding_density="80cm x 40cm (cordons pierreux conseillés)",
            seeds_per_hole="Démarier à 3 plants maximum",
            sowing_depth_cm=3,
            fertilizer_schedule={
                "21": "NPK 15-15-15 : 100 kg/ha au premier sarclage (sol humide).",
                "45": "Urée 46% : 50 kg/ha si sol pauvre ou après pluie efficace."
            }
        ),
        "coton": CropTechProfile(
            name="Coton (zones SOFITEX)",
            cycle_days=150,
            seeding_density="80cm x 30cm en ligne continue ou 40cm en poquets",
            seeds_per_hole="Démarier à 2 plants vigoureux",
            sowing_depth_cm=3,
            fertilizer_schedule={
                "15": "NPK-SB (Spécial Coton) : 150 kg/ha après démariage.",
                "40": "Urée 46% : 50 kg/ha au début de la floraison.",
                "60": "Urée 46% : 50 kg/ha en pleine floraison."
            }
        ),
        "niébé": CropTechProfile(
            name="Niébé (variétés locales résistantes)",
            cycle_days=70,
            seeding_density="60cm x 30cm (sols sableux ou ferrugineux)",
            seeds_per_hole="2 graines par poquet",
            sowing_depth_cm=3,
            fertilizer_schedule={
                "0": "NPK 15-15-15 : 100 kg/ha au semis (fond).",
                "30": "Pas d'Urée (le niébé fixe l'azote naturellement)."
            }
        )
    }

class CropManagerTool:
    """
    Outil utilisé par l'Agent CROP pour gérer l'itinéraire technique sahélien.
    """
    
    def get_seeding_advice(self, crop_name: str) -> str:
        """
        Instructions de semis adaptées au Sahel burkinabè.
        Version robuste, contextualisée et production-ready.
        """

        crop_key = (crop_name or "").lower().strip()
        profile = SahelAgronomyDB.DATA.get(crop_key)

        if not profile:
            return f"Aucune donnée technique disponible pour la culture '{crop_name}'."

        lines = [
            f"Guide de Semis – {profile.name}",
            f"- Espacement recommandé : {profile.seeding_density}",
            f"- Densité : {profile.seeds_per_hole} graines par poquet",
            f"- Profondeur : {profile.sowing_depth_cm} cm (ne pas dépasser en zone sahélienne)",
            "",
            "Conseils pratiques (Sahel Burkina) :",
            "- Semer après une pluie utile d'au moins 20 mm pour garantir la levée.",
            "- Éviter de semer dans un sol sec : risque de levée irrégulière.",
            "- En sol sableux : privilégier des poquets légèrement enrichis en fumure organique.",
            "- En sol limoneux : casser la croûte de battance après la pluie si nécessaire.",
            "- En zone à poches de sécheresse : prévoir un ressemis rapide si la levée échoue.",
        ]

        return "\n".join(lines)

    def check_fertilizer_status(self, crop_name: str, days_after_sowing: int) -> str:
        """
        Vérifie si un apport d'engrais est nécessaire aujourd'hui.
        Version robuste et contextualisée pour le Sahel burkinabè.
        """

        # Normalisation robuste
        crop_key = (crop_name or "").lower().strip()
        profile = SahelAgronomyDB.DATA.get(crop_key)

        if not profile:
            return f"Culture inconnue : '{crop_name}'."

        # Sécurité : jours négatifs
        if days_after_sowing < 0:
            return "Le nombre de jours après semis ne peut pas être négatif."

        # Sécurité : semis trop récent
        if days_after_sowing < 3:
            return (
                f"Jour {days_after_sowing} : Trop tôt pour un apport d'engrais.\n"
                "Attendre la levée complète avant toute application."
            )

        # Recherche d'une fenêtre d'application
        for day_str, action in profile.fertilizer_schedule.items():
            target_day = int(day_str)

            # Fenêtre flexible : J-3 à J+5
            if target_day - 3 <= days_after_sowing <= target_day + 5:
                return (
                    f"✅ ACTION REQUISE (Jour {days_after_sowing})\n"
                    f"{action}\n\n"
                    "Conseil Sahel :\n"
                    "- Appliquer uniquement sur sol humide (après pluie ou arrosage).\n"
                    "- Éviter les fortes chaleurs (risque de volatilisation).\n"
                    "- Incorporer légèrement si possible pour limiter les pertes."
                )

        # Prochain apport
        next_steps = [
            int(d) for d in profile.fertilizer_schedule.keys()
            if int(d) > days_after_sowing
        ]

        if next_steps:
            delta = min(next_steps) - days_after_sowing
            return (
                f"Aucune action aujourd'hui (Jour {days_after_sowing}).\n"
                f"⏳ Prochain apport prévu dans {delta} jours."
            )

        # Tous les apports sont passés
        return (
            f"✅ Fertilisation terminée pour cette saison (Jour {days_after_sowing}).\n"
            "Surveiller l'état du feuillage et prévoir un apport foliaire si signes de carence."
        )


    def estimate_harvest(self, crop_name: str, sowing_date_str: str) -> str:
        """
        Estime la date de récolte pour une culture donnée.
        Version robuste, avec calcul réel et conseils sahéliens.
        """

        # Normalisation
        crop_key = (crop_name or "").lower().strip()
        profile = SahelAgronomyDB.DATA.get(crop_key)

        if not profile:
            return f"Culture inconnue : '{crop_name}'."

        # Vérification de la date
        try:
            sowing_date = datetime.strptime(sowing_date_str, "%Y-%m-%d")
        except ValueError:
            return (
                "Format de date invalide. Utilisez le format AAAA-MM-JJ.\n"
                "Exemple : 2025-06-15"
            )

        # Calcul de la date estimée
        harvest_date = sowing_date + timedelta(days=profile.cycle_days)

        # Construction du message
        return (
            f"Estimation Récolte – {profile.name}\n"
            f"- Cycle moyen : {profile.cycle_days} jours\n"
            f"- Semis effectué le : {sowing_date.strftime('%d/%m/%Y')}\n"
            f"- Récolte estimée : {harvest_date.strftime('%d/%m/%Y')}\n\n"
            "Conseils Sahel :\n"
            "- La date peut varier selon les pluies et la fertilité du sol.\n"
            "- En cas de stress hydrique prolongé, ajouter 7 à 15 jours.\n"
            "- Sur sols pauvres, la maturité peut être retardée.\n"
            "- Sur sols bien fumés, la récolte peut être légèrement avancée."
        )