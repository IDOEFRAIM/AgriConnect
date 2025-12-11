from dataclasses import dataclass
from typing import Dict, List, Optional


#  BASE DE CONNAISSANCE PÉDOLOGIQUE (STATIQUE, PRODUCTION):les parametres seront update en selon le sens metier

@dataclass(frozen=True)
class SoilTypeProfile:
    code: str
    name: str
    description: str
    water_retention: str
    risks: List[str]
    ces_technique: str


class SahelSoilDB:
    """
    Base statique des sols du Burkina Faso.
    Optimisée pour la production :  O(1) ou statistique, immuable, déterministe.
    """

    _TYPES: Dict[str, SoilTypeProfile] = {
        "sableux": SoilTypeProfile(
            code="SABLE",
            name="Sol Sableux / Dior",
            description="Sol léger, filtrant, pauvre en éléments nutritifs.",
            water_retention="Faible (l'eau s'infiltre rapidement)",
            risks=[
                "Lessivage des engrais (l'urée descend trop vite)",
                "Sécheresse rapide"
            ],
            ces_technique="Zaï et poquets fertilisés pour concentrer l'eau et la fumure"
        ),
        "argileux": SoilTypeProfile(
            code="ARGILE",
            name="Sol Argileux / Bas-fonds",
            description="Sol lourd, collant quand mouillé, dur quand sec.",
            water_retention="Forte (garde l'eau longtemps)",
            risks=[
                "Asphyxie racinaire en cas d'excès d'eau",
                "Travail du sol difficile"
            ],
            ces_technique="Billonnage ou buttage pour surélever les cultures"
        ),
        "gravillonnaire": SoilTypeProfile(
            code="GRAVIER",
            name="Sol Ferrugineux / Gravillonnaire",
            description="Sol rouge avec cailloux (latérite).",
            water_retention="Faible",
            risks=[
                "Érosion hydrique",
                "Enracinement difficile"
            ],
            ces_technique="Cordons pierreux et demi-lunes pour retenir l'eau"
        ),
        "limoneux": SoilTypeProfile(
            code="LIMON",
            name="Sol Limoneux (Boulis)",
            description="Sol battant, forme une croûte imperméable après la pluie.",
            water_retention="Moyenne",
            risks=[
                "Ruissellement (l'eau glisse en surface)",
                "Croûte de battance"
            ],
            ces_technique="Griffage ou scarifiage après pluie, paillage pour limiter la croûte"
        )
    }

    @classmethod
    def get_profile(cls, soil_type: str) -> Optional[SoilTypeProfile]:
        """
        Retourne un profil de sol statique.
        Fallback : sol sableux (le plus fréquent au Sahel).
        """
        if not soil_type:
            return cls._TYPES["sableux"]

        key = soil_type.lower().strip()
        return cls._TYPES.get(key, cls._TYPES["sableux"])
# 2. OUTILS D'ANALYSE (LOGIQUE MÉTIER)
class SoilDoctorTool:
    """
    Outil principal de l'Agent SOL.
    Analyse statique, déterministe, optimisée pour la production.
    """

    def analyze_texture(self, texture_input: str) -> Dict:
        """
        Analyse la texture du sol et recommande la technique CES adaptée.
        """
        profile = SahelSoilDB.get_profile(texture_input)

        if not profile:
            return {
                "found": False,
                "message": "Type de sol non reconnu (Essayez: sableux, argileux, gravillonnaire, limoneux)."
            }

        return {
            "found": True,
            "profile": profile.name,
            "retention": profile.water_retention,
            "risks": profile.risks,
            "recommendation_ces": f"Technique conseillée : {profile.ces_technique}."
        }

    def analyze_ph(self, ph_value: float) -> Dict:
        """
        Analyse l'acidité du sol (critique en zone sahélienne).
        """
        # Revoir cette partie avec les agronomes pour que les check soient coherents
        if ph_value < 5.5:
            return {
                "status": "ACIDE (Mauvais)",
                "impact": "L'acidité bloque l'assimilation des engrais minéraux (NPK).",
                "solution": "Apporter de la dolomie (400 kg/ha) ou de la cendre de bois avant les pluies."
            }

        elif 5.5 <= ph_value <= 7.5:
            return {
                "status": "NEUTRE (Idéal)",
                "impact": "Le sol est équilibré, les engrais sont bien absorbés.",
                "solution": "Maintenir des apports organiques réguliers (fumier, compost)."
            }

        else:
            return {
                "status": "ALCALIN (Basique)",
                "impact": "Blocage possible des oligo-éléments (Fer, Zinc).",
                "solution": "Éviter les cendres. Utiliser des engrais acidifiants comme le sulfate d'ammoniaque."
            }

    def recommend_organic_amendment(self, soil_state: str) -> str:
        """
        Analyse qualitative de l'état du sol et recommandations organiques.
        Optimisée pour les descriptions réelles des producteurs sahéliens.
        """

        state = soil_state.lower()

        #  Mots-clés indiquant un sol très pauvre:choisir les keyword de facon plus rigoureuse. Menez une vrai etude pour connaitre les mots les plus utilises
        severe_keywords = [
            "pauvre", "épuisé", "fatigué", "mort", "stérile",
            "clair", "jaune", "sec", "dur", "sans force"
        ]

        # mots-clés indiquant un sol moyen mais améliorable
        moderate_keywords = [
            "moyen", "pas très bon", "léger", "manque", "faible"
        ]

        # Détection des cas sévères
        if any(word in state for word in severe_keywords):
            return (
                "Amendement organique **URGENT** :\n"
                "- Le sol présente une faible fertilité.\n"
                "- Apporter **5 à 10 charrettes de fumier bien décomposé par hectare**.\n"
                "- Incorporer le fumier 2 à 3 semaines avant les pluies.\n"
                "- Si possible, préparer du **compost enrichi** (cendre + résidus + fumier).\n"
                "- Pratiquer le **Zaï** ou les **poquets fertilisés** pour concentrer l'eau et la matière organique."
            )

        # Détection des cas modérés
        if any(word in state for word in moderate_keywords):
            return (
                "Amendement organique recommandé :\n"
                "- Le sol a besoin d'être renforcé.\n"
                "- Apporter **3 à 5 charrettes de fumier** par hectare.\n"
                "- Ajouter des résidus de culture pour améliorer la structure.\n"
                "- En zone sèche, privilégier le compost plutôt que le fumier brut."
            )

        # Cas normal / entretien
        return (
            "Entretien organique :\n"
            "- Le sol semble en état correct.\n"
            "- Apporter **2 à 3 charrettes de fumier** par hectare tous les 2 ans.\n"
            "- Maintenir la fertilité avec compost, résidus de culture et rotations."
        )
    
    def recommend_phosphorus(self, soil_type: str, crop_name: str, organic_level: str = "moyen", budget: str = "moyen") -> str:
        """
        Recommandation phosphore dynamique, adaptée au sol, à la culture,
        au niveau de matière organique et au budget du producteur.
        """

        soil = soil_type.lower()
        crop = crop_name.lower()

        # Besoins en phosphore selon culture
        crop_needs = {
            "maïs": "élevé",
            "coton": "élevé",
            "sorgho": "moyen",
            "mil": "moyen",
            "niébé": "faible"
        }
        need = crop_needs.get(crop, "moyen")

        # Ajustement selon sol
        if "sable" in soil:
            soil_note = "Sol sableux : risque de lessivage, privilégier BP + fumier."
        elif "argile" in soil:
            soil_note = "Sol argileux : bonne rétention, NPK efficace."
        elif "limon" in soil:
            soil_note = "Sol limoneux : croûte de battance, incorporer profondément."
        else:
            soil_note = "Sol standard : appliquer en fumure de fond."

        # Ajustement selon matière organique
        if organic_level == "faible":
            organic_note = "Faible matière organique : augmenter les doses de BP et ajouter du fumier."
        elif organic_level == "élevé":
            organic_note = "Bonne matière organique : efficacité du phosphore améliorée."
        else:
            organic_note = "Matière organique moyenne : doses standards recommandées."

        # Ajustement selon budget
        if budget == "faible":
            dose = "200–300 kg/ha de Burkina Phosphate (BP), solution la plus économique."
        elif budget == "élevé":
            dose = "100–150 kg/ha de NPK 14-23-14 ou 15-15-15 pour un effet rapide."
        else:
            dose = "BP (200–300 kg/ha) ou NPK (100–150 kg/ha) selon disponibilité."

        return (
            f"Recommandation Phosphore pour {crop_name} :\n"
            f"- Besoin en phosphore : {need}\n"
            f"- {soil_note}\n"
            f"- {organic_note}\n"
            f"- Dose recommandée : {dose}\n\n"
            "Bonnes pratiques :\n"
            "- Incorporer dans les 5 premiers cm du sol.\n"
            "- Ne pas appliquer en surface.\n"
            "- Associer avec fumier ou compost pour maximiser l'efficacité."
        )

    
    def recommend_phosphorus(self, soil_type: str, crop_name: str, organic_level: str = "moyen", budget: str = "moyen") -> str:
        """
        Recommandation phosphore dynamique, adaptée au sol, à la culture,
        au niveau de matière organique et au budget du producteur.
        """

        # Normalisation robuste
        soil = (soil_type or "").lower().strip()
        crop = (crop_name or "").lower().strip()
        organic_level = (organic_level or "moyen").lower().strip()
        budget = (budget or "moyen").lower().strip()

        # Besoins en phosphore selon culture
        crop_needs = {
            "maïs": "élevé",
            "coton": "élevé",
            "sorgho": "moyen",
            "mil": "moyen",
            "niébé": "faible"
        }
        need = crop_needs.get(crop, "moyen")

        # Ajustement selon sol
        if "sable" in soil:
            soil_note = "Sol sableux : risque de lessivage, privilégier BP + fumier."
        elif "argile" in soil:
            soil_note = "Sol argileux : bonne rétention, NPK efficace."
        elif "limon" in soil:
            soil_note = "Sol limoneux : croûte de battance, incorporer profondément."
        else:
            soil_note = "Sol standard : appliquer en fumure de fond."

        # Ajustement selon matière organique
        if organic_level == "faible":
            organic_note = "Faible matière organique : augmenter les doses de BP et ajouter du fumier."
        elif organic_level == "élevé":
            organic_note = "Bonne matière organique : efficacité du phosphore améliorée."
        else:
            organic_note = "Matière organique moyenne : doses standards recommandées."

        # Ajustement selon budget
        if budget == "faible":
            dose = "200–300 kg/ha de Burkina Phosphate (BP), solution la plus économique."
        elif budget == "élevé":
            dose = "100–150 kg/ha de NPK 14-23-14 ou 15-15-15 pour un effet rapide."
        else:
            dose = "BP (200–300 kg/ha) ou NPK (100–150 kg/ha) selon disponibilité."

        return (
            f"Recommandation Phosphore pour {crop_name} :\n"
            f"- Besoin en phosphore : {need}\n"
            f"- {soil_note}\n"
            f"- {organic_note}\n"
            f"- Dose recommandée : {dose}\n\n"
            "Bonnes pratiques :\n"
            "- Incorporer dans les 5 premiers cm du sol.\n"
            "- Ne pas appliquer en surface.\n"
            "- Associer avec fumier ou compost pour maximiser l'efficacité."
        )
    """
    Ajoutez d autre functions utilitaires selon le besoin
    """

