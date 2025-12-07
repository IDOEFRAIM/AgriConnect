from dataclasses import dataclass
from typing import Dict, List, Optional

# ==============================================================================
# 1. BASE DE CONNAISSANCE PÉDOLOGIQUE (SAHEL BURKINA)
# ==============================================================================

@dataclass
class SoilTypeProfile:
    code: str
    name: str
    description: str
    water_retention: str  # "Faible", "Moyenne", "Forte"
    risks: List[str]      # Ex: "Lessivage", "Asphyxie", "Érosion"
    ces_technique: str    # Technique de conservation recommandée (Zaï, etc.)

class SahelSoilDB:
    """
    Expertise sur les types de sols au Burkina Faso (zones sahéliennes et soudano-sahéliennes).
    """
    TYPES = {
        "sableux": SoilTypeProfile(
            code="SABLE",
            name="Sol Sableux / Dior",
            description="Sol léger, filtrant, pauvre en éléments nutritifs.",
            water_retention="Faible (l'eau s'infiltre rapidement)",
            risks=["Lessivage des engrais (l'urée descend trop vite)", "Sécheresse rapide"],
            ces_technique="Zaï et poquets fertilisés pour concentrer l'eau et la fumure"
        ),
        "argileux": SoilTypeProfile(
            code="ARGILE",
            name="Sol Argileux / Bas-fonds",
            description="Sol lourd, collant quand mouillé, dur quand sec.",
            water_retention="Forte (garde l'eau longtemps)",
            risks=["Asphyxie racinaire en cas d'excès d'eau", "Travail du sol difficile"],
            ces_technique="Billonnage ou buttage pour surélever les cultures"
        ),
        "gravillonnaire": SoilTypeProfile(
            code="GRAVIER",
            name="Sol Ferrugineux / Gravillonnaire",
            description="Sol rouge avec cailloux (latérite).",
            water_retention="Faible",
            risks=["Érosion hydrique", "Enracinement difficile"],
            ces_technique="Cordons pierreux et demi-lunes pour retenir l'eau"
        ),
        "limoneux": SoilTypeProfile(
            code="LIMON",
            name="Sol Limoneux (Boulis)",
            description="Sol battant, forme une croûte imperméable après la pluie.",
            water_retention="Moyenne",
            risks=["Ruissellement (l'eau glisse en surface)", "Croûte de battance"],
            ces_technique="Griffage ou scarifiage après pluie, paillage pour limiter la croûte"
        )
    }

# ==============================================================================
# 2. OUTILS D'ANALYSE (LOGIQUE MÉTIER)
# ==============================================================================

class SoilDoctorTool:
    """
    L'outil principal de l'Agent SOL pour conseiller les producteurs sahéliens.
    """
    
    def analyze_texture(self, texture_input: str) -> Dict:
        """
        Recommande la technique de conservation (CES) selon la texture du sol.
        """
        key = texture_input.lower()
        matched_profile = None
        for t_key, profile in SahelSoilDB.TYPES.items():
            if t_key in key:
                matched_profile = profile
                break
        
        if not matched_profile:
            return {"found": False, "message": "Type de sol non reconnu (Essayez: sableux, argileux, gravillonnaire, limoneux)."}
            
        return {
            "found": True,
            "profile": matched_profile.name,
            "retention": matched_profile.water_retention,
            "risks": matched_profile.risks,
            "recommendation_ces": f"Technique conseillée : {matched_profile.ces_technique}."
        }

    def analyze_ph(self, ph_value: float) -> Dict:
        """
        Analyse l'acidité du sol (important en zone cotonnière et céréalière).
        """
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
        Conseils sur la matière organique (clé de la fertilité sahélienne).
        """
        if any(word in soil_state.lower() for word in ["pauvre", "clair", "fatigué"]):
            return (
                "Amendement organique urgent :\n"
                "- Sol épuisé, faible fertilité.\n"
                "- Apporter 5 à 10 charrettes de fumier bien décomposé par hectare.\n"
                "- Si possible, préparer du compost en tas avant la saison des pluies."
            )
        else:
            return (
                "Entretien organique :\n"
                "- Apporter 2 à 3 charrettes de fumier par hectare tous les 2 ans.\n"
                "- Maintenir la fertilité par des apports réguliers de compost ou résidus de culture."
            )

    def recommend_phosphorus(self) -> str:
        """
        Spécifique Burkina : le phosphore est le facteur limitant majeur.
        """
        return (
            "Conseil Phosphore (Burkina Faso) :\n"
            "- La majorité des sols sahéliens manquent de phosphore.\n"
            "- Utiliser le Burkina Phosphate (BP) en fumure de fond (200-300 kg/ha).\n"
            "- Moins coûteux que le NPK importé et améliore durablement la fertilité."
        )
