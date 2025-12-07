from dataclasses import dataclass
from typing import List, Dict, Optional

# ==============================================================================
# 1. BASE DE DONNÉES DES PROGRAMMES (CONTEXTE BURKINA SAHÉLIEN)
# ==============================================================================

@dataclass
class SubsidyProgram:
    id: str
    name: str
    provider: str        # Ex: État, Projet Banque Mondiale, ONG
    category: str        # Ex: Intrants, Équipement, Cash
    target_crops: List[str]
    target_zones: List[str] # "NATIONAL" ou zones spécifiques
    requirements: List[str]
    deadline_period: str # Ex: "Mai - Juin"
    is_active: bool

class SahelGrantDB:
    """
    Répertoire simulé des opportunités de financement au Burkina Faso.
    """
    PROGRAMS = [
        SubsidyProgram(
            id="SUBV_ETAT_2025",
            name="Opération Intrants Agricoles (Engrais & Semences)",
            provider="Ministère de l'Agriculture (DGPV)",
            category="INTRANTS",
            target_crops=["maïs", "riz", "sorgho", "coton"],
            target_zones=["NATIONAL"],
            requirements=[
                "Être membre d'une SCOOPS/GIE (Coopérative) immatriculée",
                "Avoir été recensé par l'agent d'agriculture local",
                "Payer la caution (apport personnel)"
            ],
            deadline_period="Avril - Mai",
            is_active=True
        ),
        SubsidyProgram(
            id="PROJET_IRRIGATION",
            name="Kit Pompage Solaire (Projet PARIIS/PNSA)",
            provider="Projet Étatique/Bailleur",
            category="EQUIPEMENT",
            target_crops=["maraîchage", "oignon", "tomate"],
            target_zones=["Boucle du Mouhoun", "Nord", "Centre-Nord"],
            requirements=[
                "Disposer d'un titre foncier ou APFR (Attestation de Possession Foncière Rurale)",
                "Apport personnel de 20%",
                "Être dans une zone aménageable"
            ],
            deadline_period="Janvier - Mars",
            is_active=True
        ),
        SubsidyProgram(
            id="FEMMES_RESILIENTES",
            name="Fonds d'Appui aux Activités Génératrices (AGR)",
            provider="ONG / FAARF",
            category="CASH",
            target_crops=["niébé", "sésame", "karité"],
            target_zones=["NATIONAL"],
            requirements=[
                "Groupement féminin reconnu",
                "Compte bancaire ou Mobicash au nom du groupement"
            ],
            deadline_period="Toute l'année",
            is_active=True
        )
    ]

# ==============================================================================
# 2. MOTEUR DE VÉRIFICATION & SÉCURITÉ
# ==============================================================================

class GrantExpertTool:
    """
    Outil d'analyse pour l'Agent Subvention (adapté au contexte sahélien).
    """

    def find_opportunities(self, user_profile: Dict) -> List[Dict]:
        """
        Matche le profil du paysan avec les offres disponibles.
        """
        user_crop = user_profile.get("crop", "").lower()
        user_zone = user_profile.get("zone", "Inconnue")
        is_coop_member = user_profile.get("is_coop_member", False)
        gender = user_profile.get("gender", "M")

        matches = []
        
        for prog in SahelGrantDB.PROGRAMS:
            if not prog.is_active:
                continue

            # 1. Filtre Culture
            crop_match = any(target in user_crop or user_crop == "toutes" for target in prog.target_crops)
            
            # 2. Filtre Zone
            zone_match = ("NATIONAL" in prog.target_zones) or (user_zone in prog.target_zones)

            # 3. Filtre Genre (spécifique aux programmes femmes)
            gender_match = True
            if "FEMMES" in prog.id and gender != "F":
                gender_match = False

            if crop_match and zone_match and gender_match:
                missing_reqs = []
                if "SCOOPS" in str(prog.requirements) and not is_coop_member:
                    missing_reqs.append("Adhésion Coopérative (SCOOPS)")
                
                matches.append({
                    "program_name": prog.name,
                    "provider": prog.provider,
                    "status": "ÉLIGIBLE" if not missing_reqs else "CONDITIONNEL",
                    "missing_documents": missing_reqs,
                    "deadline": prog.deadline_period
                })
        
        return matches

    def check_scam(self, message_content: str) -> Dict:
        """
        Détecte les faux messages de subvention (arnaques fréquentes au Sahel).
        """
        msg = message_content.lower()
        red_flags = []
        
        keywords_risk = [
            "envoyer frais de dossier", "orange money", "western union", 
            "cliquez ici pour retirer", "fonds d'urgence", "onu", "faarf gratuit"
        ]
        
        for kw in keywords_risk:
            if kw in msg:
                red_flags.append(f"Mot suspect : '{kw}'")
        
        if "http" in msg and ".gov.bf" not in msg:
            red_flags.append("Lien non officiel (ne finit pas par .bf)")

        if red_flags:
            return {
                "is_scam": True,
                "risk_level": "CRITIQUE",
                "warning": "ARNAQUE détectée. Ne payez rien et vérifiez auprès des services agricoles.",
                "reasons": red_flags
            }
        else:
            return {
                "is_scam": False,
                "risk_level": "FAIBLE",
                "warning": "Semble légitime, mais vérifiez toujours auprès de la mairie ou du CRA."
            }

    def get_application_guide(self, program_type: str) -> str:
        """Procédure administrative adaptée au Burkina Faso."""
        if "intrant" in program_type.lower():
            return (
                "Procédure Subvention Engrais :\n"
                "1. Rejoindre une SCOOPS (coopérative) dans votre village.\n"
                "2. Vous faire enregistrer sur la liste du CRA (Chambre Régionale d'Agriculture).\n"
                "3. Payer votre caution à la banque (Ecobank/Coris) sur le compte du Trésor ou du projet.\n"
                "4. Garder le reçu. La livraison se fait via le magasin de la coopérative."
            )
        
        elif "foncier" in program_type.lower() or "irrigation" in program_type.lower():
            return (
                "Procédure Équipement/Irrigation :\n"
                "1. Obtenir une APFR (Attestation de Possession Foncière Rurale) à la mairie.\n"
                "2. Déposer une demande manuscrite au Directeur Régional de l'Agriculture.\n"
                "3. Attendre la commission de sélection."
            )
            
        return "Rapprochez-vous de votre ZAT (Zone d'Appui Technique) ou de la Maison du Paysan."
