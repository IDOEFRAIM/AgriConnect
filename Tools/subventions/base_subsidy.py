from dataclasses import dataclass
from typing import List, Dict, Optional

# 1. BASE DE DONNÉES DES PROGRAMMES (CONTEXTE BURKINA SAHÉLIEN)

@dataclass
class SubsidyProgram:
    id: str
    name: str
    provider: str
    category: str
    target_crops: List[str]
    target_zones: List[str]
    requirements: List[str]
    deadline_period: str
    is_active: bool


class SahelGrantDB:
    """
    Répertoire statique des opportunités de financement au Burkina Faso.
    Optimisé pour la production : lookup rapide, données immuables.
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
                "Disposer d'un titre foncier ou APFR",
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


# 2. MOTEUR D’ANALYSE DES SUBVENTIONS

class GrantExpertTool:
    """
    Outil d'analyse pour l'Agent Subvention.
    Production-ready : robuste, déterministe, adapté au contexte sahélien.
    """

    # 1. MATCHING DES OPPORTUNITÉS
    def find_opportunities(self, user_profile: Dict) -> List[Dict]:
        """
        Matche le profil du paysan avec les offres disponibles.
        Version renforcée, robuste et adaptée au contexte sahélien.
        """

        # Normalisation robuste des entrées
        user_crop = (user_profile.get("crop") or "").lower().strip()
        user_zone = (user_profile.get("zone") or "").strip()
        is_coop_member = bool(user_profile.get("is_coop_member", False))
        gender = (user_profile.get("gender") or "M").upper()

        matches = []

        for prog in SahelGrantDB.PROGRAMS:
            if not prog.is_active:
                continue

            # 1. MATCHING CULTURE (amélioré)
            crop_match = (
                user_crop in prog.target_crops
                or any(user_crop.startswith(t) for t in prog.target_crops)
                or user_crop == "toutes"
            )

            # 2. MATCHING ZONE
            zone_match = (
                "NATIONAL" in prog.target_zones
                or user_zone in prog.target_zones
            )

            # 3. MATCHING GENRE (programmes femmes)
            gender_match = True
            if "FEMME" in prog.id.upper() and gender != "F":
                gender_match = False

            # 4. SI MATCH → vérifier les conditions
            if crop_match and zone_match and gender_match:

                missing_reqs = []

                # Vérification coopérative
                if any("SCOOPS" in req.upper() for req in prog.requirements) and not is_coop_member:
                    missing_reqs.append("Adhésion Coopérative (SCOOPS)")

                # Score d’éligibilité (optionnel mais utile)
                score = 100
                if missing_reqs:
                    score -= 40
                if not is_coop_member and "INTRANTS" in prog.category:
                    score -= 20

                matches.append({
                    "program_name": prog.name,
                    "provider": prog.provider,
                    "category": prog.category,
                    "status": "ÉLIGIBLE" if not missing_reqs else "CONDITIONNEL",
                    "missing_documents": missing_reqs,
                    "deadline": prog.deadline_period,
                    "confidence": "élevée" if score >= 80 else "moyenne",
                    "eligibility_score": score
                })

        return matches
    
    # 2. DÉTECTION D’ARNAQUES
    def check_scam(self, message_content: str) -> Dict:
        """
        Détecte les faux messages de subvention (arnaques fréquentes au Sahel).
        """

        msg = (message_content or "").lower()
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

        return {
            "is_scam": False,
            "risk_level": "FAIBLE",
            "warning": "Semble légitime, mais vérifiez toujours auprès de la mairie ou du CRA."
        }

    # 3. GUIDE ADMINISTRATIF:
    def get_application_guide(self, program_type: str) -> str:
        """
        Procédure administrative adaptée au Burkina Faso.
        Version enrichie, réaliste et adaptée aux différents types de programmes.
        """

        p = (program_type or "").lower().strip()

        # 1. SUBVENTIONS INTRANTS (ENGRAIS, SEMENCES)
        if "intrant" in p or "engrais" in p or "semence" in p:
            return (
                "Procédure Subvention Engrais & Semences :\n"
                "1. Adhérer à une SCOOPS/GIE reconnue dans votre village.\n"
                "2. Vérifier que vous êtes recensé par l'agent d'agriculture (fiche de recensement).\n"
                "3. Vous inscrire sur la liste du CRA (Chambre Régionale d'Agriculture).\n"
                "4. Payer la caution à la banque (Ecobank, Coris, BOA) sur le compte du Trésor.\n"
                "5. Garder le reçu bancaire : il sera exigé lors de la distribution.\n"
                "6. Retirer les intrants au magasin de la coopérative à la date annoncée.\n\n"
                "Erreurs fréquentes :\n"
                "- Ne pas payer la caution à un numéro Orange Money personnel.\n"
                "- Ne jamais envoyer d'argent à un individu.\n"
                "- Vérifier que la coopérative est bien immatriculée."
            )

        # 2. IRRIGATION / ÉQUIPEMENT (POMPAGE SOLAIRE, MOTOPOMPES)
        if "irrigation" in p or "pompage" in p or "équipement" in p:
            return (
                "Procédure Équipement / Irrigation :\n"
                "1. Obtenir une APFR (Attestation de Possession Foncière Rurale) à la mairie.\n"
                "2. Faire une demande écrite adressée au Directeur Régional de l'Agriculture.\n"
                "3. Joindre : copie CNIB, plan de parcelle, photos du site, preuve d'appartenance à un groupement.\n"
                "4. Déposer le dossier au service technique (DRAAH ou ZAT).\n"
                "5. Attendre la visite technique pour vérifier l'aménageabilité.\n"
                "6. Participer à la réunion de sélection (commission régionale).\n"
                "7. Apporter l'apport personnel (10–20%) si le projet l'exige.\n\n"
                "Conseils pratiques :\n"
                "- Préparer un plan simple du périmètre.\n"
                "- Avoir un numéro Mobicash ou bancaire pour les paiements officiels."
            )

        # 3. FONCIER (APFR, TITRE FONCIER)
        if "foncier" in p or "terre" in p or "parcelle" in p:
            return (
                "Procédure Foncier Rural (APFR) :\n"
                "1. Se rendre à la mairie (service foncier rural).\n"
                "2. Faire une demande d'APFR avec : CNIB, plan de localisation, témoins du village.\n"
                "3. Participer à la visite de terrain avec le comité villageois.\n"
                "4. Attendre la validation du dossier par la mairie.\n"
                "5. Retirer l'APFR (document officiel reconnu par l'État).\n\n"
                "Note : L'APFR est obligatoire pour la plupart des projets irrigation/équipement."
            )

        # 4. PROGRAMMES FEMMES (FAARF, AGR, ONG)
        if "femme" in p or "femmes" in p or "faarf" in p:
            return (
                "Procédure Programmes Femmes (FAARF / AGR) :\n"
                "1. Constituer un groupement féminin reconnu (récépissé).\n"
                "2. Ouvrir un compte bancaire ou Mobicash au nom du groupement.\n"
                "3. Préparer : liste des membres, PV de réunion, plan d'activité.\n"
                "4. Déposer le dossier au FAARF ou à la mairie selon le programme.\n"
                "5. Participer à la formation obligatoire (gestion & AGR).\n"
                "6. Attendre la validation et le décaissement.\n\n"
                "Attention : Aucun agent FAARF ne demande de frais de dossier par téléphone."
            )

        # 5. PROGRAMMES CASH / AGR / ONG
        if "cash" in p or "agr" in p or "ong" in p or "projet" in p:
            return (
                "Procédure Programmes Cash / AGR / ONG :\n"
                "1. Vérifier l'annonce auprès de la mairie ou du CRA.\n"
                "2. Préparer : CNIB, plan d'activité, numéro Mobicash.\n"
                "3. Déposer le dossier au bureau du projet ou via la plateforme officielle.\n"
                "4. Participer à l'entretien de sélection.\n"
                "5. Suivre la formation obligatoire.\n\n"
                "Ne jamais payer des frais de dossier via Orange Money."
            )

        # 6. CAS GÉNÉRIQUE
        return (
            "Procédure Générale Subventions :\n"
            "1. Vérifier l'existence du programme auprès de la mairie ou du CRA.\n"
            "2. Préparer les documents : CNIB, plan d'activité, preuve d'appartenance à un groupement.\n"
            "3. Déposer le dossier au service technique (ZAT / DRAAH).\n"
            "4. Participer à la visite ou à l'entretien.\n"
            "5. Garder tous les reçus et documents officiels.\n\n"
            "⚠️ Règle d'or : Ne jamais envoyer d'argent à un numéro personnel."
        )