import logging
import re
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IntentClassifier")

# ======================================================================
# 1. SCHÉMAS DE SORTIE ET PROMPT
# ======================================================================

class IntentOutput(BaseModel):
    """Schéma de sortie strict pour la validation JSON."""
    intent: str = Field(description="L'intention : METEO, MARCHE, MIXTE, CROP, SOIL, HEALTH, SUBSIDY, REPORT, UNKNOWN")
    confidence: float = Field(description="Score de confiance entre 0 et 1.")
    reasoning: Optional[str] = Field(description="Brève explication du choix.")

class AgriScopeOutput(BaseModel):
    """Schéma pour vérifier si question est agricole."""
    is_agricultural: bool = Field(description="True si la question concerne l'agriculture/élevage au Burkina Faso")
    confidence: float = Field(description="Score de confiance entre 0 et 1")
    reason: str = Field(description="Explication courte: pourquoi agricole ou non")

INTENTS = ["METEO", "MARCHE", "MIXTE", "CROP", "SOIL", "HEALTH", "SUBSIDY", "REPORT", "UNKNOWN"]

SYSTEM_PROMPT = """
Tu es l'expert en classification d'intentions d'AgriConnect Burkina. 
Ton rôle est d'orienter l'agriculteur vers le bon service.

LISTE DES SERVICES :
- METEO : Prévisions de pluie, vents, chaleur, calendrier climatique.
- MARCHE : Prix des produits (maïs, riz), vente, achat, lieux de commerce.
- MIXTE : Si la question porte à la fois sur la METEO ET sur le MARCHE (ex: vendre avant la pluie).
- CROP : Conseils techniques (semis, écartement, NPK/Urée).
- SOIL : Récupération des terres (Zaï, demi-lunes), pH, compost.
- HEALTH : Insectes (chenille légionnaire), maladies des plantes, pesticides.
- SUBSIDY : Engrais subventionnés, prix officiels de l'État.
- REPORT : Signalement d'urgence (inondation, invasion de criquets).
- UNKNOWN : Salutations ou hors sujet.

CONSIGNE : Réponds UNIQUEMENT avec un JSON valide respectant le schéma demandé.
"""

AGRISCOPE_PROMPT = """
Tu es un filtre intelligent pour AgriConnect Burkina (assistant agricole).

Ton rôle : Déterminer si la question de l'utilisateur concerne l'AGRICULTURE/ÉLEVAGE au Burkina Faso.

ACCEPTER (is_agricultural=true) :
- Questions sur cultures (maïs, coton, sésame, mil, sorgho, riz, etc.)
- Météo/climat pour agriculture (pluie, sécheresse, températures)
- Prix de produits agricoles et marchés
- Santé des plantes (maladies, insectes, traitements)
- Sols et fertilité (compost, engrais, techniques)
- Élevage (bétail, volaille, santé animale)
- Subventions/programmes agricoles
- Organismes : INERA, SONAGESS, Chambres d'Agriculture, ONG agricoles
- Techniques : irrigation, zaï, demi-lunes, rotation
- Formations agricoles
- Marchés locaux (Koutiala, Bobo-Dioulasso, Ouagadougou)

REJETER (is_agricultural=false) :
- Histoire/guerres (Vietnam, conflits mondiaux)
- Musique/art (rap, artistes, culture non-agricole)
- Sport (football, basketball)
- Politique générale (élections, gouvernement NON lié à agriculture)
- Technologie non-agricole (smartphones, réseaux sociaux)
- Santé humaine (médecine, hôpitaux) SAUF si lié aux agriculteurs
- Éducation générale (écoles, universités) SAUF formations agricoles

EXEMPLES :
✅ "Prix du maïs à Koutiala" → AGRICOLE
✅ "Comment traiter chenille légionnaire" → AGRICOLE
✅ "Météo Bobo demain" → AGRICOLE (contexte agricole)
✅ "Contacter INERA" → AGRICOLE
❌ "Guerre du Vietnam" → NON AGRICOLE
❌ "Artistes rap burkinabé" → NON AGRICOLE
❌ "Match de foot ce soir" → NON AGRICOLE

CONSIGNE : JSON strict avec is_agricultural (bool), confidence (0-1), reason (string).
"""

# ======================================================================
# 2. CLASSE AGRISCOPE CHECKER (Vérifie si question est agricole)
# ======================================================================

class AgriScopeChecker:
    """
    Filtre intelligent basé LLM pour détecter les questions hors-sujet.
    Remplace la méthode keyword-based qui n'est pas scalable.
    """
    def __init__(self, llm_client: Optional[ChatGroq] = None):
        self.parser = JsonOutputParser(pydantic_object=AgriScopeOutput)
        
        if llm_client is None:
            try:
                from backend.src.agriconnect.services.llm_clients import client
                self.llm = client
            except ImportError:
                self.llm = None
                logger.warning("❌ Impossible d'importer le client LLM pour AgriScope.")
        else:
            self.llm = llm_client

        # Construction de la chaîne LCEL
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", AGRISCOPE_PROMPT),
            ("human", "Question de l'utilisateur : {query}\n\n{format_instructions}")
        ])

        if self.llm:
            self.chain = self.prompt | self.llm | self.parser
    
    def check_scope(self, query: str) -> Dict[str, Any]:
        """
        Vérifie si la question concerne l'agriculture.
        
        Returns:
            Dict avec is_agricultural (bool), confidence (float), reason (str)
        """
        if not query or len(query.strip()) < 3:
            return {
                "is_agricultural": False,
                "confidence": 1.0,
                "reason": "Question trop courte ou vide"
            }
        
        if not self.llm:
            logger.warning("⚠️ LLM indisponible, acceptation par défaut")
            return {
                "is_agricultural": True,
                "confidence": 0.5,
                "reason": "LLM non disponible, pas de filtre"
            }
        
        try:
            response = self.chain.invoke({
                "query": query,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            is_agri = response.get("is_agricultural", True)
            confidence = response.get("confidence", 0.5)
            reason = response.get("reason", "Classification automatique")
            
            if is_agri:
                logger.info(f"✅ Question AGRICOLE acceptée (conf: {confidence:.2f})")
            else:
                logger.warning(f"🚫 Question HORS-SUJET rejetée: {reason}")
            
            return {
                "is_agricultural": is_agri,
                "confidence": confidence,
                "reason": reason
            }
            
        except Exception as e:
            logger.warning(f"❌ Erreur AgriScope LLM: {e}")
            # En cas d'erreur, on accepte par défaut (meilleur UX que blocage)
            return {
                "is_agricultural": True,
                "confidence": 0.3,
                "reason": f"Erreur classification: {str(e)}"
            }

# ======================================================================
# 3. CLASSE INTENTCLASSIFIER
# ======================================================================

class IntentClassifier:
    def __init__(self, llm_client: Optional[ChatGroq] = None):
        """
        Initialise le classificateur.
        :param llm_client: Doit être une instance de ChatGroq (Runnable), pas le SDK Groq brut.
        """
        self.parser = JsonOutputParser(pydantic_object=IntentOutput)
        
        # Gestion du client (Correction de l'erreur LCEL)
        if llm_client is None:
            try:
                from backend.src.agriconnect.services.llm_clients import client # ChatGroq client
                self.llm = client
            except ImportError:
                self.llm = None
                logger.warning("❌ Impossible d'importer le client LLM.")
        else:
            self.llm = llm_client

        # Construction de la chaîne LCEL
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "Question de l'agriculteur : {query}\n\n{format_instructions}")
        ])

        if self.llm:
            # L'opérateur | fonctionne car self.llm est un Runnable (ChatGroq)
            self.chain = self.prompt | self.llm | self.parser
        
        # Mots-clés de secours pour le mode hors-ligne ou erreur LLM
        self._fallback_rules = {
            "MARCHE": r"(prix|marché|vendre|achat|commerce|cfa|fcfa|sac|coûte|grossiste)",
            "METEO": r"(pluie|vent|météo|temps|pleuvoir|ciel|chaleur|orage|saison)",
        }

    def predict(self, query: str) -> str:
        """
        Détection hybride : LLM d'abord, Regex en secours.
        """
        if not query or len(query.strip()) < 3:
            return "UNKNOWN"

        # 1. Tentative via LLM (Intelligence sémantique)
        if self.llm:
            try:
                response = self.chain.invoke({
                    "query": query,
                    "format_instructions": self.parser.get_format_instructions()
                })
                intent = response.get("intent", "UNKNOWN").upper()
                confidence = response.get("confidence", 0.0)

                if intent in INTENTS and confidence > 0.6:
                    return intent
            except Exception as e:
                logger.warning(f"⚠️ Erreur Chain LLM : {e}. Basculement sur Regex.")

        # 2. Pipeline de Secours (Regex - Gère spécifiquement le MIXTE)
        query_lower = query.lower()
        has_meteo = bool(re.search(self._fallback_rules["METEO"], query_lower))
        has_marche = bool(re.search(self._fallback_rules["MARCHE"], query_lower))

        if has_meteo and has_marche:
            return "MIXTE"
        if has_meteo:
            return "METEO"
        if has_marche:
            return "MARCHE"

        return "UNKNOWN"