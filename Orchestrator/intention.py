import logging
import json
import re
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

# LangChain Imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IntentClassifier")

# ======================================================================
# 1. SCHÉMAS ET PROMPTS
# ======================================================================

class IntentOutput(BaseModel):
    """Schéma de sortie strict pour assurer la compatibilité avec l'orchestrateur."""
    intent: str = Field(description="L'intention détectée parmi : METEO, CROP, SOIL, HEALTH, SUBSIDY, UNKNOWN")
    confidence: float = Field(description="Score de confiance entre 0 et 1.")
    reasoning: Optional[str] = Field(description="Brève explication du choix.")

INTENTS = ["METEO", "CROP", "SOIL", "HEALTH", "SUBSIDY", "UNKNOWN"]

SYSTEM_PROMPT = """
Tu es le module de classification d'intentions d'une plateforme d'intelligence agricole au Sahel (Burkina Faso).
Analyse la requête de l'agriculteur et classe-la rigoureusement.

LISTE DES INTENTIONS :
- METEO : Alertes pluie, vent, calendrier climatique.
- CROP : Conseils semis, doses engrais (NPK/Urée), espacements, récolte.
- SOIL : Qualité de terre, Zaï, compost, pH, érosion.
- HEALTH : Ravageurs (chenilles, criquets), maladies, symptômes, traitements bio.
- SUBSIDY : Prix subventionnés, aide de l'État (MAAH), warrantage, alertes arnaques/phishing.
- UNKNOWN : Salutations, remerciements, ou hors-sujet.

INSTRUCTIONS :
1. Réponds UNIQUEMENT au format JSON.
2. Si l'utilisateur parle de prix des engrais ou de SMS suspects, choisis SUBSIDY.
3. Si l'utilisateur décrit des tâches sur les feuilles ou des insectes, choisis HEALTH.
"""

# ======================================================================
# 2. CLASSE CLASSIFICATEUR
# ======================================================================

class IntentClassifier:
    def __init__(self, model_name: str = "mistral", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.parser = JsonOutputParser(pydantic_object=IntentOutput)
        
        # Initialisation du LLM
        try:
            self.llm = ChatOllama(
                model=model_name, 
                base_url=base_url, 
                temperature=0, 
                format="json"  # Force Ollama à sortir du JSON
            )
            logger.info(f"✅ IntentClassifier lié à Ollama ({model_name})")
        except Exception as e:
            self.llm = None
            logger.error(f"❌ Échec initialisation Ollama: {e}")

        # Prompt Template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "Question : {query}\n\n{format_instructions}")
        ])

        # Pipeline
        if self.llm:
            self.chain = self.prompt | self.llm | self.parser
        
        # Règles de secours (Regex) enrichies
        self._fallback_rules = {
            "SUBSIDY": r"(argent|payer|subvention|aide|arnaque|sms|financement|dossier|prix|fcfa|warrantage)",
            "HEALTH": r"(maladie|insecte|chenille|tache|traiter|manger|puceron|ravageur|soigner|pourri)",
            "SOIL": r"(sol|terre|ph|sable|argile|zaï|fertilité|caillou|compost|fumure)",
            "METEO": r"(pluie|vent|chaud|météo|temps|pleuvoir|climat|orage|inondation)",
            "CROP": r"(semer|semis|récolte|engrais|npk|urée|planter|hectare|rendement|maïs|coton|sorgho|niébé)"
        }

    def predict(self, query: str) -> str:
        """Détection hybride : LLM -> Mots-clés -> Unknown."""
        if not query or len(query.strip()) < 3:
            return "UNKNOWN"

        # 1. Tentative LLM
        if self.llm:
            try:
                response = self.chain.invoke({
                    "query": query,
                    "format_instructions": self.parser.get_format_instructions()
                })
                
                intent = response.get("intent", "").upper().strip()
                if intent in INTENTS:
                    logger.info(f"🤖 LLM Predict: {intent} (Conf: {response.get('confidence', 0)})")
                    return intent
            except Exception as e:
                logger.warning(f"⚠️ LLM indisponible, passage aux Regex. Erreur: {e}")

        # 2. Tentative Regex (Robuste si le LLM est lent ou déconnecté)
        query_lower = query.lower()
        for intent, pattern in self._fallback_rules.items():
            if re.search(pattern, query_lower):
                logger.info(f"🔍 Regex Predict: {intent}")
                return intent

        return "UNKNOWN"

# ======================================================================
# 3. TESTS
# ======================================================================

if __name__ == "__main__":
    classifier = IntentClassifier()
    
    test_cases = [
        "J'ai reçu un SMS me demandant 5000 FCFA pour mes engrais", # SUBSIDY
        "Mes feuilles de maïs sont mangées par des chenilles",      # HEALTH
        "Quand dois-je semer mon sorgho cette année ?",              # CROP
        "Est-ce qu'il va pleuvoir à Bobo-Dioulasso demain ?",       # METEO
        "Comment enrichir une terre trop sableuse avec du Zaï ?"     # SOIL
    ]

    print("\n--- 🧪 VALIDATION CLASSIFICATEUR ---")
    for q in test_cases:
        res = classifier.predict(q)
        print(f"Q: {q} \n=> INTENT: {res}\n")