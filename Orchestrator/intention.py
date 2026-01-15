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
Tu es l'expert en classification d'intentions d'AgriConnect Burkina. 
Ton rôle est d'orienter l'agriculteur vers le bon service.

LISTE DES SERVICES :
- METEO : Prévisions de pluie, dates de début de saison, vents violents.
- CROP : Conseils de semis (écartement), fertilisation (NPK/Urée), rendements.
- SOIL : Techniques de récupération des terres (Zaï, demi-lunes), pH, compost.
- HEALTH : Identification des insectes (chenille légionnaire), maladies (jaunisse), traitements bio-pesticides.
- SUBSIDY : Prix officiels des intrants, subventions gouvernementales, alertes aux fraudes/arnaques SMS.
- UNKNOWN : Salutations ou questions non agricoles.

CONSIGNE : Réponds UNIQUEMENT avec un JSON valide.
"""

# ======================================================================
# 2. CLASSE CLASSIFICATEUR
# ======================================================================

class IntentClassifier:
    def __init__(self, model_name: str = "mistral", base_url: str = "http://localhost:11434"):
        self.parser = JsonOutputParser(pydantic_object=IntentOutput)
        
        try:
            self.llm = ChatOllama(
                model=model_name, 
                base_url=base_url, 
                temperature=0, 
                format="json"
            )
            logger.info(f"✅ Classificateur actif : {model_name}")
        except Exception:
            self.llm = None
            logger.error("❌ Ollama injoignable. Mode dégradé (Regex) activé.")

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "Question : {query}\n\n{format_instructions}")
        ])

        if self.llm:
            self.chain = self.prompt | self.llm | self.parser
        
        # Mots-clés de secours (Burkina-Specifiques)
        self._fallback_rules = {
            "SUBSIDY": r"(argent|payer|subvention|aide|arnaque|sms|prix|fcfa|warrantage|crédit)",
            "HEALTH": r"(maladie|insecte|chenille|manger|puceron|ravageur|soigner|pourri|poudre)",
            "SOIL": r"(sol|terre|ph|sable|argile|zaï|fertilité|caillou|compost|fumure|pierraille)",
            "METEO": r"(pluie|vent|chaud|météo|temps|pleuvoir|climat|orage|inondation)",
            "CROP": r"(semer|semis|récolte|engrais|npk|urée|planter|hectare|maïs|coton|sorgho)"
        }

    def predict(self, query: str) -> str:
        """Détection hybride sécurisée."""
        if not query or len(query.strip()) < 3:
            return "UNKNOWN"

        # 1. Pipeline LLM
        if self.llm:
            try:
                response = self.chain.invoke({
                    "query": query,
                    "format_instructions": self.parser.get_format_instructions()
                })
                intent = response.get("intent", "UNKNOWN").upper()
                if intent in INTENTS and response.get("confidence", 0) > 0.7:
                    return intent
            except Exception as e:
                logger.warning(f"Fallback Regex dû à : {e}")

        # 2. Pipeline de Secours (Regex)
        query_lower = query.lower()
        for intent, pattern in self._fallback_rules.items():
            if re.search(pattern, query_lower):
                return intent

        return "UNKNOWN"