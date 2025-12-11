# orchestrator/intent_classifier.py

import logging
import json
from typing import Dict, List, Any, Optional
import requests

# NOUVELLES IMPORTATIONS LANGCHAIN/OLLAMA (CORRIGÉES)
from langchain_core.prompts import PromptTemplate  # Correct
from pydantic import BaseModel, Field, ValidationError # Correct pour la compatibilité
from langchain_community.llms import Ollama as OllamaLLM # Importation correcte du LLM Ollama
from langchain_core.output_parsers import JsonOutputParser 

logger = logging.getLogger("IntentClassifier")

INTENTS = [
    "METEO",
    "CROP",
    "SOIL",
    "HEALTH",
    "SUBSIDY",
    "UNKNOWN"
]

# Modèle de sortie structuré Pydantic (utilisant pydantic_v1 de LangChain)
class IntentOutput(BaseModel):
    """Sortie structurée pour la classification d'intention."""
    intent: str = Field(..., description=f"Doit être strictement une des valeurs: {', '.join(INTENTS)}.")

# Modèle de Prompt
CLASSIFICATION_PROMPT = """
Tu es un classificateur d'intention agricole. Ton rôle est de déterminer l'objectif principal de la question de l'utilisateur.

Les intentions possibles sont:
- METEO: Questions sur le temps, la pluie, le vent, la chaleur, le climat ou les conditions de traitement phytosanitaire.
- CROP: Questions sur les pratiques agronomiques de routine: semis, récolte, engrais, irrigation, densité, écartement.
- SOIL: Questions sur la terre, la texture (sableux, argileux), le pH, la fertilité, ou l'amélioration du sol.
- HEALTH: Questions sur les maladies, les insectes, les ravageurs, les symptômes ou les traitements phytosanitaires.
- SUBSIDY: Questions sur les aides, les subventions, le financement, les documents ou les arnaques.
- UNKNOWN: Toute autre question non liée à l'agriculture ou trop ambiguë.

QUESTION DE L'UTILISATEUR: {query}

Analyse la question et choisis l'intention la plus appropriée dans la liste {intent_list}.
{format_instructions}
"""


class IntentClassifier:
    def __init__(self, ollama_url="http://localhost:11434", model_name="mistral"):
        self.ollama_url = ollama_url
        self.model_name = model_name

        # Initialisation du client Ollama (LLM)
        self.llm = OllamaLLM(model=self.model_name, base_url=self.ollama_url)
        self.parser = JsonOutputParser(pydantic_object=IntentOutput)
        
        self._fallback_keywords = {
            "SUBSIDY": ["arnaque", "argent", "subvention", "aide", "financement", "payer", "sms", "dossier"],
            "HEALTH": ["maladie", "insecte", "chenille", "feuille", "tache", "ravageur"],
            "SOIL": ["sol", "terre", "sable", "argile", "zaï", "ph", "texture"],
            "METEO": ["pluie", "pleuvoir", "vent", "chaud", "temps", "météo", "climat", "traitement", "pulvériser"],
            "CROP": ["semer", "semis", "récolte", "engrais", "npk", "urée", "densité", "planter"]
        }
    
    # Méthode de classification par mot-clé (mécanisme de secours) (INCHANGÉE)
    def _fallback_keyword_classification(self, query: str) -> str:
        q = query.lower()
        for intent, keys in self._fallback_keywords.items():
            if any(k in q for k in keys):
                return intent
        return "UNKNOWN"
    
    # Classification structurée par Ollama
    def _ollama_classify(self, query: str) -> Optional[str]:
        """Appelle Ollama pour obtenir la classification en mode structuré."""
        try:
            # Création du Prompt final avec les instructions de formatage
            prompt = PromptTemplate(
                template=CLASSIFICATION_PROMPT,
                input_variables=["query"],
                partial_variables={
                    "format_instructions": self.parser.get_format_instructions(),
                    "intent_list": str(INTENTS)
                },
            )

            # Chaînage : Prompt | LLM | Parser
            # NOTE: L'opérateur de pipe '|' est la manière recommandée pour chaîner les runnables
            chain = prompt | self.llm | self.parser 
            
            # Exécution de la chaîne
            result = chain.invoke({"query": query})
            
            # Validation Pydantic et Nettoyage
            # Note: Si le parser JSON est utilisé, result est un dict, pas un objet Pydantic
            validated_output = IntentOutput(**result)
            
            # Vérification de l'intention dans la liste INTENTS
            intent = validated_output.intent.upper().strip()
            if intent in INTENTS:
                 return intent
            
            logger.warning(f"Sortie LLM invalide: {intent}. Utilisation du fallback.")
            return None 
            
        except requests.exceptions.ConnectionError:
            logger.error("Erreur de connexion à Ollama. Utilisation du fallback.")
            return None
        except Exception as e:
            logger.error(f"Erreur d'appel ou de parsing Ollama. Utilisation du fallback. Erreur: {e}")
            return None

    def predict(self, query: str) -> str:
        if not query:
            return "UNKNOWN"

        # --- TENTATIVE 1 : Classification Structurée par Ollama ---
        ollama_intent = self._ollama_classify(query)
        
        if ollama_intent:
            logger.info(f"Intention prédite (Ollama): {ollama_intent}")
            return ollama_intent

        # --- TENTATIVE 2 : Classification par Mots-Clés (Fallback) ---
        intent = self._fallback_keyword_classification(query)
        
        logger.warning(f"Intention prédite (Fallback): {intent}")
        return intent

# ==============================================================================
# 3. Exemple d'utilisation (pour tester le module)
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    try:
        classifier = IntentClassifier(model_name="mistral")
        
        test_queries = {
            "CROP": "Quand est la période optimale pour mettre de l'engrais NPK sur le maïs ?",
            "HEALTH": "J'ai des chenilles qui mangent les feuilles de mes tomates, c'est quoi ?",
            "METEO": "Est-ce que je peux pulvériser mon champ aujourd'hui avec le vent qu'il y a ?",
            "SUBSIDY": "Mon voisin me demande de l'argent pour un dossier d'aide, est-ce une arnaque ?",
            "SOIL": "Mon sol est très sableux et l'eau s'échappe, que dois-je faire ?",
            "UNKNOWN": "Quelle est la capitale de l'Australie ?"
        }

        print("\n--- TEST DE CLASSIFICATION STRUCTURÉE ---")
        
        
        for expected_intent, q in test_queries.items():
            print(f"\n[Q: {q}]")
            predicted = classifier.predict(q)
            print(f"-> PRÉDICTION : {predicted} (Attendu: {expected_intent})")
            assert predicted == expected_intent or predicted == "UNKNOWN", f"Échec pour {expected_intent}"
            
    except requests.exceptions.ConnectionError:
        print("\n!!! AVERTISSEMENT !!!")
        print("Ollama n'est pas démarré (ou URL/Port incorrect). Le test n'a pas pu exécuter la classification LLM.")
        print("La classification basculera uniquement sur le mécanisme de mots-clés.")
    except Exception as e:
        print(f"\n!!! ERREUR CRITIQUE !!! {e}")