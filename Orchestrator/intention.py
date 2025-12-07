import logging
import requests
import json
from typing import Dict, List, Any
from langchain_core.prompts import PromptTemplate
# Importations Pydantic pour la structuration de la sortie
try:
    from pydantic import BaseModel, Field, ValidationError
except ImportError:
    # Fournir un mock ou un message d'erreur si Pydantic n'est pas installé

    logging.warning("Pydantic n'est pas installé. La classification structurée ne fonctionnera pas correctement.")
    

# Configuration des logs pour ce module
logger = logging.getLogger("IntentClassifier")

# Définition des intentions pour la classification
INTENTS = [
    "METEO",    # Météo, Climat, Pluie, Température
    "CROP",     # Culture, Semis, Récolte, Engrais (calendrier), Itinéraire technique
    "SOIL",     # Sol, Terre, Texture, Amendement, Conservation de l'eau
    "HEALTH",   # Santé de la culture, Maladies, Insectes, Traitement
    "SUBSIDY",  # Subvention, Aide financière, Fraude, Administratif
    "UNKNOWN"   # Non classifiable
]

# ==============================================================================
# 1. Schéma Pydantic pour la Sortie LLM
# ==============================================================================

class IntentOutput(BaseModel):
    """
    Schéma de sortie strict pour la classification d'intention.
    Le LLM doit générer un JSON qui respecte ce format.
    """
    intent: str = Field(..., description=f"L'intention principale de la requête utilisateur. Doit être une des valeurs suivantes: {', '.join(INTENTS)}.")


# ==============================================================================
# 2. Classificateur d'Intention
# ==============================================================================

class IntentClassifier:
    """
    Détermine l'intention de l'utilisateur en appelant un modèle LLM via Ollama
    avec un schéma de sortie Pydantic strict.
    """
    
    def __init__(self, ollama_url: str = "http://localhost:11434", model_name: str = "mistral"):
        self.ollama_url = ollama_url
        self.model_name = model_name
        
        # Logique de secours par mots-clés
        self._fallback_keywords: Dict[str, List[str]] = {
            "SUBSIDY": ["arnaque", "argent", "subvention", "aide", "financement", "payer", "sms", "dossier"],
            "HEALTH": ["maladie", "insecte", "chenille", "feuille", "tache", "bête", "manger", "traitement", "remède", "ravageur"],
            "SOIL": ["sol", "terre", "sable", "argile", "zaï", "cordon", "restaurer", "pauvre", "ph", "texture"],
            "METEO": ["pluie", "pleuvoir", "vent", "chaud", "temps", "météo", "climat", "sécheresse", "averse"],
            "CROP": ["semer", "semis", "récolte", "engrais", "npk", "urée", "densité", "planter", "quand", "variété", "cultiver", "champ", "plan"]
        }


    def _fallback_keyword_classification(self, query: str) -> str:
        """Logique de classification par mots-clés de secours."""
        q = query.lower()
        for intent, keys in self._fallback_keywords.items():
            if any(k in q for k in keys):
                return intent
        return "UNKNOWN"

    def _call_llm_for_classification(self, query: str) -> str:
        """Appelle l'API Ollama en utilisant Pydantic pour structurer la réponse."""
        
        # 1. Générer le JSON Schema Pydantic
        try:
            json_schema = IntentOutput.model_json_schema()
            json_schema_str = json.dumps(json_schema, indent=2)
        except NameError:
            # Si Pydantic n'est pas disponible, on passe directement au fallback
            logger.error("Pydantic n'est pas disponible. Utilisation du fallback direct.")
            return self._fallback_keyword_classification(query)

        # 2. Construire l'instruction Système avec le schéma
        system_prompt = (
            "Tu es un classificateur d'intention expert pour l'agriculture. "
            "Votre unique tâche est de classer la requête utilisateur dans au moins une des catégories de la liste INTENTS. "
            "Tu vas lire la question de l'utilisateur et essayer de devinez lesquelles des agents doit ou doivent repondre a la question"
            "Répondez UNIQUEMENT avec un objet JSON. Ne fournissez aucune explication ni texte supplémentaire. "
            "Le format de sortie DOIT se conformer STRICTEMENT au schéma JSON suivant: "
            f"\n---\n{json_schema_str}\n---\n"
        )
        promptTemplate = """
Par exemple si l'utilisateur demande :Dois je planter du mil.
On sait qu'on doit avoir recours aux agents meteo ,soils,crop preferentiellement car ils permettent de donner une meilleur reponse.Maintenant ,repons a cette question
{query}
"""
        prompt = PromptTemplate(
            template=promptTemplate,
            input_variables=['query']
        )
        final_prompt = prompt.format(query=query)
        payload = {
            "model": self.model_name,
            "prompt": final_prompt,
            "system": system_prompt,
            "stream": False,
            "format": "json" # Signal natif à Ollama
        }

        try:
            ollama_api_url = f"{self.ollama_url.rstrip('/')}/api/generate"
            logger.info(f"Connecting to Ollama at {ollama_api_url}...")
            
            response = requests.post(ollama_api_url, json=payload, timeout=20)
            response.raise_for_status() 
            
            data: Dict[str, Any] = response.json()
            generated_text = data.get('response', '')
            
            # 3. Valider la sortie JSON avec Pydantic
            try:
                # model_validate_json gère la désérialisation et la validation en une étape
                validated_output = IntentOutput.model_validate_json(generated_text)
                intents = [i.upper() for i in validated_output.intents]
                return intents
                return intent
                
            except ValidationError as e:
                logger.error(f"Erreur de validation Pydantic (format JSON incorrect). Erreur: {e}")
                logger.debug(f"JSON brut reçu: {generated_text[:200]}...")
                # En cas d'échec de validation, on utilise le fallback
                return self._fallback_keyword_classification(query)
            
            except json.JSONDecodeError:
                logger.error(f"Réponse Ollama non-JSON malgré le format: 'json'. JSON brut: {generated_text[:200]}...")
                return self._fallback_keyword_classification(query)

        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur de connexion/timeout à Ollama à {self.ollama_url}: {e}. Utilisation du fallback.")
            return self._fallback_keyword_classification(query)
        except Exception as e:
            logger.error(f"Erreur inattendue lors de l'appel à l'API Ollama: {e}")
            return "UNKNOWN"


    def predict(self, query: str) -> str:
        """
        Méthode principale pour obtenir l'intention.
        """
        if not query:
            return "UNKNOWN"
            
        try:
            intent = self._call_llm_for_classification(query)
            
            if intent not in INTENTS:
                logger.warning(f"Intention LLM '{intent}' n'est pas dans la liste autorisée. Retour à UNKNOWN.")
                return "UNKNOWN"
                
            logger.info(f"Intention prédite: {intent}")
            return intent
            
        except Exception as e:
            logger.error(f"Erreur globale lors de la prédiction de l'intention: {e}")
            return "UNKNOWN"
        
if __name__ == "__main__":
    clf = IntentClassifier()
    queries = [
        "Quand mettre l'engrais ?",      # CROP
        "Il y a des chenilles dans mon champ",  # HEALTH
        "Mon sol est très sableux",      # SOIL
        "Va-t-il pleuvoir demain ?",     # METEO
        "J'ai reçu un SMS pour une aide",# SUBSIDY
        "Parle-moi de musique"           # UNKNOWN
    ]
    for q in queries:
        intent = clf.predict(q)
        print(f"Query: {q} -> Intent: {intent}")