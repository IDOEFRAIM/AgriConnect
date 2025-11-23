# generator.py

import requests
import json

class UniversalGenerator:
    def __init__(self, model="llama3:8b", endpoint="http://localhost:11434"):
        """
        Initialise le générateur universel avec un modèle et un endpoint.
        :param model: Nom du modèle Ollama (ex: 'llama2', 'mistral', 'gemma')
        :param endpoint: URL de l'API Ollama (par défaut: http://localhost:11434)
        """
        self.model = model
        self.endpoint = endpoint.rstrip("/")  # sécurité: éviter double slash

    def generate(self, query: str, prompt: str) -> str:
        """
        Envoie un prompt à l'API Ollama et retourne la réponse générée.
        :param query: Requête utilisateur initiale (utile pour logs ou évaluation)
        :param prompt: Prompt final à envoyer au modèle
        :return: Réponse générée ou message d'erreur
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(f"{self.endpoint}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()

        except requests.exceptions.ConnectionError:
            return "❌ Impossible de se connecter à l'endpoint Ollama. Vérifie qu'il est bien lancé."

        except requests.exceptions.HTTPError as e:
            return f"❌ Erreur HTTP : {e.response.status_code} - {e.response.text}"

        except Exception as e:
            return f"❌ Erreur inattendue : {str(e)}"