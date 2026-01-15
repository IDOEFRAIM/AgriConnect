# api.py
import os
import sys

# Ajoute le dossier parent (AgConnect) au sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
    
import logging
from flask import Flask, request, jsonify

# Import de ton orchestrateur
from orchestrator import AgriculturalOrchestrator, OrchestratorState

# Initialisation Flask
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Initialisation de l'orchestrateur + graphe
orchestrator = AgriculturalOrchestrator()
graph = orchestrator.get_graph()

@app.route("/api/ask", methods=["POST"])
def ask():
    """
    Endpoint principal :
    {
        "user_id": "123",
        "zone_id": "Mopti",
        "query": "Mon sol est sableux, que faire ?"
    }
    """
    try:
        data = request.get_json()

        if not data or "query" not in data:
            return jsonify({"error": "Champ 'query' manquant"}), 400

        user_id = data.get("user_id", "anonymous")
        zone_id = data.get("zone_id", "ouaga")
        query = data["query"]

        # Construction de l'état initial
        state: OrchestratorState = {
            "user_id": user_id,
            "zone_id": zone_id,
            "user_query": query,
            "intent": "",
            "context_data": {},
            "final_response": "",
            "execution_trace": [],
            "meteo_data": None,
            "culture_config": None,
            "soil_config": None,
            "user_profile": None
        }

        # Exécution du graphe LangGraph
        result = graph.invoke(state)

        print(result)
        return jsonify({
            "response": result["final_response"],
            "intent": result["intent"],
            "trace": result["execution_trace"]
        })

    except Exception as e:
        logging.exception("Erreur API")
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "API Agricole opérationnelle ✅"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)