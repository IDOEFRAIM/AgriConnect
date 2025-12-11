from flask import Flask, request
from Orchestrator.main_orchestrator import AgriculturalOrchestrator
app = Flask(__name__)

@app.route("/gen", methods=["POST"])
def generate():
    payload = request.get_json()  
    
    print(payload)             
    orchestrator = AgriculturalOrchestrator()
    main_graph = orchestrator.get_graph()
    
    print("\n>>> 🤖 DÉMARRAGE DU SYSTÈME AGRI-COPILOTE (BURKINA) <<<\n")
    result = main_graph.invoke({
            "user_id": "U12345",
            "zone_id": "Koudougou", 
            "user_query": payload.get('message','donne un conseil agricole'),
            "intent": "",
            "context_data": {},
            "final_response": "",
            "execution_trace": []
        })
    """
    # --- JEU DE TEST MULTI-AGENTS ---
    test_queries = [
        ("Meteo", "Est-ce qu'il va pleuvoir à Koudougou pour semer ?"),
        ("Health", "J'ai des chenilles dans le cornet de mon maïs, aidez-moi !"),
        ("Subsidy", "J'ai reçu un SMS pour payer 5000F pour le fonds ONU, c'est vrai ?"),
        ("Crop", "Quand est-ce que je dois mettre l'engrais NPK ?"),
        ("Soil", "Ma terre est sableuse et ne retient pas l'eau.")
    ]
    
    for category, query in test_queries:
        print(f"\n📢 USER ({category}): {query}")
        print("-" * 50)
        
        result = main_graph.invoke({
            "user_id": "U12345",
            "zone_id": "Koudougou", 
            "user_query": query,
            "intent": "",
            "context_data": {},
            "final_response": "",
            "execution_trace": []
        })
        
        print(f"🧠 ORCHESTRATOR TRACE: { ' -> '.join(result['execution_trace']) }")
        print("\n🤖 RÉPONSE FINALE :")
        print(result["final_response"])
        print("=" * 60)   
        """
    return {"status": "received", "data": result}

if __name__ == "__main__":
    app.run(debug=True,host='localhost',port=5000)