# AgConnect

finetuner un model special pour chaque agent et un model effiscient pour l orchestrator


### 3. Comment initialiser et exécuter le service

Pour que cela fonctionne, vous devez instancier `ChatOllama` et le passer à `CropManagementService`.

```python
# Exemple d'initialisation et d'exécution dans votre code principal

# 1. Configurer ChatOllama (Assurez-vous qu'Ollama est en cours d'exécution)
# Remplacez 'mistral' par le nom de votre modèle léger téléchargé via Ollama
ollama_client = ChatOllama(model="mistral", temperature=0.1) 

# 2. Initialiser le Service Agent avec le client Ollama
crop_service = CropManagementService(llm_client=ollama_client)

# 3. Obtenir le Graph compilé
app = crop_service.get_graph()

# 4. Définir l'état initial
initial_state = {
    "zone_id": "field_001",
    "user_query": "Quand est-ce que je dois mettre de l'engrais sur mon maïs ?",
    "culture_config": {
        "crop_name": "Maïs",
        "sowing_date": "2025-09-01", # Exemple de date de semis
        "simulated_day": 70 # Pour les tests
    },
    "technical_advice": None,
    "final_response": "",
    "status": "INITIAL"
}

# 5. Exécuter le Graph
result = app.invoke(initial_state)

# Affichage du résultat final formaté par Mistral/Ollama
print("\n--- Réponse Finale Formatée par LLM ---")
print(result["final_response"])