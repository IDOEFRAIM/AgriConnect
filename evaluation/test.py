import json, time,os
import requests

import json, time
import requests

"""
Ci joint 20 questions portant sur l'agriculture. Nous avons demande a GEMINI de 
nous generer 20 questions pour gagner en temps. Ces questions seront ensuite envoye au model pour qu'il reponde
"""

# Nous chargeons les 20 questions
with open("evaluation/questions.json") as f:
    questions = json.load(f)

results = []

""" On itere sur ls 20 questions et on envout au model qui nous renvoit sa reponse.
Nous pouvons ainsi evaluer ses reponses grace au temps d'execution
"""
for q in questions:
    start = time.time()
    response = requests.post(
        os.getenv("OLLAMA_MODEL_ID","http://localhost:11434/api/generate"),
        json={
            "model": "gemma:2b",
            "prompt": q,
            "stream": False
        }
    )
    end = time.time()

    reply = response.json()["response"]

    results.append({
        "question": q,
        "response": reply.strip(),
        "expected_answer": "",  # À remplir manuellement ou via un autre fichier
        "response_time": round(end - start, 2)
    })

# Sauvegarder les résultats
with open("evaluation/results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)