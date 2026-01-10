import requests
from bs4 import BeautifulSoup
import time
import json

url = "https://cesamcentrale.org/Code-de-bourse/"
headers = {
    "User-Agent": "Mozilla/5.0",
    "Content-Type": "application/x-www-form-urlencoded"
}

results = {}

for i in range(20250120, 20250140):
    payload = {
        "matricule_amci": str(i)
    }

    try:
        response = requests.post(url, data=payload, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find(id="table_matricule")

        if table:
            content = table.get_text(separator="\n", strip=True)
            print(content)
            results[str(i)] = content
            print(f"\n✅ Code {i} → Données extraites")
        else:
            results[str(i)] = None
            print(f"\n⚠️ Code {i} → Aucun contenu trouvé")

    except Exception as e:
        results[str(i)] = f"Erreur: {str(e)}"
        print(f"\n❌ Code {i} → Erreur : {e}")

    time.sleep(10)

# Sauvegarde dans bourse.json
with open("bourse.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n📁 Tous les résultats ont été sauvegardés dans bourse.json")