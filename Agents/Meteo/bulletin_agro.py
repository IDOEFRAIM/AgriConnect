import requests
from bs4 import BeautifulSoup
import io
import json
import csv
from PyPDF2 import PdfReader

def scrape_dynamic_content(url: str) -> dict:
    try:
        response = requests.get(url, timeout=10)
        content_type = response.headers.get("Content-Type", "").lower()

        if "text/html" in content_type:
            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.title.string if soup.title else "Sans titre"
            paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
            return {
                "url": url,
                "type": "html",
                "title": title,
                "content": " ".join(paragraphs[:5])
            }

        elif "application/pdf" in content_type:
            pdf_file = io.BytesIO(response.content)
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages[:3]:
                if page.extract_text():
                    text += page.extract_text() + "\n"
            return {
                "url": url,
                "type": "pdf",
                "title": url.split("/")[-1],
                "content": text.strip()
            }

        else:
            return {
                "url": url,
                "type": "unknown",
                "title": "N/A",
                "content": f"Type non pris en charge : {content_type}"
            }

    except Exception as e:
        return {"url": url, "type": "error", "title": "Erreur", "content": str(e)}

# --- Scraper la page d’index ---
index_url = "https://meteoburkina.bf/produits/bulletin-agrometeorologique-mensuel/"
response = requests.get(index_url)
soup = BeautifulSoup(response.text, "html.parser")

# Extraire tous les liens vers les bulletins
links = []
for a in soup.find_all("a", href=True):
    href = a["href"]
    if "bulletin" in href or "situation" in href or href.endswith(".pdf"):
        if not href.startswith("http"):
            href = "https://meteoburkina.bf" + href
        links.append(href)

links = list(set(links))  # supprimer doublons

print("Liens trouvés:", links)

# --- Récupérer tous les documents ---
all_docs = [scrape_dynamic_content(link) for link in links]

# --- Sauvegarder en JSON ---
with open("json/bulletinAgro.json", "w", encoding="utf-8") as f:
    json.dump(all_docs, f, indent=2, ensure_ascii=False)

print("✅ bulletinAgro.json créé avec", len(all_docs), "documents")

# --- Sauvegarder en CSV ---
with open("csv/bulletinAgro.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["URL", "Type", "Titre", "Résumé/Contenu"])
    for doc in all_docs:
        # Tronquer le contenu pour éviter un CSV trop lourd
        content = doc["content"].replace("\n", " ")[:500]
        writer.writerow([doc["url"], doc["type"], doc["title"], content])

print("✅ bulletinAgro.csv créé avec", len(all_docs), "documents")
