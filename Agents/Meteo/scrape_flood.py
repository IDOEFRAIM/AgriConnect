# scrape_fanfar_forecasts.py
from playwright.sync_api import sync_playwright
import os, json, time, hashlib, urllib.parse

OUT = "flood_forecasts"
RAW = os.path.join(OUT, "raw")
os.makedirs(RAW, exist_ok=True)

def safe_name(s):
    return urllib.parse.quote_plus(s)[:200]

def save_text(url, text):
    h = hashlib.sha1(url.encode()).hexdigest()[:10]
    fname = f"resp_{safe_name(url)}_{h}.json"
    path = os.path.join(RAW, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False, slow_mo=150)
    page = browser.new_page()

    # Handler pour les réponses réseau
    def on_response(response):
        try:
            url = response.url
            ct = response.headers.get("content-type", "")
            # Filtrer uniquement les JSON liés aux prévisions
            if "application/json" in ct and any(k in url for k in ("forecast","timeseries","discharge","flow","flood")):
                text = response.text()
                path = save_text(url, text)
                print("Prévisions sauvegardées dans", path)

                # Extraction simple des données
                try:
                    data = json.loads(text)
                    if "timeseries" in data:
                        for ts in data["timeseries"]:
                            print("Station:", ts.get("station"))
                            for point in ts.get("values", []):
                                print("  Date:", point[0], "Valeur:", point[1])
                except Exception as e:
                    print("Erreur parsing JSON:", e)
        except Exception as e:
            print("Erreur on_response:", e)

    page.on("response", on_response)

    # Navigation vers FANFAR
    try:
        page.goto("https://fanfar.eu/fr/piv/", timeout=120000, wait_until="load")
    except Exception as e:
        print("Navigation warning:", e)

    print("Attente des requêtes réseau...")
    time.sleep(15)

    browser.close()
    print("Terminé. Données sauvegardées dans", RAW)
