import csv
import json
import os
from playwright.sync_api import sync_playwright

# --- Nettoyeurs ---
def nettoyer_nom_ville(nom):
    return nom.strip().lower().replace("  ", " ").replace("-", " ").title()

def nettoyer_nom_indicateur(nom):
    nom = nom.strip().lower()
    if "minimale" in nom:
        return "Temp. min"
    elif "maximale" in nom:
        return "Temp. max"
    elif "précipitation" in nom:
        return "Précipitations"
    return nom.title()

def serie_valide(data):
    return any(d is not None for d in data)

# --- Dossiers de sortie ---
os.makedirs("csv", exist_ok=True)
os.makedirs("json", exist_ok=True)

# --- Scraping ---
def ensure_checked(page, selector):
    cb = page.query_selector(selector)
    if cb and not cb.is_checked():
        cb.click()
        page.wait_for_timeout(500)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto("https://meteoburkina.bf/le-climat-de-nos-villes/")

    ville_elements = page.query_selector_all("#city_select option")
    all_data = {}

    for ville in ville_elements:
        ville_nom = nettoyer_nom_ville(ville.inner_text())
        ville_value = ville.get_attribute("value")

        page.select_option("#city_select", ville_value)
        page.wait_for_timeout(1000)

        ensure_checked(page, "text=Température minimale")
        ensure_checked(page, "text=Température maximale")
        ensure_checked(page, "text=Précipitation")

        chart_data = page.evaluate("""
            () => {
                const charts = Highcharts.charts;
                return charts.map(chart => {
                    if (!chart) return null;
                    return chart.series.map(s => ({
                        name: s.name,
                        data: s.data.map(point => point.y)
                    }));
                });
            }
        """)

        chart_data = [s for chart in chart_data if chart for s in chart]
        all_data[ville_nom] = chart_data

    browser.close()

# --- Structuration par mois ---
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

structured_data = {}
for ville, series_list in all_data.items():
    structured_data[ville] = {}
    for serie in series_list:
        name = nettoyer_nom_indicateur(serie["name"])
        data = serie["data"]
        if not serie_valide(data):
            continue
        structured_data[ville][name] = {
            months[i]: (data[i] if i < len(data) and data[i] is not None else "NA")
            for i in range(len(months))
        }

# --- Export CSV ---
csv_path = "csv/climat.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    header = ["Ville", "Indicateur"] + months
    writer.writerow(header)

    for ville, indicateurs in structured_data.items():
        for indicateur, valeurs in indicateurs.items():
            row = [ville, indicateur] + [valeurs[m] for m in months]
            writer.writerow(row)
            print("Écrit:", row)

print(f" Fichier CSV généré : {csv_path}")

# --- Export JSON ---
json_path = "json/climat.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(structured_data, f, indent=2, ensure_ascii=False)

print(f" Fichier JSON généré : {json_path}")

#python Agents/Meteo/climat.py
#python utils/metbd.py
