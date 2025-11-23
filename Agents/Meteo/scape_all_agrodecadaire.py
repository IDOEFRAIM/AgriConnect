from playwright.sync_api import sync_playwright
import requests
import fitz  # PyMuPDF
import json
import csv
import os
import numpy as np
from PIL import Image

#  Dossiers de sortie
os.makedirs("bulletins_pdf", exist_ok=True)
os.makedirs("bulletins_json", exist_ok=True)
os.makedirs("bulletins_csv", exist_ok=True)
os.makedirs("images_agrodecadaire", exist_ok=True)

#  Nettoyeur de texte
def nettoyer_texte_brut(texte):
    fragments_inutiles = [
        "Nos services", "Nos produits", "Organisation", "DONNÉES ET OUTILS"
    ]
    for frag in fragments_inutiles:
        texte = texte.replace(frag, "")
    return texte.strip()

# Nettoyeur de doublons par texte
def nettoyer_doublons_par_texte(pages):
    vus = set()
    uniques = []
    for page in pages:
        texte = page["text"].strip()
        if texte not in vus:
            vus.add(texte)
            uniques.append(page)
    return uniques

#  Filtrage des images inutiles
def is_useless_image(pix):
    try:
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        arr = np.array(img)
        if pix.width < 100 or pix.height < 100:
            return True
        std_color = np.std(arr, axis=(0, 1))
        if np.max(std_color) < 5:
            return True
        ratio = pix.width / pix.height
        if ratio > 5 or ratio < 0.2:
            return True
        return False
    except Exception:
        return True

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False, slow_mo=200)
    page = browser.new_page()

    # Étape 1 : Accéder à la page principale
    page.goto("https://meteoburkina.bf/produits/bulletin-agrometeologique-decadaire/")
    page.wait_for_timeout(5000)

    # Extraire les liens vers les pages de bulletins
    subpage_links = page.evaluate("""
    () => Array.from(document.querySelectorAll("a"))
        .map(a => a.href)
        .filter(href => href.includes("/bulletin-agrometeologique-decadaire/bulletin-agrom"))
    """)
    print(f" {len(subpage_links)} pages de bulletin trouvées.")

    pdf_links = []

    # Étape 2 : Accéder à chaque sous-page et extraire le lien PDF
    for sub_url in subpage_links:
        try:
            page.goto(sub_url)
            page.wait_for_timeout(3000)

            pdf_url = page.evaluate("""
            () => {
                const links = Array.from(document.querySelectorAll("a"));
                const pdf = links.find(a => a.href.endsWith(".pdf"));
                return pdf ? pdf.href : null;
            }
            """)
            if pdf_url:
                pdf_links.append(pdf_url)
                print(f"📄 PDF trouvé : {pdf_url}")
            else:
                print(f" Aucun PDF trouvé sur : {sub_url}")
        except Exception as e:
            print(f" Erreur sur {sub_url} : {e}")

    # Étape 3 : Télécharger et extraire chaque PDF
    for url in pdf_links:
        filename = url.split("/")[-1]
        pdf_path = f"bulletins_pdf/{filename}"

        try:
            r = requests.get(url)
            r.raise_for_status()
            with open(pdf_path, "wb") as f:
                f.write(r.content)
            print(f" Téléchargé : {filename}")
        except Exception as e:
            print(f" Erreur de téléchargement : {e}")
            continue

        #  Extraction du contenu
        doc = fitz.open(pdf_path)
        data = []

        for page_num, page in enumerate(doc):
            page_text = nettoyer_texte_brut(page.get_text().strip())
            images = []

            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if is_useless_image(pix):
                    continue
                img_filename = f"images_agrodecadaire/{filename}_p{page_num+1}_i{img_index+1}.png"
                if pix.n < 5:
                    pix.save(img_filename)
                else:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                    pix.save(img_filename)
                images.append(img_filename)

            data.append({
                "page": page_num + 1,
                "text": page_text,
                "images": images
            })

        # Nettoyage des doublons
        data = nettoyer_doublons_par_texte(data)

        #  Sauvegarde JSON
        with open(f"bulletins_json/{filename.replace('.pdf', '.json')}", "w", encoding="utf-8") as f_json:
            json.dump(data, f_json, ensure_ascii=False, indent=2)

        #  Sauvegarde CSV
        with open(f"bulletins_csv/{filename.replace('.pdf', '.csv')}", "w", encoding="utf-8", newline='') as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(["Page", "Texte"])
            for entry in data:
                writer.writerow([entry["page"], entry["text"]])

        print(f" Extraction terminée pour : {filename}")

    browser.close()

print(" Tous les bulletins ont été récupérés, nettoyés, extraits et sauvegardés.")
