import requests
import fitz  # PyMuPDF
import json
import csv
import os

#  Étape 1 : Télécharger le PDF
pdf_url = "https://meteoburkina.bf/documents/660/BADimport requests
import fitz  # PyMuPDF
import json
import csv
import os
import numpy as np
from PIL import Image

#  Étape 1 : Télécharger le PDF
pdf_url = "https://meteoburkina.bf/documents/660/BAD25111.pdf"
pdf_path = "agrodecadaire.pdf"

response = requests.get(pdf_url)
with open(pdf_path, "wb") as f:
    f.write(response.content)

# Créer un dossier pour les images extraites
image_dir = "images_agrodecadaire"
os.makedirs(image_dir, exist_ok=True)

#  Fonction de filtrage des images inutiles
def is_useless_image(pix):
    try:
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        arr = np.array(img)

        # 1. Taille minimale
        if pix.width < 100 or pix.height < 100:
            return True

        # 2. Couleur dominante uniforme
        avg_color = arr.mean(axis=(0, 1))
        if np.std(avg_color) < 5:
            return True

        # 3. Ratio extrême
        ratio = pix.width / pix.height
        if ratio > 5 or ratio < 0.2:
            return True

        return False
    except Exception:
        return True  # Si erreur, on ignore l'image

#  Étape 2 : Extraire texte et images utiles
doc = fitz.open(pdf_path)
data = []

for page_num, page in enumerate(doc):
    page_text = page.get_text().strip()
    images = []

    for img_index, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)

        if is_useless_image(pix):
            continue

        img_filename = f"{image_dir}/page{page_num+1}_img{img_index+1}.png"
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

#  Étape 3 : Sauvegarder en JSON
with open("agrodecadaire.json", "w", encoding="utf-8") as f_json:
    json.dump(data, f_json, ensure_ascii=False, indent=2)

#  Étape 4 : Sauvegarder en CSV (texte uniquement)
with open("agrodecadaire.csv", "w", encoding="utf-8", newline='') as f_csv:
    writer = csv.writer(f_csv)
    writer.writerow(["Page", "Texte"])

    for entry in data:
        writer.writerow([entry["page"], entry["text"]])

print(" Extraction terminée : agrodecadaire.json, agrodecadaire.csv et images utiles extraites.")25111.pdf"
pdf_path = "agrodecadaire.pdf"

response = requests.get(pdf_url)
with open(pdf_path, "wb") as f:
    f.write(response.content)

#  Étape 2 : Extraire le texte et les images
doc = fitz.open(pdf_path)
data = []

# Créer un dossier pour les images
os.makedirs("images_agrodecadaire", exist_ok=True)

for page_num, page in enumerate(doc):
    page_text = page.get_text().strip()
    images = []

    for img_index, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)
        img_filename = f"images_agrodecadaire/page{page_num+1}_img{img_index+1}.png"

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

# Étape 3 : Sauvegarder en JSON
with open("json/agrodecadaire.json", "w", encoding="utf-8") as f_json:
    json.dump(data, f_json, ensure_ascii=False, indent=2)

#  Étape 4 : Sauvegarder en CSV (texte uniquement)
with open("csv/agrodecadaire.csv", "w", encoding="utf-8", newline='') as f_csv:
    writer = csv.writer(f_csv)
    writer.writerow(["Page", "Texte"])

    for entry in data:
        writer.writerow([entry["page"], entry["text"]])

print(" Extraction terminée : agrodecadaire.json et agrodecadaire.csv créés.")
