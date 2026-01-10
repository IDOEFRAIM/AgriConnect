import requests
import fitz  # PyMuPDF
import json
import csv
import os
import logging
import numpy as np
from PIL import Image
from typing import List, Dict, Any

class AgroDecadaireTool:
    """
    Outil pour télécharger et extraire des informations des bulletins agro-décadaires.
    """

    def __init__(self, output_dir: str = "agro_decadaire_output"):
        self.logger = logging.getLogger("AgroDecadaireTool")
        self.output_dir = output_dir
        self.image_dir = os.path.join(output_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)

    def process_bulletin(self, pdf_url: str, pdf_filename: str = "agrodecadaire.pdf") -> Dict[str, Any]:
        """
        Télécharge, extrait le texte et les images d'un bulletin PDF.
        """
        pdf_path = os.path.join(self.output_dir, pdf_filename)
        
        if not self._download_pdf(pdf_url, pdf_path):
            return {"error": "Failed to download PDF"}

        data = self._extract_content(pdf_path)
        self._save_results(data)
        
        return {"status": "success", "data_path": os.path.join(self.output_dir, "agrodecadaire.json")}

    def _download_pdf(self, url: str, path: str) -> bool:
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(path, "wb") as f:
                f.write(response.content)
            return True
        except Exception as e:
            self.logger.error(f"Error downloading PDF: {e}")
            return False

    def _is_useless_image(self, pix) -> bool:
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
            return True

    def _extract_content(self, pdf_path: str) -> List[Dict[str, Any]]:
        doc = fitz.open(pdf_path)
        data = []

        for page_num, page in enumerate(doc):
            page_text = page.get_text().strip()
            images = []

            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)

                if self._is_useless_image(pix):
                    continue

                img_filename = os.path.join(self.image_dir, f"page{page_num+1}_img{img_index+1}.png")
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
        return data

    def _save_results(self, data: List[Dict[str, Any]]):
        json_path = os.path.join(self.output_dir, "agrodecadaire.json")
        csv_path = os.path.join(self.output_dir, "agrodecadaire.csv")

        with open(json_path, "w", encoding="utf-8") as f_json:
            json.dump(data, f_json, ensure_ascii=False, indent=2)

        with open(csv_path, "w", encoding="utf-8", newline='') as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(["Page", "Texte"])
            for entry in data:
                writer.writerow([entry["page"], entry["text"]])
