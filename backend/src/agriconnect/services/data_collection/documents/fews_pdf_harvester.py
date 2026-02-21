import requests
from bs4 import BeautifulSoup
import os
import json
import re
from urllib.parse import urljoin
from datetime import datetime
import pdfplumber

# --- CONFIGURATION ---
BASE_DIR = "backend/sources/raw_data/fews_net"
PDF_DIR = os.path.join(BASE_DIR, "pdfs")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
TEXT_DIR = os.path.join(BASE_DIR, "texts")

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)


class AnamBulletinScraper:
    def __init__(self):
        self.base_url = "https://fews.net"
        self.search_url = (
            "https://fews.net/search?"
            "region%5B0%5D=520&"
            "report_type%5B0%5D=5&"
            "report_type%5B1%5D=6&"
            "report_type%5B2%5D=34&"
            "report_type%5B3%5D=7&"
            "report_type%5B4%5D=17&"
            "page_type=report"
        )
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/119.0.0.0 Safari/537.36"
            )
        }

    # --------------------------------------------------
    # 1. LISTE DES RAPPORTS
    # --------------------------------------------------
    def get_report_pages(self):
        """Récupère les liens PDF directs ET les pages de rapports depuis la page de recherche."""
        print("🔍 Scan de la page FEWS NET Burkina Faso…")
        try:
            res = requests.get(self.search_url, headers=self.headers, timeout=20)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, "html.parser")

            reports = []
            for a in soup.find_all("a", href=True):
                href = a["href"]
                text = a.get_text(strip=True)

                # 1) Lien direct vers un PDF (Download the report)
                if ".pdf" in href.lower():
                    full_url = urljoin(self.base_url, href)
                    reports.append({"title": text or "PDF Report", "url": full_url, "direct_pdf": True})
                    continue

                # 2) Lien vers une page de rapport en texte
                if "/burkina-faso/" in href and len(text) > 10:
                    clean_href = href.split("?")[0]
                    full_url = urljoin(self.base_url, clean_href)
                    reports.append({"title": text, "url": full_url, "direct_pdf": False})

            # Éviter doublons
            unique = []
            seen = set()
            for r in reports:
                if r["url"] not in seen:
                    seen.add(r["url"])
                    unique.append(r)

            return unique

        except Exception as e:
            print(f"❌ Erreur listing : {e}")
            return []

    # --------------------------------------------------
    # 2. TROUVER LE PDF VIA /print
    # --------------------------------------------------
    def find_pdf_on_print_page(self, report_url):
        print_url = report_url.rstrip("/") + "/print"

        try:
            res = requests.get(print_url, headers=self.headers, timeout=15)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, "html.parser")

            # Méthode principale : lien .pdf
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if ".pdf" in href.lower():
                    return urljoin(self.base_url, href)

            # Méthode de secours FEWS NET
            link = soup.select_one("div.field-name-field-document-file a")
            if link:
                return urljoin(self.base_url, link["href"])

            return None

        except Exception:
            return None

    # --------------------------------------------------
    # 3. TÉLÉCHARGEMENT DU PDF
    # --------------------------------------------------
    def download_file(self, url, title):
        safe_name = (
            re.sub(r"[^\w\s-]", "", title)
            .strip()
            .replace(" ", "_")
            .lower()[:70]
        )

        filepath = os.path.join(PDF_DIR, f"{safe_name}.pdf")

        if os.path.exists(filepath):
            return filepath, True

        try:
            r = requests.get(url, headers=self.headers, stream=True, timeout=30)
            r.raise_for_status()

            with open(filepath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            return filepath, False

        except Exception:
            return None, False

    # --------------------------------------------------
    # 4. EXTRACTION TEXTE DEPUIS PDF
    # --------------------------------------------------
    def extract_text_from_pdf(self, pdf_path):
        text = ""

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            return text.strip()

        except Exception:
            return ""

    # --------------------------------------------------
    # 5. PIPELINE COMPLET
    # --------------------------------------------------
    # --------------------------------------------------
    def run(self):
        print("🚀 DÉMARRAGE DU MOISSONNAGE FEWS NET — BURKINA FASO\n")

        reports = self.get_report_pages()
        print(f"📈 {len(reports)} rapports détectés.\n")

        for i, report in enumerate(reports, start=1):
            print(f"[{i}/{len(reports)}] 📄 Analyse : {report['title'][:80]}")

            pdf_url = None

            # --------------------------------------------------
            # 1️⃣ CAS : lien PDF direct (Download the report)
            # --------------------------------------------------
            if report.get("direct_pdf"):
                pdf_url = report["url"]

            else:
                # --------------------------------------------------
                # 2️⃣ CAS : tentative via /print
                # --------------------------------------------------
                pdf_url = self.find_pdf_on_print_page(report["url"])

                # --------------------------------------------------
                # 3️⃣ CAS : bouton "Download the report" sur la page
                # --------------------------------------------------
                if not pdf_url:
                    try:
                        r = requests.get(report["url"], headers=self.headers, timeout=20)
                        r.raise_for_status()
                        soup = BeautifulSoup(r.text, "html.parser")

                        for a in soup.find_all("a", href=True):
                            if "download" in a.get_text(strip=True).lower() and ".pdf" in a["href"].lower():
                                pdf_url = urljoin(self.base_url, a["href"])
                                break
                    except Exception:
                        pdf_url = None

            # --------------------------------------------------
            # PDF introuvable
            # --------------------------------------------------
            if not pdf_url:
                print("   ⚠️ PDF introuvable")
                continue

            # --------------------------------------------------
            # Téléchargement
            # --------------------------------------------------
            pdf_path, skipped = self.download_file(pdf_url, report["title"])

            if not pdf_path:
                print("   ❌ Échec téléchargement")
                continue

            status = "DÉJÀ PRÉSENT" if skipped else "TÉLÉCHARGÉ"
            print(f"   ✅ PDF {status}")

            # --------------------------------------------------
            # Extraction texte PDF
            # --------------------------------------------------
            pdf_text = self.extract_text_from_pdf(pdf_path)

            if len(pdf_text) < 200:
                print("   ⚠️ PDF très court (Key Message / Update) — conservé")
            else:
                print(f"   🧠 Texte extrait : {len(pdf_text)} caractères")

            # --------------------------------------------------
            # Sauvegarde texte
            # --------------------------------------------------
            text_path = os.path.join(
                TEXT_DIR, os.path.basename(pdf_path).replace(".pdf", ".txt")
            )
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(pdf_text)

            # --------------------------------------------------
            # Sauvegarde métadonnées
            # --------------------------------------------------
            meta_path = os.path.join(
                METADATA_DIR, os.path.basename(pdf_path).replace(".pdf", ".json")
            )

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "title": report["title"],
                        "source_page": report.get("source_page", report["url"]),
                        "pdf_url": pdf_url,
                        "pdf_path": pdf_path,
                        "text_path": text_path,
                        "extracted_text_length": len(pdf_text),
                        "harvested_at": datetime.now().isoformat(),
                    },
                    f,
                    indent=4,
                    ensure_ascii=False,
                )

            print("   💾 Métadonnées et texte sauvegardés\n")

    
# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    harvester = FewsPdfHarvester()
    harvester.run()
