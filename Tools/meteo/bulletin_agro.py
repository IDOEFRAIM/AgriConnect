import requests
from bs4 import BeautifulSoup
import io
import json
import csv
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
import logging
import os

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DocumentScraper")

try:
    from PyPDF2 import PdfReader
except ImportError:
    logger.error("La librairie PyPDF2 n'est pas installée. Veuillez l'installer: pip install pypdf2")
    PdfReader = None


class DocumentScraper:
    """
    Classe robuste pour scraper des liens de documents (HTML ou PDF)
    à partir d'une page d'index et extraire leur contenu.
    """

    def __init__(self, index_url: str, base_url: str = "https://meteoburkina.bf"):
        """
        Initialise le scraper.

        Args:
            index_url (str): L'URL de la page d'archive des bulletins (e.g., /bulletin-agrometeorologique-mensuel/).
            base_url (str): L'URL de base pour normaliser les liens relatifs.
        """
        self.base_url = base_url.rstrip("/")
        self.index_url = urljoin(self.base_url + "/", index_url)
        self.scraped_data: List[Dict[str, Any]] = []
        logger.info("Scraper initialisé. Index URL: %s", self.index_url)
        if not PdfReader:
             logger.warning("PyPDF2 est manquant. L'extraction de contenu PDF sera désactivée.")

    def _normalize_url(self, href: str) -> str:
        """Normalise un lien relatif ou absolu en lien absolu complet."""
        if not href.startswith("http"):
            return urljoin(self.base_url + "/", href)
        return href

    def _fetch_index_links(self) -> List[str]:
        """Récupère et filtre les liens des documents depuis la page d'index."""
        logger.info("Récupération de la page d'index: %s", self.index_url)
        try:
            response = requests.get(self.index_url, timeout=15)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error("Erreur lors de la récupération de la page d'index: %s", e)
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        
        # Filtres robustes : liens qui semblent pointer vers un bulletin ou un document
        keywords = ["bulletin", "situation", "document", "rapport"]
        links = []
        for a in soup.find_all("a", href=True):
            href = self._normalize_url(a["href"])
            href_lower = href.lower()
            
            # Condition de filtrage plus stricte pour éviter les liens de navigation
            is_relevant = False
            
            # 1. Lien vers un fichier PDF/DOC/etc.
            if href_lower.endswith((".pdf", ".doc", ".docx", ".xls", ".xlsx")):
                is_relevant = True
            
            # 2. Lien contenant un mot-clé pertinent dans le chemin d'URL
            if any(keyword in href_lower for keyword in keywords):
                # Éviter les liens de navigation de base ou les ancres internes
                if urlparse(href).netloc == urlparse(self.index_url).netloc and len(urlparse(href).path.split('/')) > 3:
                     is_relevant = True

            if is_relevant:
                # Filtrer les liens qui ne sont que des ancres (#)
                if href and not href.endswith("#") and href not in links:
                    links.append(href)

        logger.info("%d liens de documents uniques trouvés.", len(links))
        return links

    def _scrape_document_content(self, url: str) -> Dict[str, Any]:
        """Télécharge un document (HTML ou PDF) et extrait un résumé de son contenu."""
        logger.debug("Scraping du document: %s", url)
        try:
            response = requests.get(url, timeout=10, stream=True)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()

            if "application/pdf" in content_type and PdfReader:
                # Traitement PDF
                pdf_file = io.BytesIO(response.content)
                reader = PdfReader(pdf_file)
                text = ""
                # Extraire le texte des 3 premières pages
                for page in reader.pages[:3]:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
                
                title = url.split("/")[-1].replace(".pdf", "").replace("-", " ").strip()
                return {
                    "url": url,
                    "type": "pdf",
                    "title": title or "PDF sans titre",
                    "content": text.strip()
                }

            elif "text/html" in content_type:
                # Traitement HTML
                soup = BeautifulSoup(response.text, "html.parser")
                title = soup.title.string if soup.title else "HTML Sans titre"
                # Extraire le texte des 5 premiers paragraphes ou divs de contenu
                content_elements = soup.find_all(["p", "div", "article"], limit=5)
                paragraphs = [p.get_text(strip=True) for p in content_elements if p.get_text(strip=True)]
                content_summary = " ".join(paragraphs)
                
                return {
                    "url": url,
                    "type": "html",
                    "title": title,
                    "content": content_summary
                }

            else:
                # Type non géré
                return {
                    "url": url,
                    "type": "unknown",
                    "title": "N/A",
                    "content": f"Type non pris en charge : {content_type[:50]}"
                }

        except requests.exceptions.RequestException as e:
            logger.error("Erreur de requête pour %s: %s", url, e)
            return {"url": url, "type": "error", "title": "Erreur HTTP", "content": str(e)}
        except Exception as e:
            logger.error("Erreur d'extraction pour %s: %s", url, e)
            return {"url": url, "type": "error", "title": "Erreur d'extraction", "content": str(e)}

    def run_scrape(self) -> List[Dict[str, Any]]:
        """Exécute le processus complet de scraping."""
        links = self._fetch_index_links()
        if not links:
            logger.warning("Aucun lien à scraper. Arrêt de l'exécution.")
            return []

        all_docs = []
        for i, link in enumerate(links):
            logger.info("Progression: %d/%d - Scraping: %s", i + 1, len(links), link)
            doc_data = self._scrape_document_content(link)
            all_docs.append(doc_data)
        
        self.scraped_data = all_docs
        logger.info("Scraping terminé. %d documents traités.", len(all_docs))
        return all_docs

    def save_to_json(self, filepath: str, data: Optional[List[Dict[str, Any]]] = None) -> None:
        """Sauvegarde les données scrapées dans un fichier JSON."""
        data_to_save = data if data is not None else self.scraped_data
        if not data_to_save:
            logger.warning("Aucune donnée à sauvegarder en JSON.")
            return
            
        try:
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)
            logger.info("✅ Fichier JSON créé avec succès: %s (%d documents)", filepath, len(data_to_save))
        except Exception as e:
            logger.error("Erreur lors de la sauvegarde en JSON: %s", e)

    def save_to_csv(self, filepath: str, data: Optional[List[Dict[str, Any]]] = None) -> None:
        """Sauvegarde les données scrapées dans un fichier CSV."""
        data_to_save = data if data is not None else self.scraped_data
        if not data_to_save:
            logger.warning("Aucune donnée à sauvegarder en CSV.")
            return

        try:
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["URL", "Type", "Titre", "Résumé/Contenu"])
                for doc in data_to_save:
                    # Tronquer et nettoyer le contenu pour le CSV
                    content = doc.get("content", "").replace("\n", " ").strip()[:500]
                    writer.writerow([
                        doc.get("url", ""), 
                        doc.get("type", "N/A"), 
                        doc.get("title", "N/A"), 
                        content
                    ])
            logger.info("✅ Fichier CSV créé avec succès: %s (%d documents)", filepath, len(data_to_save))
        except Exception as e:
            logger.error("Erreur lors de la sauvegarde en CSV: %s", e)


if __name__ == "__main__":
    # --- Exemple d'utilisation de la classe ---

    # 1. Définir l'URL d'index ciblée (ici, les bulletins mensuels de votre exemple)
    INDEX_PAGE = "produits/bulletin-agrometeorologique-mensuel/"
    
    # 2. Créer une instance du scraper
    scraper = DocumentScraper(index_url=INDEX_PAGE)

    # 3. Exécuter le scraping
    results = scraper.run_scrape()

    if results:
        # 4. Sauvegarder les résultats
        scraper.save_to_json("json_output/bulletinAgro.json")
        scraper.save_to_csv("csv_output/bulletinAgro.csv")
    else:
        logger.info("Aucun résultat traité, aucune sauvegarde effectuée.")