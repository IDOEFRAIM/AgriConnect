import os
import time
import json
import logging
import io
from typing import List, Dict, Optional, Set, Any
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pypdf import PdfReader

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SonagessScraper")


# -------------------------
# Configuration par défaut
# -------------------------
START_URL = "https://sonagess.bf/?page_id=239"
TIMEOUT = 30
SLEEP = 0.5
MAX_DEPTH = 1
MAX_PAGES = 50  # Réduit pour éviter de scanner tout le site par défaut
OUTPUT_DIR = "backend/sources/raw_data/sonagess_pdfs"
CHECKPOINT_FILE = "sonagess_checkpoint.json"
DOWNLOAD_WORKERS = 4
USER_AGENT = "Mozilla/5.0 (compatible; SonagessScraper/1.0)"

class SonagessScraper:
    """
    Scraper pour le site de la SONAGESS (Société Nationale de Gestion du Stock de Sécurité Alimentaire).
    Récupère les bulletins de prix et autres documents pertinents.
    """
    
    def __init__(self,
                 start_url: str = START_URL,
                 timeout: int = TIMEOUT,
                 sleep_between_requests: float = SLEEP,
                 max_depth: int = MAX_DEPTH,
                 max_pages: int = MAX_PAGES,
                 download: bool = True,
                 output_dir: str = OUTPUT_DIR,
                 checkpoint_file: str = CHECKPOINT_FILE,
                 download_workers: int = DOWNLOAD_WORKERS,
                 user_agent: str = USER_AGENT):
        
        self.start_url = start_url
        self.timeout = timeout
        self.sleep = sleep_between_requests
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.download = download
        self.output_dir = output_dir
        self.checkpoint_file = checkpoint_file
        self.download_workers = download_workers
        self.user_agent = user_agent

        # Session avec retries robustes
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504))
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update({"User-Agent": self.user_agent})

        if self.download:
            os.makedirs(self.output_dir, exist_ok=True)

        # État du scraper
        self._seen_printed: Set[str] = set()
        self._visited: Set[str] = set()
        self._found: Dict[str, Dict] = {}
        
        # Chargement du checkpoint si existant
        self._load_checkpoint()

    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._found.update(data.get("found", {}))
                self._visited.update(set(data.get("visited", [])))
                logger.info("Checkpoint chargé: %d pdfs trouvés, %d pages visitées", len(self._found), len(self._visited))
            except Exception as e:
                logger.warning("Impossible de charger le checkpoint: %s", e)

    def _save_checkpoint(self):
        try:
            with open(self.checkpoint_file, "w", encoding="utf-8") as f:
                json.dump({"found": self._found, "visited": list(self._visited)}, f, ensure_ascii=False, indent=2)
            logger.debug("Checkpoint sauvegardé")
        except Exception as e:
            logger.warning("Échec sauvegarde checkpoint: %s", e)

    def _normalize_url(self, href: str, base: str) -> str:
        if not href:
            return ""
        href = href.strip()
        if href.startswith("//"):
            href = "https:" + href
        if not urlparse(href).scheme:
            href = urljoin(base, href)
        return href.split("#")[0]

    def _is_same_domain(self, base: str, url: str) -> bool:
        try:
            return urlparse(base).netloc == urlparse(url).netloc
        except Exception:
            return False

    def _safe_filename_from_url(self, url: str) -> str:
        name = os.path.basename(urlparse(url).path) or f"doc_{int(time.time())}.pdf"
        # Nettoyage basique du nom de fichier
        return "".join([c for c in name if c.isalnum() or c in "._- "]).strip()

    def _fetch_html(self, url: str) -> Optional[str]:
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            logger.debug("Fetch HTML échoué %s : %s", url, e)
            return None

    def _is_likely_pdf(self, url: str) -> bool:
        url_lower = url.lower()
        if url_lower.endswith(".pdf"):
            return True
        # Vérification HEAD coûteuse, on l'évite si l'extension est explicite
        ext = os.path.splitext(urlparse(url).path)[1]
        if ext and ext.lower() != ".pdf":
            return False
        
        try:
            resp = self.session.head(url, allow_redirects=True, timeout=5)
            ctype = resp.headers.get("Content-Type", "")
            return "application/pdf" in ctype.lower()
        except Exception:
            return False

    def _download_pdf(self, url: str) -> Optional[str]:
        try:
            resp = self.session.get(url, stream=True, timeout=self.timeout)
            resp.raise_for_status()
            
            ctype = resp.headers.get("Content-Type", "")
            if "application/pdf" not in ctype.lower() and not url.lower().endswith(".pdf"):
                logger.warning("Contenu non PDF détecté pour %s (Content-Type=%s)", url, ctype)
                return None
                
            filename = self._safe_filename_from_url(url)
            out_path = os.path.join(self.output_dir, filename)
            
            # Éviter de retélécharger si le fichier existe et a une taille > 0
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                logger.info("Fichier déjà existant: %s", out_path)
                return out_path

            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logger.info("Téléchargé: %s", out_path)
            return out_path
        except Exception as e:
            logger.warning("Échec téléchargement %s : %s", url, e)
            return None

    def _extract_text_from_pdf(self, filepath: str) -> str:
        """Extrait le texte d'un fichier PDF local."""
        if not HAS_PYPDF2:
            return "Extraction impossible: PyPDF2 manquant."
        
        try:
            reader = PdfReader(filepath)
            text = ""
            # Limiter aux 5 premières pages pour les gros rapports
            for page in reader.pages[:5]:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text.strip()
        except Exception as e:
            logger.error("Erreur lecture PDF %s: %s", filepath, e)
            return "Erreur lors de la lecture du PDF."

    def _crawl(self):
        """Parcourt le site pour trouver des liens PDF."""
        queue: List[Dict] = [{"url": self.start_url, "depth": 0}]
        pages_visited = 0
        while queue and pages_visited < self.max_pages:
            node = queue.pop(0)
            page_url = node["url"]
            depth = node["depth"]

            if page_url in self._visited:
                continue
            self._visited.add(page_url)
            pages_visited += 1
            logger.info("Visite (%d/%d) depth=%d : %s", pages_visited, self.max_pages, depth, page_url)

            html = self._fetch_html(page_url)
            time.sleep(self.sleep)
            if not html:
                continue

            soup = BeautifulSoup(html, "html.parser")

            # Process anchors using a helper to keep logic small
            for a in soup.find_all("a", href=True):
                self._process_anchor_in_crawl(a, page_url, depth, queue)

            # Checkpoint périodique
            if pages_visited % 5 == 0:
                self._save_checkpoint()

    def _process_anchor_in_crawl(self, a_tag, page_url: str, depth: int, queue: List[Dict]):
        raw = a_tag.get("href")
        norm = self._normalize_url(raw, page_url)
        if not norm:
            return

        # If it's a PDF, register it
        if self._is_likely_pdf(norm):
            key = norm.split("?")[0]
            if key in self._found:
                return
            self._found[key] = {
                "url": norm,
                "source_page": page_url,
                "title": (a_tag.get_text(strip=True) or "Document SONAGESS"),
                "type": "pdf",
                "downloaded_path": None,
                "content": ""
            }
            if key not in self._seen_printed:
                logger.info("[PDF trouvé] %s", norm)
                self._seen_printed.add(key)
            return

        # Otherwise, consider queuing the internal link
        if depth < self.max_depth and self._is_same_domain(self.start_url, norm):
            if norm not in self._visited:
                queue.append({"url": norm, "depth": depth + 1})

    def _process_downloads_and_extraction(self):
        """Télécharge les PDFs trouvés et extrait leur contenu."""
        to_process = [k for k, v in self._found.items() if not v.get("downloaded_path") or not v.get("content")]

        if not to_process:
            logger.info("Tous les documents trouvés ont déjà été traités.")
            return

        logger.info("Traitement de %d documents...", len(to_process))

        # Schedule and run downloads via helper
        self._schedule_downloads(to_process)

        # Extract contents
        self._extract_contents_for_keys(to_process)

    def _schedule_downloads(self, keys: List[str]):
        to_download = [key for key in keys if not self._found[key].get("downloaded_path")]
        if not to_download:
            return

        with ThreadPoolExecutor(max_workers=self.download_workers) as ex:
            future_to_key = {ex.submit(self._download_pdf, self._found[key]["url"]): key for key in to_download}
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    path = future.result()
                    if path:
                        self._found[key]["downloaded_path"] = path
                except Exception as e:
                    logger.error("Erreur process download %s: %s", key, e)

    def _extract_contents_for_keys(self, keys: List[str]):
        for key in keys:
            info = self._found[key]
            path = info.get("downloaded_path")
            if not path or not os.path.exists(path) or info.get("content"):
                continue
            logger.info("Extraction texte: %s", path)
            content = self._extract_text_from_pdf(path)
            self._found[key]["content"] = content

    def run(self) -> Dict[str, Any]:
        """Exécute le scraping complet."""
        logger.info("Démarrage du scraping SONAGESS...")
        
        # 1. Crawl pour trouver les liens
        self._crawl()
        
        # 2. Téléchargement et extraction
        if self.download:
            self._process_downloads_and_extraction()
        
        # 3. Sauvegarde finale
        self._save_checkpoint()
        
        # Préparation des résultats formatés
        results = list(self._found.values())
        
        logger.info("Terminé. %d documents traités.", len(results))
        return {
            "status": "SUCCESS",
            "message": f"{len(results)} documents SONAGESS traités.",
            "results": results
        }

if __name__ == "__main__":
    scraper = SonagessScraper(max_pages=10)  # Limite pour test rapide
    result = scraper.run()
    print(json.dumps(result, indent=2, ensure_ascii=False))
