# sona.py
import os
import time
import json
import logging
from typing import List, Dict, Optional, Set
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -------------------------
# Configuration
# -------------------------
START_URL = "https://sonagess.bf/?page_id=239"
TIMEOUT = 30
SLEEP = 0.4
MAX_DEPTH = 1
MAX_PAGES = 200
OUTPUT_DIR = "sonagess_pdfs"
CHECKPOINT_FILE = "sonagess_checkpoint.json"
DOWNLOAD_WORKERS = 4
USER_AGENT = "Mozilla/5.0 (compatible; SonagessScraper/1.0)"

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("sona")

# -------------------------
# Utilitaires
# -------------------------
def normalize(href: str, base: str) -> str:
    if not href:
        return ""
    href = href.strip()
    if href.startswith("//"):
        href = "https:" + href
    if not urlparse(href).scheme:
        href = urljoin(base, href)
    return href.split("#")[0]

def same_domain(base: str, url: str) -> bool:
    try:
        return urlparse(base).netloc == urlparse(url).netloc
    except Exception:
        return False

def safe_filename_from_url(url: str) -> str:
    name = os.path.basename(urlparse(url).path) or f"doc_{int(time.time())}.pdf"
    return name.replace("/", "_").replace("\\", "_")

# -------------------------
# Classe SonagessScraper
# -------------------------
class SonagessScraper:
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

        # session with robust retries
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504))
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update({"User-Agent": self.user_agent})

        if self.download:
            os.makedirs(self.output_dir, exist_ok=True)

        # state
        self._seen_printed: Set[str] = set()
        self._visited: Set[str] = set()
        self._found: Dict[str, Dict] = {}
        # load checkpoint if exists
        self._load_checkpoint()

    # -------------------------
    # Checkpoint
    # -------------------------
    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._found.update(data.get("found", {}))
                self._visited.update(data.get("visited", []))
                logger.info("Checkpoint chargé: %d pdfs, %d pages", len(self._found), len(self._visited))
            except Exception as e:
                logger.warning("Impossible de charger checkpoint: %s", e)

    def _save_checkpoint(self):
        try:
            with open(self.checkpoint_file, "w", encoding="utf-8") as f:
                json.dump({"found": self._found, "visited": list(self._visited)}, f, ensure_ascii=False, indent=2)
            logger.debug("Checkpoint sauvegardé")
        except Exception as e:
            logger.warning("Échec sauvegarde checkpoint: %s", e)

    # -------------------------
    # Requêtes et détection PDF
    # -------------------------
    def _fetch_html(self, url: str) -> Optional[str]:
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            logger.debug("Fetch HTML échoué %s : %s", url, e)
            return None

    def _head_is_pdf(self, url: str) -> bool:
        try:
            resp = self.session.head(url, allow_redirects=True, timeout=self.timeout)
            ctype = resp.headers.get("Content-Type", "")
            return "application/pdf" in ctype.lower()
        except Exception:
            return False

    def _likely_pdf(self, url: str) -> bool:
        # Si l'URL finit par .pdf, on considère PDF sans HEAD
        if url.lower().endswith(".pdf"):
            return True
        # Si l'URL a une extension non-pdf, on évite HEAD
        ext = os.path.splitext(urlparse(url).path)[1]
        if ext and ext.lower() != ".pdf":
            return False
        # sinon on fait HEAD (coûteux)
        return self._head_is_pdf(url)

    # -------------------------
    # Téléchargement concurrent
    # -------------------------
    def _download_pdf(self, url: str) -> Optional[str]:
        try:
            resp = self.session.get(url, stream=True, timeout=self.timeout)
            resp.raise_for_status()
            ctype = resp.headers.get("Content-Type", "")
            if "application/pdf" not in ctype.lower() and not url.lower().endswith(".pdf"):
                logger.warning("Contenu non PDF détecté pour %s (Content-Type=%s)", url, ctype)
                return None
            filename = safe_filename_from_url(url)
            out_path = os.path.join(self.output_dir, filename)
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logger.info("Téléchargé %s", out_path)
            return out_path
        except Exception as e:
            logger.warning("Échec téléchargement %s : %s", url, e)
            return None

    def _download_all_concurrent(self):
        to_download = [info for key, info in self._found.items() if not info.get("downloaded_path")]
        if not to_download:
            return
        logger.info("Démarrage téléchargement concurrent de %d fichiers", len(to_download))
        with ThreadPoolExecutor(max_workers=self.download_workers) as ex:
            futures = {ex.submit(self._download_pdf, info["pdf_url"]): key for key, info in self._found.items() if not info.get("downloaded_path")}
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    path = fut.result()
                    self._found[key]["downloaded_path"] = path
                except Exception as e:
                    logger.warning("Erreur téléchargement %s : %s", key, e)

    # -------------------------
    # Crawl BFS limité
    # -------------------------
    def collect_pdfs(self) -> List[Dict]:
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
                # sauvegarde checkpoint et continue
                if pages_visited % 10 == 0:
                    self._save_checkpoint()
                continue

            soup = BeautifulSoup(html, "html.parser")
            anchors = soup.find_all("a", href=True)
            iframes = soup.find_all("iframe", src=True)

            # process anchors
            for a in anchors:
                raw = a.get("href")
                norm = normalize(raw, page_url)
                if not norm:
                    continue

                # détection PDF optimisée
                if self._likely_pdf(norm):
                    key = norm.split("?")[0]
                    if key not in self._found:
                        self._found[key] = {
                            "pdf_url": norm,
                            "source_page": page_url,
                            "anchor_text": (a.get_text(strip=True) or ""),
                            "downloaded_path": None
                        }
                        # affichage unique et immédiat
                        if key not in self._seen_printed:
                            logger.info("[PDF trouvé] %s (source: %s)", norm, page_url)
                            self._seen_printed.add(key)
                    continue

                # enqueue internal pages if allowed
                if depth < self.max_depth and same_domain(self.start_url, norm):
                    if norm not in self._visited:
                        queue.append({"url": norm, "depth": depth + 1})

            # process iframes
            for iframe in iframes:
                src = normalize(iframe.get("src"), page_url)
                if not src:
                    continue
                if self._likely_pdf(src):
                    key = src.split("?")[0]
                    if key not in self._found:
                        self._found[key] = {
                            "pdf_url": src,
                            "source_page": page_url,
                            "anchor_text": "iframe",
                            "downloaded_path": None
                        }
                        if key not in self._seen_printed:
                            logger.info("[PDF trouvé iframe] %s (source: %s)", src, page_url)
                            self._seen_printed.add(key)

            # regex fallback pour URLs PDF brutes
            import re
            for m in re.finditer(r"(https?://[^\s'\"<>]+\.pdf)", html, re.I):
                pdf_url = m.group(1)
                key = pdf_url.split("?")[0]
                if key not in self._found:
                    self._found[key] = {
                        "pdf_url": pdf_url,
                        "source_page": page_url,
                        "anchor_text": "regex",
                        "downloaded_path": None
                    }
                    if key not in self._seen_printed:
                        logger.info("[PDF trouvé regex] %s (source: %s)", pdf_url, page_url)
                        self._seen_printed.add(key)

            # checkpoint périodique
            if pages_visited % 10 == 0:
                self._save_checkpoint()

        # sauvegarde finale checkpoint
        self._save_checkpoint()

        # téléchargement si demandé
        if self.download and self._found:
            self._download_all_concurrent()

        # construire résultat
        results = list(self._found.values())
        return results

    # -------------------------
    # Run et sauvegarde index
    # -------------------------
    def run(self, save_index: Optional[str] = "sonagess_index.json") -> List[Dict]:
        results = self.collect_pdfs()
        # résumé final
        total_found = len(results)
        total_downloaded = sum(1 for r in results if r.get("downloaded_path"))
        logger.info("Run terminé. PDFs trouvés: %d, téléchargés: %d", total_found, total_downloaded)
        print(f"Total PDFs trouvés: {total_found}  |  téléchargés: {total_downloaded}")

        if save_index:
            try:
                with open(save_index, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.info("Index sauvegardé: %s", save_index)
            except Exception as e:
                logger.warning("Impossible de sauvegarder l'index: %s", e)
        return results

# -------------------------
# Exécution directe
# -------------------------
if __name__ == "__main__":
    
    scraper = SonagessScraper(download=True, max_depth=1, max_pages=200, output_dir=OUTPUT_DIR)
    pdfs = scraper.run()
    for p in pdfs:
        print(p.get("pdf_url"), "->", p.get("downloaded_path"))