# sonagess_scraper_notify.py
import os
import time
import json
import logging
from typing import List, Dict, Optional, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("sonagess.notify")

DEFAULT_START_URL = "https://sonagess.bf/?page_id=239"
DEFAULT_TIMEOUT = 15
DEFAULT_SLEEP = 0.4
DEFAULT_MAX_DEPTH = 1
DEFAULT_MAX_PAGES = 200
DEFAULT_OUTPUT_DIR = "sonagess_pdfs"

class SonagessScraper:
    def __init__(self,
                 start_url: str = DEFAULT_START_URL,
                 timeout: int = DEFAULT_TIMEOUT,
                 sleep_between_requests: float = DEFAULT_SLEEP,
                 max_depth: int = DEFAULT_MAX_DEPTH,
                 max_pages: int = DEFAULT_MAX_PAGES,
                 download: bool = False,
                 output_dir: str = DEFAULT_OUTPUT_DIR,
                 user_agent: str = "Mozilla/5.0 (compatible; SonagessScraper/1.0)"):
        self.start_url = start_url
        self.timeout = timeout
        self.sleep = sleep_between_requests
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.download = download
        self.output_dir = output_dir
        self.user_agent = user_agent

        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.8, status_forcelist=(429, 500, 502, 503, 504))
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.session.headers.update({"User-Agent": self.user_agent})

        if self.download:
            os.makedirs(self.output_dir, exist_ok=True)

    def _normalize(self, href: str, base: str) -> str:
        if not href:
            return ""
        href = href.strip()
        if href.startswith("//"):
            href = "https:" + href
        if not urlparse(href).scheme:
            href = urljoin(base, href)
        return href.split("#")[0]

    def _is_same_domain(self, url: str) -> bool:
        try:
            return urlparse(url).netloc == urlparse(self.start_url).netloc
        except Exception:
            return False

    def _head_is_pdf(self, url: str) -> bool:
        try:
            resp = self.session.head(url, allow_redirects=True, timeout=self.timeout)
            ctype = resp.headers.get("Content-Type", "")
            return "application/pdf" in ctype.lower()
        except Exception:
            return False

    def _fetch_html(self, url: str) -> Optional[str]:
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            logger.debug("Fetch HTML failed %s : %s", url, e)
            return None

    def _download_pdf(self, url: str) -> Optional[str]:
        try:
            resp = self.session.get(url, stream=True, timeout=self.timeout)
            resp.raise_for_status()
            ctype = resp.headers.get("Content-Type", "")
            if "application/pdf" not in ctype.lower() and not url.lower().endswith(".pdf"):
                logger.warning("Le lien ne semble pas être un PDF (Content-Type=%s): %s", ctype, url)
                return None
            filename = os.path.basename(urlparse(url).path) or f"doc_{int(time.time())}.pdf"
            safe = filename.replace("/", "_").replace("\\", "_")
            out_path = os.path.join(self.output_dir, safe)
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logger.info("Téléchargé: %s", out_path)
            return out_path
        except Exception as e:
            logger.warning("Échec téléchargement %s : %s", url, e)
            return None

    def collect_pdfs(self) -> List[Dict]:
        queue: List[Dict] = [{"url": self.start_url, "depth": 0}]
        visited: Set[str] = set()
        found: Dict[str, Dict] = {}
        pages_visited = 0

        while queue and pages_visited < self.max_pages:
            node = queue.pop(0)
            page_url = node["url"]
            depth = node["depth"]
            if page_url in visited:
                continue
            visited.add(page_url)
            pages_visited += 1
            logger.info("Visite (%d/%d) depth=%d : %s", pages_visited, self.max_pages, depth, page_url)

            html = self._fetch_html(page_url)
            time.sleep(self.sleep)
            if not html:
                continue

            soup = BeautifulSoup(html, "html.parser")
            anchors = soup.find_all("a", href=True)
            iframes = soup.find_all("iframe", src=True)

            for a in anchors:
                raw = a.get("href")
                norm = self._normalize(raw, page_url)
                if not norm:
                    continue

                # direct .pdf link or HEAD says pdf
                is_pdf = norm.lower().endswith(".pdf") or self._head_is_pdf(norm)
                if is_pdf:
                    key = norm.split("?")[0]
                    if key not in found:
                        found[key] = {
                            "pdf_url": norm,
                            "source_page": page_url,
                            "anchor_text": (a.get_text(strip=True) or ""),
                            "downloaded_path": None
                        }
                        # Immediate feedback to user
                        msg = f"[PDF trouvé] {norm} (source: {page_url})"
                        logger.info(msg)
                        print(msg)
                    continue

                # enqueue internal pages
                if depth < self.max_depth and self._is_same_domain(norm):
                    if norm not in visited:
                        queue.append({"url": norm, "depth": depth + 1})

            for iframe in iframes:
                src = self._normalize(iframe.get("src"), page_url)
                if not src:
                    continue
                is_pdf = src.lower().endswith(".pdf") or self._head_is_pdf(src)
                if is_pdf:
                    key = src.split("?")[0]
                    if key not in found:
                        found[key] = {
                            "pdf_url": src,
                            "source_page": page_url,
                            "anchor_text": "iframe",
                            "downloaded_path": None
                        }
                        msg = f"[PDF trouvé iframe] {src} (source: {page_url})"
                        logger.info(msg)
                        print(msg)

            # regex fallback
            import re
            for m in re.finditer(r"(https?://[^\s'\"<>]+\.pdf)", html, re.I):
                pdf_url = m.group(1)
                key = pdf_url.split("?")[0]
                if key not in found:
                    found[key] = {
                        "pdf_url": pdf_url,
                        "source_page": page_url,
                        "anchor_text": "regex",
                        "downloaded_path": None
                    }
                    msg = f"[PDF trouvé regex] {pdf_url} (source: {page_url})"
                    logger.info(msg)
                    print(msg)

        results: List[Dict] = []
        for k, info in found.items():
            if self.download:
                path = self._download_pdf(info["pdf_url"])
                info["downloaded_path"] = path
                time.sleep(self.sleep)
            results.append(info)

        logger.info("Total PDFs trouvés: %d", len(results))
        print(f"Total PDFs trouvés: {len(results)}")
        return results

    def run(self, save_index: Optional[str] = None) -> List[Dict]:
        results = self.collect_pdfs()
        if save_index:
            try:
                with open(save_index, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.info("Index sauvegardé: %s", save_index)
            except Exception as e:
                logger.warning("Impossible de sauvegarder l'index: %s", e)
        return results

if __name__ == "__main__":
    scraper = SonagessScraper(download=True, max_depth=1, max_pages=200, output_dir="sonagess_pdfs")
    pdfs = scraper.run(save_index="sonagess_index.json")
    for p in pdfs:
        print(p["pdf_url"], "->", p["downloaded_path"])