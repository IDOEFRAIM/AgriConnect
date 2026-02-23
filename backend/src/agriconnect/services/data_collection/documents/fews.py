import requests
from bs4 import BeautifulSoup
import re
import os
import json
from datetime import datetime

BASE_RAW_DIR = "backend/sources/raw_data/fews_net/"

class FewsNetScraper:
    def __init__(self, country_slug="burkina-faso"):
        self.country_slug = country_slug
        self.base_dir = os.path.join(BASE_RAW_DIR, country_slug)
        os.makedirs(self.base_dir, exist_ok=True)
        self.url = f"https://fews.net/fr/west-africa/{self.country_slug}"

    def slugify(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s-]', '', text)
        return re.sub(r'[-\s]+', '_', text).strip('_')
    def _fetch(self, url, timeout=15):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            res = requests.get(url, headers=headers, timeout=timeout)
            if res.status_code != 200:
                return None, f"HTTP {res.status_code}"
            return res.text, None
        except Exception as exc:
            return None, str(exc)

    def _select_content_div(self, soup: BeautifulSoup):
        return soup.select_one('main') or \
               soup.select_one('article') or \
               soup.select_one('.node--type-report') or \
               soup.select_one('#main-content')

    def _clean_content_div(self, content_div):
        # Remove typical noisy tags and known selectors
        for tag in content_div.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()
        # remove common classes by CSS selector fallback
        for sel in ['.breadcrumb', '.region-sidebar-first']:
            for node in content_div.select(sel):
                node.decompose()

    def _extract_text_from_div(self, content_div):
        text = content_div.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text)
        return text

    def _is_valid_text(self, text, min_chars=200):
        return bool(text) and len(text) >= min_chars

    def extract_deep_content(self, url):
        # Use the regular URL (print versions often redirect to PDFs)
        print(f"    Fetching: {url}")
        html, err = self._fetch(url)
        if err:
            print(f"❌ Erreur lors du fetch {url}: {err}")
            return None

        soup = BeautifulSoup(html, 'html.parser')
        content_div = self._select_content_div(soup)
        if not content_div:
            print("❌ Aucun conteneur principal trouvé (main/article)")
            return None

        self._clean_content_div(content_div)
        text = self._extract_text_from_div(content_div)
        if not self._is_valid_text(text):
            print(f"⚠️ Contenu récupéré trop court ({len(text)} chars)")
            return None

        return text

    def run(self):
        print(f"🚀 Démarrage du moissonnage profond : {self.country_slug}")
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(self.url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        links_found = 0
        for link in soup.find_all('a', href=True):
            href = link['href']
            if not (f"/{self.country_slug}/" in href and any(x in href for x in ["mise-jour", "perspectives", "key-message"])):
                continue

            full_url = f"https://fews.net{href}" if href.startswith('/') else href
            title = link.get_text(strip=True)
            if len(title) <= 15:
                continue

            links_found += 1
            print(f"📄 Analyse de : {title[:50]}...")

            content = self.extract_deep_content(full_url)
            if not content or len(content) <= 500:
                print(f"⚠️ Contenu trop court ou vide pour : {title}")
                continue

            report_obj = {
                "metadata": {
                    "title": title,
                    "url": full_url,
                    "date_scraped": datetime.now().isoformat(),
                    "char_count": len(content)
                },
                "content": content
            }

            filename = f"{self.slugify(title)}.json"
            with open(os.path.join(self.base_dir, filename), "w", encoding="utf-8") as f:
                json.dump(report_obj, f, ensure_ascii=False, indent=4)
            print(f"✅ Sauvegardé ({len(content)} caractères)")

        if links_found == 0:
            print("❌ Aucun lien de rapport trouvé. Vérifiez l'URL source.")

if __name__ == "__main__":
    scraper = FewsNetScraper()
    scraper.run()