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

    def extract_deep_content(self, url):
        # On utilise l'URL normale car la version print peut être un PDF ou rediriger
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            print(f"    Fetching: {url}")
            res = requests.get(url, headers=headers, timeout=15)
            if res.status_code != 200:
                print(f"❌ Erreur HTTP {res.status_code} sur {url}")
                return None
            
            soup = BeautifulSoup(res.text, 'html.parser')
            
            # Nouvelle stratégie de sélecteurs pour le site FEWS NET 2024/2025
            # On cherche d'abord la balise main, puis article, puis des classes génériques
            content_div = soup.select_one('main') or \
                          soup.select_one('article') or \
                          soup.select_one('.node--type-report') or \
                          soup.select_one('#main-content')
            
            if content_div:
                # On nettoie les éléments inutiles (menus, scripts, etc.)
                for tag in content_div(['script', 'style', 'nav', 'footer', 'header', 'aside', '.breadcrumb', '.region-sidebar-first']):
                    tag.decompose()
                
                # On récupère le texte
                text = content_div.get_text(separator=' ', strip=True)
                
                # Nettoyage
                text = re.sub(r'\s+', ' ', text)
                
                # Vérification basique: si on a récupéré juste le titre ou menu
                if len(text) < 200:
                    print(f"⚠️ Contenu récupéré trop court ({len(text)} chars)")
                    return None
                    
                return text
            
            print("❌ Aucun conteneur principal trouvé (main/article)")
            return None
            
        except Exception as e:
            print(f"❌ Exception sur {url}: {e}")
            return None

    def run(self):
        print(f"🚀 Démarrage du moissonnage profond : {self.country_slug}")
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(self.url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        
        links_found = 0
        for link in soup.find_all('a', href=True):
            href = link['href']
            if f"/{self.country_slug}/" in href and any(x in href for x in ["mise-jour", "perspectives", "key-message"]):
                full_url = f"https://fews.net{href}" if href.startswith('/') else href
                title = link.get_text(strip=True)
                
                if len(title) > 15:
                    links_found += 1
                    print(f"📄 Analyse de : {title[:50]}...")
                    
                    content = self.extract_deep_content(full_url)
                    
                    if content and len(content) > 500: # On vérifie qu'on a du vrai texte
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
                    else:
                        print(f"⚠️ Contenu trop court ou vide pour : {title}")
        
        if links_found == 0:
            print("❌ Aucun lien de rapport trouvé. Vérifiez l'URL source.")

if __name__ == "__main__":
    scraper = FewsNetScraper()
    scraper.run()