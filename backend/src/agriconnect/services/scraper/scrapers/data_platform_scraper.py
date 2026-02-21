"""
DataPlatformScraper - Exploration de plateformes de données statistiques et catalogues.

Plateformes supportées:
1. microdata.insd.bf: Catalogue de microdonnées INSD Burkina Faso
2. FEWS NET Data Book: Livre de données sur la sécurité alimentaire
3. FAO Agricultural Survey Data: Données d'enquêtes agricoles
4. Data.gov Burkina Faso: Données ouvertes gouvernementales

Stratégies:
- Exploration de catalogues avec pagination
- Extraction de métadonnées de datasets (titre, description, format, liens)
- Téléchargement de fichiers de données (CSV, Excel, JSON)
- Scraping de pages de documentation de données
"""

import requests
from bs4 import BeautifulSoup
import re
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from urllib.parse import urljoin, urlparse
import hashlib

logger = logging.getLogger("DataPlatformScraper")


class DataPlatformScraper:
    """
    Scraper pour plateformes de données et catalogues statistiques.
    """

    def __init__(self, output_dir: str = "backend/sources/raw_data/data_platforms"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.datasets_dir = self.output_dir / "datasets"
        self.datasets_dir.mkdir(exist_ok=True)
        
        self.metadata_dir = self.output_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/json,*/*'
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def identify_platform(self, url: str) -> str:
        """Identifie le type de plateforme à partir de l'URL."""
        url_lower = url.lower()
        
        if 'microdata.insd.bf' in url_lower:
            return 'insd_microdata'
        elif 'fews.net' in url_lower and 'data' in url_lower:
            return 'fews_data'
        elif 'fao.org' in url_lower and ('data' in url_lower or 'survey' in url_lower):
            return 'fao_data'
        elif 'data.gov' in url_lower:
            return 'data_gov'
        else:
            return 'generic'

    def scrape_insd_microdata(self, url: str) -> Dict[str, Any]:
        """
        Scrape le catalogue INSD de microdonnées.
        """
        result = {
            'status': 'failed',
            'url': url,
            'platform': 'insd_microdata'
        }
        
        try:
            response = self.session.get(url, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extraction du titre de la page/dataset
            title = soup.find('h1') or soup.find('h2')
            result['title'] = title.get_text(strip=True) if title else "Dataset INSD"
            
            # Chercher les métadonnées de dataset
            metadata_sections = soup.find_all(['div', 'section'], class_=re.compile('metadata|overview', re.I))
            
            dataset_info = {}
            for section in metadata_sections:
                # Chercher les paires label: valeur
                labels = section.find_all(['dt', 'strong', 'label'])
                for label in labels:
                    label_text = label.get_text(strip=True).lower()
                    value_elem = label.find_next_sibling()
                    if value_elem:
                        value = value_elem.get_text(strip=True)
                        if 'description' in label_text:
                            dataset_info['description'] = value
                        elif 'producer' in label_text or 'producteur' in label_text:
                            dataset_info['producer'] = value
                        elif 'year' in label_text or 'année' in label_text:
                            dataset_info['year'] = value
                        elif 'coverage' in label_text or 'couverture' in label_text:
                            dataset_info['coverage'] = value
            
            result.update(dataset_info)
            
            # Chercher les liens de téléchargement
            download_links = []
            for a in soup.find_all('a', href=True):
                href = a['href']
                text = a.get_text(strip=True).lower()
                
                # Identifier les liens de téléchargement
                if any(word in text for word in ['download', 'télécharger', 'export']) or \
                   any(ext in href.lower() for ext in ['.csv', '.xlsx', '.xls', '.json', '.xml', '.dta', '.sav']):
                    full_url = urljoin(url, href)
                    download_links.append({
                        'url': full_url,
                        'label': a.get_text(strip=True),
                        'format': self._guess_file_format(href)
                    })
            
            result['download_links'] = download_links
            
            # Extraire le contenu principal
            main_content = soup.find('main') or soup.find('div', class_=re.compile('content|main', re.I))
            if main_content:
                for tag in main_content(['script', 'style', 'nav', 'aside']):
                    tag.decompose()
                result['content'] = main_content.get_text(separator='\n', strip=True)
            
            result['status'] = 'success'
            logger.info(f"✅ INSD dataset scraped: {result['title']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur scraping INSD {url}: {e}")
            result['error'] = str(e)
            return result

    def scrape_fews_data(self, url: str) -> Dict[str, Any]:
        """
        Scrape FEWS NET Data Book et données de sécurité alimentaire.
        """
        result = {
            'status': 'failed',
            'url': url,
            'platform': 'fews_data'
        }
        
        try:
            response = self.session.get(url, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Titre
            title = soup.find('h1') or soup.find('title')
            result['title'] = title.get_text(strip=True) if title else "FEWS Data"
            
            # Chercher les tableaux de données
            tables = soup.find_all('table')
            result['tables_count'] = len(tables)
            
            # Chercher les liens de téléchargement de données
            download_links = []
            for a in soup.find_all('a', href=True):
                href = a['href']
                if any(ext in href.lower() for ext in ['.csv', '.xlsx', '.json', '.zip']):
                    download_links.append({
                        'url': urljoin(url, href),
                        'label': a.get_text(strip=True)
                    })
            
            result['download_links'] = download_links
            
            # Contenu principal
            content = soup.get_text(separator='\n', strip=True)
            result['content'] = content
            result['status'] = 'success'
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur scraping FEWS Data {url}: {e}")
            result['error'] = str(e)
            return result

    def scrape_generic_data_page(self, url: str) -> Dict[str, Any]:
        """
        Scraping générique pour plateformes de données non spécifiques.
        """
        result = {
            'status': 'failed',
            'url': url,
            'platform': 'generic'
        }
        
        try:
            response = self.session.get(url, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Titre
            title = soup.find('h1') or soup.find('title')
            result['title'] = title.get_text(strip=True) if title else "Data Platform"
            
            # Chercher les métadonnées structurées (JSON-LD, DCAT, etc.)
            for script in soup.find_all('script', {'type': 'application/ld+json'}):
                try:
                    data = json.loads(script.string)
                    if isinstance(data, dict) and data.get('@type') == 'Dataset':
                        result['structured_metadata'] = data
                        break
                except:
                    pass
            
            # Chercher tous les liens de fichiers de données
            data_links = []
            data_extensions = ['.csv', '.xlsx', '.xls', '.json', '.xml', '.geojson', 
                             '.zip', '.dta', '.sav', '.rds']
            
            for a in soup.find_all('a', href=True):
                href = a['href']
                if any(ext in href.lower() for ext in data_extensions):
                    data_links.append({
                        'url': urljoin(url, href),
                        'text': a.get_text(strip=True),
                        'format': self._guess_file_format(href)
                    })
            
            result['data_links'] = data_links
            
            # Description
            desc = soup.find('meta', {'name': 'description'})
            if desc:
                result['description'] = desc.get('content')
            
            # Contenu
            main = soup.find('main') or soup.find('article')
            if main:
                for tag in main(['script', 'style']):
                    tag.decompose()
                result['content'] = main.get_text(separator='\n', strip=True)
            else:
                result['content'] = soup.get_text(separator='\n', strip=True)[:5000]
            
            result['status'] = 'success'
            return result
            
        except Exception as e:
            logger.error(f"Erreur scraping generic data {url}: {e}")
            result['error'] = str(e)
            return result

    def _guess_file_format(self, url: str) -> str:
        """Devine le format de fichier à partir de l'URL."""
        url_lower = url.lower()
        
        formats = {
            '.csv': 'CSV',
            '.xlsx': 'Excel',
            '.xls': 'Excel',
            '.json': 'JSON',
            '.xml': 'XML',
            '.zip': 'ZIP',
            '.geojson': 'GeoJSON',
            '.dta': 'Stata',
            '.sav': 'SPSS',
            '.rds': 'R Data'
        }
        
        for ext, format_name in formats.items():
            if ext in url_lower:
                return format_name
        
        return 'Unknown'

    def download_dataset(self, url: str, filename: Optional[str] = None) -> Optional[Path]:
        """
        Télécharge un fichier de données.
        """
        try:
            response = self.session.get(url, timeout=60, stream=True)
            response.raise_for_status()
            
            if not filename:
                # Extraire depuis Content-Disposition ou URL
                content_disp = response.headers.get('Content-Disposition', '')
                filename_match = re.search(r'filename[^;=\n]*=([\'"]?)(.+?)\1', content_disp)
                
                if filename_match:
                    filename = filename_match.group(2)
                else:
                    url_filename = url.split('/')[-1].split('?')[0]
                    filename = url_filename if url_filename else f"dataset_{hashlib.md5(url.encode()).hexdigest()[:8]}.dat"
            
            # Nettoyer le nom de fichier
            filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
            filepath = self.datasets_dir / filename
            
            # Téléchargement
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"✅ Dataset téléchargé: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"Erreur téléchargement dataset {url}: {e}")
            return None

    def scrape(self, url: str) -> Dict[str, Any]:
        """
        Point d'entrée principal - détecte la plateforme et route.
        """
        platform = self.identify_platform(url)
        logger.info(f"Plateforme détectée: {platform} pour {url}")
        
        if platform == 'insd_microdata':
            result = self.scrape_insd_microdata(url)
        elif platform == 'fews_data':
            result = self.scrape_fews_data(url)
        else:
            result = self.scrape_generic_data_page(url)
        
        # Sauvegarder les métadonnées
        if result.get('status') == 'success':
            url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
            meta_filename = f"data_platform_{url_hash}.json"
            meta_path = self.metadata_dir / meta_filename
            
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            result['metadata_file'] = str(meta_path)
        
        return result


if __name__ == "__main__":
    # Test
    scraper = DataPlatformScraper()
    
    test_urls = [
        "https://microdata.insd.bf/index.php/catalog/83",
        "https://help.fews.net/fde/v1/burkina-faso-data-book"
    ]
    
    for url in test_urls:
        print(f"\n{'='*60}")
        result = scraper.scrape(url)
        print(f"Status: {result.get('status')}")
        print(f"Titre: {result.get('title')}")
        print(f"Liens de données: {len(result.get('data_links', []))}")
