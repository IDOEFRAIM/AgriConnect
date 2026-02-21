"""
FaoDoiResolver - Résolution et extraction de publications FAO via DOI.

Stratégie:
1. Résoudre le DOI vers l'URL canonique FAO
2. Scraper la page de publication pour extraire métadonnées
3. Localiser et télécharger le PDF principal
4. Extraire les métadonnées structurées (auteurs, date, résumé, etc.)
5. Fallback: Scraping HTML si PDF indisponible

Note: Les publications FAO suivent généralement un format standard avec:
- DOI: https://doi.org/10.4060/XXXXXX
- URL: https://www.fao.org/documents/card/en/c/XXXXXX
- PDF: https://www.fao.org/3/XXXXXX/XXXXXX.pdf
"""

import requests
from bs4 import BeautifulSoup
import re
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from urllib.parse import urljoin
from datetime import datetime

from .pdf_downloader import PdfDownloader

logger = logging.getLogger("FaoDoiResolver")


class FaoDoiResolver:
    """
    Résout les DOIs FAO et extrait le contenu complet des publications.
    """

    def __init__(self, output_dir: str = "backend/sources/raw_data/fao_publications"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.pdf_downloader = PdfDownloader(output_dir=str(self.output_dir / "pdfs"))
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def resolve_doi(self, doi_url: str) -> Optional[str]:
        """
        Résout un DOI vers l'URL canonique de la publication.
        """
        try:
            # Les DOIs redirigent automatiquement
            response = self.session.head(doi_url, allow_redirects=True, timeout=15)
            canonical_url = response.url
            logger.info(f"DOI résolu: {doi_url} -> {canonical_url}")
            return canonical_url
        except Exception as e:
            logger.error(f"Erreur résolution DOI {doi_url}: {e}")
            return None

    def extract_fao_code(self, url: str) -> Optional[str]:
        """
        Extrait le code FAO depuis une URL (ex: cb9479fr depuis .../cb9479fr).
        """
        patterns = [
            r'/([a-z]{2}\d{4,}[a-z]{2})',  # Pattern standard: cb9479fr
            r'/c/([A-Z0-9]+)',              # Pattern alternatif
            r'\.org/3/([a-z]{2}\d{4,}[a-z]{2})',  # Dans chemin /3/
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                return match.group(1).lower()
        
        return None

    def construct_pdf_url(self, fao_code: str) -> str:
        """
        Construit l'URL directe du PDF à partir du code FAO.
        """
        return f"https://www.fao.org/3/{fao_code}/{fao_code}.pdf"

    def scrape_publication_page(self, url: str) -> Dict[str, Any]:
        """
        Scrape la page de publication FAO pour extraire métadonnées et liens.
        """
        metadata = {
            'source_url': url,
            'scraped_at': datetime.now().isoformat()
        }
        
        try:
            response = self.session.get(url, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extraction du titre
            title = None
            title_selectors = [
                ('h1', {'class': 'title'}),
                ('h1', {}),
                ('meta', {'property': 'og:title'}),
                ('title', {})
            ]
            
            for tag, attrs in title_selectors:
                elem = soup.find(tag, attrs)
                if elem:
                    if tag == 'meta':
                        title = elem.get('content')
                    else:
                        title = elem.get_text(strip=True)
                    if title:
                        break
            
            metadata['title'] = title or "Publication FAO"
            
            # Extraction de l'abstract/résumé
            abstract = None
            abstract_selectors = [
                ('div', {'class': 'abstract'}),
                ('div', {'class': 'summary'}),
                ('meta', {'name': 'description'}),
                ('meta', {'property': 'og:description'})
            ]
            
            for tag, attrs in abstract_selectors:
                elem = soup.find(tag, attrs)
                if elem:
                    if tag == 'meta':
                        abstract = elem.get('content')
                    else:
                        abstract = elem.get_text(strip=True)
                    if abstract and len(abstract) > 50:
                        break
            
            metadata['abstract'] = abstract
            
            # Extraction des métadonnées Dublin Core / OpenGraph
            meta_tags = soup.find_all('meta')
            for meta in meta_tags:
                name = meta.get('name') or meta.get('property', '')
                content = meta.get('content')
                
                if content:
                    if 'author' in name.lower():
                        metadata['author'] = content
                    elif 'date' in name.lower() or 'published' in name.lower():
                        metadata['publication_date'] = content
                    elif 'language' in name.lower():
                        metadata['language'] = content
                    elif 'isbn' in name.lower():
                        metadata['isbn'] = content
            
            # Chercher les liens de téléchargement PDF
            pdf_links = []
            for a in soup.find_all('a', href=True):
                href = a['href']
                if href.endswith('.pdf') or 'download' in href.lower():
                    full_url = urljoin(url, href)
                    pdf_links.append(full_url)
            
            metadata['pdf_links'] = pdf_links
            
            # Extraction du contenu HTML principal (fallback)
            content_div = soup.find('div', {'class': re.compile('content|main|article', re.I)}) or \
                         soup.find('main') or \
                         soup.find('article')
            
            if content_div:
                # Nettoyer les éléments non pertinents
                for tag in content_div(['script', 'style', 'nav', 'footer', 'aside']):
                    tag.decompose()
                
                content_text = content_div.get_text(separator='\n', strip=True)
                metadata['html_content'] = content_text
            
            return metadata
            
        except Exception as e:
            logger.error(f"Erreur scraping page {url}: {e}")
            return metadata

    def process_fao_doi(self, doi_url: str) -> Dict[str, Any]:
        """
        Pipeline complet pour une publication FAO via DOI.
        """
        result = {
            'status': 'failed',
            'doi': doi_url,
            'source_type': 'fao_publication'
        }
        
        # 1. Résolution du DOI
        canonical_url = self.resolve_doi(doi_url)
        if not canonical_url:
            result['error'] = 'Échec résolution DOI'
            return result
        
        result['url'] = canonical_url
        
        # 2. Extraction du code FAO
        fao_code = self.extract_fao_code(canonical_url)
        if fao_code:
            result['fao_code'] = fao_code
            logger.info(f"Code FAO extrait: {fao_code}")
        
        # 3. Scraping de la page de publication
        page_metadata = self.scrape_publication_page(canonical_url)
        result.update(page_metadata)
        
        # 4. Tentative de téléchargement du PDF
        pdf_downloaded = False
        
        # Méthode 1: URL construite depuis le code FAO
        if fao_code:
            pdf_url = self.construct_pdf_url(fao_code)
            logger.info(f"Tentative téléchargement: {pdf_url}")
            
            try:
                pdf_result = self.pdf_downloader.process_pdf(pdf_url, extract_text=True, max_pages=30)
                if pdf_result.get('status') == 'success':
                    result['pdf_data'] = pdf_result
                    result['content'] = pdf_result.get('content', '')
                    result['file_path'] = pdf_result.get('file_path')
                    pdf_downloaded = True
                    logger.info(f"✅ PDF téléchargé et traité: {fao_code}")
            except Exception as e:
                logger.warning(f"Erreur téléchargement PDF construit: {e}")
        
        # Méthode 2: Liens PDF trouvés dans la page
        if not pdf_downloaded and page_metadata.get('pdf_links'):
            for pdf_link in page_metadata['pdf_links'][:3]:  # Limiter à 3 tentatives
                try:
                    logger.info(f"Tentative lien de page: {pdf_link}")
                    pdf_result = self.pdf_downloader.process_pdf(pdf_link, extract_text=True, max_pages=30)
                    if pdf_result.get('status') == 'success':
                        result['pdf_data'] = pdf_result
                        result['content'] = pdf_result.get('content', '')
                        result['file_path'] = pdf_result.get('file_path')
                        pdf_downloaded = True
                        logger.info(f"✅ PDF téléchargé depuis lien de page")
                        break
                except Exception as e:
                    logger.warning(f"Erreur téléchargement {pdf_link}: {e}")
        
        # 5. Fallback: Utiliser le contenu HTML
        if not pdf_downloaded:
            logger.warning(f"PDF non accessible, utilisation du contenu HTML")
            result['content'] = page_metadata.get('html_content', '')
            result['extraction_method'] = 'html_fallback'
        else:
            result['extraction_method'] = 'pdf'
        
        # 6. Sauvegarder les métadonnées complètes
        if fao_code:
            metadata_filename = f"{fao_code}_metadata.json"
        else:
            code_hash = abs(hash(doi_url)) % 100000
            metadata_filename = f"fao_{code_hash}_metadata.json"
        
        metadata_path = self.output_dir / metadata_filename
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        result['metadata_file'] = str(metadata_path)
        result['status'] = 'success'
        
        return result

    def scrape(self, doi_url: str) -> Dict[str, Any]:
        """Point d'entrée compatible avec ResourceManager."""
        return self.process_fao_doi(doi_url)


if __name__ == "__main__":
    # Test
    resolver = FaoDoiResolver()
    
    test_dois = [
        "https://doi.org/10.4060/cb9479fr",
        "https://doi.org/10.4060/cd3185en"
    ]
    
    for doi in test_dois:
        print(f"\n{'='*60}")
        print(f"Test: {doi}")
        result = resolver.process_fao_doi(doi)
        print(f"Status: {result.get('status')}")
        print(f"Titre: {result.get('title')}")
        print(f"Contenu extrait: {len(result.get('content', ''))} chars")
        if result.get('file_path'):
            print(f"Fichier: {result['file_path']}")
