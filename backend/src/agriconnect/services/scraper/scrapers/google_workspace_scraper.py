"""
GoogleWorkspaceScraper - Extraction de contenu depuis Google Drive, Docs, Forms.

Stratégies d'extraction:
1. Google Drive Share Links: Convertir en export URLs direct
2. Google Docs: Extraire en HTML ou Plain Text via API publique
3. Google Forms: Extraire la structure du formulaire (questions, options)
4. Fallback: Scraping HTML si API non disponible

Note: Les liens partagés publiquement peuvent être accédés sans authentification.
"""

import requests
from bs4 import BeautifulSoup
import re
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import time

logger = logging.getLogger("GoogleWorkspaceScraper")


class GoogleWorkspaceScraper:
    """
    Scraper spécialisé pour Google Workspace (Drive, Docs, Forms).
    """

    def __init__(self, output_dir: str = "backend/sources/raw_data/google_workspace"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def identify_link_type(self, url: str) -> str:
        """Identifie le type de lien Google."""
        url_lower = url.lower()
        
        if 'drive.google.com' in url_lower or 'share.google' in url_lower:
            return 'drive'
        elif 'docs.google.com/document' in url_lower:
            return 'docs'
        elif 'docs.google.com/spreadsheets' in url_lower:
            return 'sheets'
        elif 'docs.google.com/presentation' in url_lower:
            return 'slides'
        elif 'docs.google.com/forms' in url_lower:
            return 'forms'
        else:
            return 'unknown'

    def extract_file_id(self, url: str) -> Optional[str]:
        """Extrait l'ID du fichier depuis une URL Google."""
        patterns = [
            r'/d/([a-zA-Z0-9-_]+)',  # /d/FILE_ID
            r'id=([a-zA-Z0-9-_]+)',   # ?id=FILE_ID
            r'/([a-zA-Z0-9-_]+)/edit', # /FILE_ID/edit
            r'/([a-zA-Z0-9-_]+)/view', # /FILE_ID/view
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None

    def scrape_drive_file(self, url: str) -> Dict[str, Any]:
        """
        Extrait un fichier Google Drive.
        Stratégie: Tenter d'exporter en PDF ou HTML si document.
        """
        file_id = self.extract_file_id(url)
        if not file_id:
            return {'status': 'failed', 'error': 'Impossible d\'extraire l\'ID du fichier'}
        
        # Tentative 1: Export en PDF (fonctionne pour Docs, Sheets, Slides)
        export_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        try:
            response = self.session.get(export_url, timeout=30, allow_redirects=True)
            
            # Vérifier si c'est un avertissement de scan virus
            if 'download_warning' in response.text or 'virus scan' in response.text.lower():
                # Extraire le vrai lien de téléchargement
                soup = BeautifulSoup(response.text, 'html.parser')
                download_link = soup.find('a', {'id': 'uc-download-link'})
                if download_link and download_link.get('href'):
                    confirm_url = 'https://drive.google.com' + download_link['href']
                    response = self.session.get(confirm_url, timeout=30)
            
            # Sauvegarder le fichier
            if response.status_code == 200:
                # Déterminer l'extension depuis Content-Type
                content_type = response.headers.get('Content-Type', '').lower()
                
                extension = 'bin'
                if 'pdf' in content_type:
                    extension = 'pdf'
                elif 'html' in content_type:
                    extension = 'html'
                elif 'text/plain' in content_type:
                    extension = 'txt'
                elif 'officedocument' in content_type:
                    if 'wordprocessing' in content_type:
                        extension = 'docx'
                    elif 'spreadsheet' in content_type:
                        extension = 'xlsx'
                    elif 'presentation' in content_type:
                        extension = 'pptx'
                
                filename = f"gdrive_{file_id}.{extension}"
                filepath = self.output_dir / filename
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                # Extraction de texte si possible
                content_text = self._extract_text_from_file(filepath, extension)
                
                return {
                    'status': 'success',
                    'url': url,
                    'source_type': 'google_drive',
                    'file_id': file_id,
                    'file_path': str(filepath),
                    'content_type': content_type,
                    'content': content_text,
                    'metadata': {
                        'file_size': len(response.content),
                        'extension': extension
                    }
                }
            else:
                # Fallback: Essayer d'accéder à la page de preview
                return self._scrape_preview_page(file_id, url)
                
        except Exception as e:
            logger.error(f"Erreur scraping Drive file {file_id}: {e}")
            return {'status': 'failed', 'url': url, 'error': str(e)}

    def _scrape_preview_page(self, file_id: str, original_url: str) -> Dict[str, Any]:
        """Scrape la page de preview Google Drive."""
        preview_url = f"https://drive.google.com/file/d/{file_id}/preview"
        
        try:
            response = self.session.get(preview_url, timeout=20)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extraire métadonnées visibles
            title = soup.find('title')
            title_text = title.get_text() if title else f"Document {file_id}"
            
            # Chercher du contenu textuel
            content_div = soup.find('div', {'class': 'ndfHFb-c4YZDc-Wrql6b'}) or \
                         soup.find('div', {'role': 'main'})
            
            content_text = ""
            if content_div:
                content_text = content_div.get_text(separator='\n', strip=True)
            
            return {
                'status': 'success',
                'url': original_url,
                'source_type': 'google_drive_preview',
                'file_id': file_id,
                'title': title_text,
                'content': content_text,
                'metadata': {
                    'extraction_method': 'preview_page'
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur scraping preview page {file_id}: {e}")
            return {'status': 'failed', 'url': original_url, 'error': str(e)}

    def scrape_google_doc(self, url: str) -> Dict[str, Any]:
        """
        Extrait un Google Doc.
        Utilise l'export HTML public.
        """
        file_id = self.extract_file_id(url)
        if not file_id:
            return {'status': 'failed', 'error': 'ID document introuvable'}
        
        # URL d'export HTML public
        export_url = f"https://docs.google.com/document/d/{file_id}/export?format=html"
        
        try:
            response = self.session.get(export_url, timeout=20)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extraire le titre
                title = soup.find('title')
                title_text = title.get_text() if title else f"Google Doc {file_id}"
                
                # Extraire le contenu
                body = soup.find('body')
                content_text = body.get_text(separator='\n', strip=True) if body else ""
                
                # Sauvegarder le HTML
                html_filename = f"gdoc_{file_id}.html"
                html_path = self.output_dir / html_filename
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                return {
                    'status': 'success',
                    'url': url,
                    'source_type': 'google_docs',
                    'file_id': file_id,
                    'title': title_text,
                    'content': content_text,
                    'file_path': str(html_path),
                    'metadata': {
                        'content_length': len(content_text),
                        'export_format': 'html'
                    }
                }
            else:
                return {'status': 'failed', 'url': url, 'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            logger.error(f"Erreur scraping Google Doc {file_id}: {e}")
            return {'status': 'failed', 'url': url, 'error': str(e)}

    def scrape_google_form(self, url: str) -> Dict[str, Any]:
        """
        Extrait la structure d'un Google Form (questions, options).
        """
        try:
            response = self.session.get(url, timeout=20)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extraire le titre du formulaire
            title_elem = soup.find('div', {'class': 'freebirdFormviewerViewHeaderTitle'}) or \
                        soup.find('h1')
            title = title_elem.get_text(strip=True) if title_elem else "Google Form"
            
            # Extraire la description
            desc_elem = soup.find('div', {'class': 'freebirdFormviewerViewHeaderDescription'})
            description = desc_elem.get_text(strip=True) if desc_elem else ""
            
            # Extraire les questions
            questions = []
            question_divs = soup.find_all('div', {'class': re.compile('freebirdFormviewerComponent')})
            
            for idx, q_div in enumerate(question_divs, 1):
                question_text_elem = q_div.find('div', {'role': 'heading'}) or \
                                    q_div.find('div', {'class': re.compile('freebirdFormviewerComponent.*Title')})
                
                if question_text_elem:
                    question_text = question_text_elem.get_text(strip=True)
                    
                    # Chercher les options de réponse
                    options = []
                    option_elems = q_div.find_all('span', {'class': re.compile('.*OptionText')}) or \
                                  q_div.find_all('div', {'class': re.compile('.*Option')})
                    
                    for opt in option_elems:
                        opt_text = opt.get_text(strip=True)
                        if opt_text:
                            options.append(opt_text)
                    
                    questions.append({
                        'question_number': idx,
                        'question': question_text,
                        'options': options if options else None
                    })
            
            # Sauvegarder la structure
            form_data = {
                'title': title,
                'description': description,
                'questions': questions,
                'url': url
            }
            
            json_filename = f"gform_{hash(url) % 100000}.json"
            json_path = self.output_dir / json_filename
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(form_data, f, indent=2, ensure_ascii=False)
            
            # Créer un résumé textuel
            content_text = f"Formulaire: {title}\n\n{description}\n\n"
            for q in questions:
                content_text += f"Q{q['question_number']}: {q['question']}\n"
                if q['options']:
                    content_text += "  Options: " + ", ".join(q['options']) + "\n"
                content_text += "\n"
            
            return {
                'status': 'success',
                'url': url,
                'source_type': 'google_forms',
                'title': title,
                'content': content_text,
                'file_path': str(json_path),
                'metadata': {
                    'questions_count': len(questions),
                    'has_description': bool(description)
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur scraping Google Form: {e}")
            return {'status': 'failed', 'url': url, 'error': str(e)}

    def _extract_text_from_file(self, filepath: Path, extension: str) -> str:
        """Extrait le texte d'un fichier téléchargé."""
        try:
            if extension == 'txt':
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            
            elif extension == 'html':
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    soup = BeautifulSoup(f.read(), 'html.parser')
                    return soup.get_text(separator='\n', strip=True)
            
            elif extension == 'pdf':
                # Import conditionnel pour éviter les dépendances
                try:
                    import pdfplumber
                    with pdfplumber.open(filepath) as pdf:
                        text = ""
                        for page in pdf.pages[:10]:  # Limiter aux 10 premières pages
                            text += page.extract_text() or ""
                        return text
                except ImportError:
                    logger.warning("pdfplumber non installé, extraction PDF ignorée")
                    return "[PDF - extraction non disponible]"
            
            else:
                return f"[Fichier {extension} - extraction non supportée]"
                
        except Exception as e:
            logger.error(f"Erreur extraction texte depuis {filepath}: {e}")
            return "[Erreur extraction]"

    def scrape(self, url: str) -> Dict[str, Any]:
        """
        Point d'entrée principal - détecte le type et route vers le bon handler.
        """
        link_type = self.identify_link_type(url)
        
        logger.info(f"Type détecté: {link_type} pour {url}")
        
        if link_type == 'drive':
            return self.scrape_drive_file(url)
        elif link_type == 'docs':
            return self.scrape_google_doc(url)
        elif link_type == 'forms':
            return self.scrape_google_form(url)
        else:
            # Tentative générique
            return self.scrape_drive_file(url)


if __name__ == "__main__":
    # Test
    scraper = GoogleWorkspaceScraper()
    
    test_urls = [
        "https://docs.google.com/forms/d/e/1FAIpQLSduR9VwiyApSfLsXGEu6oklvzNXOkYF-S6m0IV1zu1P3TsDVw/viewform"
    ]
    
    for url in test_urls:
        print(f"\n{'='*60}")
        print(f"Test: {url}")
        result = scraper.scrape(url)
        print(json.dumps(result, indent=2, ensure_ascii=False))
