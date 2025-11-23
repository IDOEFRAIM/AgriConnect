import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import io
from PyPDF2 import PdfReader

def check_robots_txt(url: str) -> dict:
    """Analyse le fichier robots.txt du site."""
    base_url = url.split("/")[0] + "//" + url.split("/")[2]
    robots_url = f"{base_url}/robots.txt"
    try:
        response = requests.get(robots_url, timeout=5)
        if response.status_code != 200:
            return {"status": "not found", "allowed": True}
        content = response.text.lower()
        return {
            "status": "found",
            "allowed": "disallow: /" not in content
        }
    except Exception as e:
        return {"status": "error", "allowed": False, "error": str(e)}

def check_mentions_legales(url: str) -> dict:
    """Vérifie la présence d'une page de mentions légales sur le site."""
    base_url = url.split("/")[0] + "//" + url.split("/")[2]
    possible_paths = [
        "/mentions-legales",
        "/mentions_legales",
        "/legal",
        "/legal-notice",
        "/legal_notice"
    ]
    
    for path in possible_paths:
        full_url = f"{base_url}{path}"
        try:
            response = requests.get(full_url, timeout=5)
            if response.status_code == 200 and "mentions" in response.text.lower():
                return {"status": "found", "url": full_url}
        except Exception as e:
            continue  # Ignore and try next path

    return {"status": "not found", "url": None}

def scrape_page_content(url: str) -> dict:
    """
    Scrape le contenu principal de la page (titre + paragraphes). sans forcement verifier si c'est autorise
    """
    try:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            return {"status": "error", "code": response.status_code}
        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.title.string if soup.title else "Sans titre"
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        return {
            "status": "success",
            "title": title,
            "paragraphs": paragraphs[:5]  # Limite à 5 paragraphes pour l'exemple
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def scrape_with_checks(url: str) -> dict:
    """Vérifie robots.txt, mentions légales, puis scrape si autorisé."""
    result = {
        "robots": check_robots_txt(url),
        "mentions_legales": check_mentions_legales(url),
        "scraping": None
    }

    if result["robots"].get("allowed", False):
        result["scraping"] = scrape_page_content(url)
    else:
        result["scraping"] = {"status": "forbidden", "message": "Scraping interdit par robots.txt"}

    return result



def scrape_dynamic_content(url: str) -> dict:
    """Scrape dynamiquement du HTML ou du PDF selon le type de contenu."""
    try:
        response = requests.get(url, timeout=10)
        content_type = response.headers.get("Content-Type", "").lower()

        if "text/html" in content_type:
            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.title.string if soup.title else "Sans titre"
            paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
            return {
                "type": "html",
                "status": "success",
                "title": title,
                "paragraphs": paragraphs[:5]
            }

        elif "application/pdf" in content_type:
            pdf_file = io.BytesIO(response.content)
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages[:3]:  # Limite à 3 pages pour l'exemple
                text += page.extract_text() + "\n"
            return {
                "type": "pdf",
                "status": "success",
                "text": text.strip()
            }

        else:
            return {
                "type": "unknown",
                "status": "unsupported",
                "message": f"Type de contenu non pris en charge : {content_type}"
            }

    except Exception as e:
        return {"status": "error", "message": str(e)}
    

"""
https://meteoburkina.bf/le-climat-de-nos-villes/
retrieve 
- Analyser les tendances climatiques locales
- Construire des modèles agroclimatiques
- Adapter les recommandations agricoles selon les saisons

"""