# rag_pipeline_single_file.py

"""
PIPELINE RAG MONOLITHIQUE
Ce fichier unique regroupe l'ensemble des fonctionnalites d'extraction, de 
nettoyage, de caching (lru_cache), d'un service d'embedding simule et d'orchestration.
"""

import time
import logging
from typing import Dict, Optional, Any, List
import os, json
import urllib.robotparser
from bs4 import BeautifulSoup
import requests
from langdetect import detect
from pdfminer.high_level import extract_text as extract_pdf_text
import mimetypes
from pathlib import Path
from datetime import datetime
import functools
import random
from hashlib import sha256
from services.utils.embedding import EmbeddingService
# --- CONFIGURATION ET SETUP ---

# Configuration du Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RAGPipeline")

# Configuration des Chemins et Constantes
DATA_DIR = Path('data/')
PDF_DIR = DATA_DIR / 'localDocuments'
THEME = "Agriculture"
MIN_TEXT_LENGTH = 20 # Longueur minimale de texte pour être considéré valide
OUTPUT_CORPUS_FILE = DATA_DIR / "corpus_embedded.json"
OUTPUT_SOURCES_FILE = DATA_DIR / "source.txt"
# Augmenté à 1536 pour simuler une dimension courante des modèles
EMBEDDING_DIMENSION = 1536 

# --- 1. TRAITEMENT DU TEXTE (Nettoyage/Normalisation) ---

def clean_text(text: str) -> str:
    """
    Nettoie le texte en retirant les sauts de ligne excessifs, 
    les tabulations et les doubles espaces.
    """
    # Remplacement des sauts de ligne et tabulations par un espace
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    
    # Remplacement des multiples espaces par un seul espace
    while '  ' in text:
        text = text.replace('  ', ' ')
        
    return text.strip()

def normalize_text(text: str) -> str:
    """
    Normalise le texte (mise en minuscule).
    """
    return text.lower()


# --- 2. FONCTIONS UTILITAIRES D'ACCÈS ET D'EXTRACTION (Avec Caching) ---

@functools.lru_cache(maxsize=1024)
def is_allowed(url: str) -> bool:
    """Vérifie l'autorisation de scraping via robots.txt (Mis en cache)."""
    try:
        parts = requests.utils.urlparse(url)
        base_url = f"{parts.scheme}://{parts.netloc}"
        robots_url = f"{base_url}/robots.txt"
        
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch("*", url)
    except Exception:
        if not url.startswith(('http', 'https')):
            return False
        return True 

def detect_format(content_type: Optional[str], url: str) -> str:
    """Détecte le format du document à partir du Content-Type ou de l'URL."""
    if content_type:
        if "pdf" in content_type: return "pdf"
        if "html" in content_type: return "html"
        if "text/plain" in content_type: return "txt"
        
    ext = Path(url).suffix.lower()
    if ".pdf" in ext: return "pdf"
    if ext in (".htm", ".html", ".php"): return "html"
    
    inferred_mime, _ = mimetypes.guess_type(url)
    if inferred_mime and "pdf" in inferred_mime: return "pdf"

    return "unknown"

@functools.lru_cache(maxsize=1024)
def extract_text(source_path: str, format_type: str) -> str:
    """
    Extrait le texte brut (Mis en cache).
    Note: Les fonctions mises en cache doivent accepter des arguments hachables.
    """
    
    if format_type == "html":
        response = requests.get(source_path, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")
        [s.extract() for s in soup(['style', 'script', 'header', 'footer', 'nav'])]
        return soup.get_text(separator=' ', strip=True)

    elif format_type == "pdf":
        if source_path.startswith(('http', 'https')):
            # Téléchargement temporaire et extraction PDF distante
            temp_file = "temp_download.pdf"
            try:
                response = requests.get(source_path, stream=True, timeout=30)
                response.raise_for_status() 
                with open(temp_file, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                text = extract_pdf_text(temp_file)
                os.remove(temp_file)
                return text
            except Exception as e:
                logger.warning(f"Échec de l'extraction PDF distante pour {source_path}: {e}")
                if os.path.exists(temp_file): os.remove(temp_file)
                return ""
        else:
            # Extraction PDF locale
            return extract_pdf_text(source_path)

    elif format_type == "txt":
        response = requests.get(source_path, timeout=10)
        return response.text
        
    return ""


# --- 3. FONCTION D'EMBEDDING (Simulation avec Caching intégré) ---


# Instanciation du service d'embedding pour usage global
EMBEDDING_SERVICE = EmbeddingService()

# --- 4. LOGIQUE MÉTIER CENTRALE ---

def handle_text_extraction(source: str, format_type: str) -> Optional[Dict[str, Any]]:
    """
    Fonction centrale: extrait, traite, enrichit les metadonnees, filtre 
    et genere l'embedding.
    """
    try:
        start_time = time.time()
        
        # 1. Extraction brute (Potentiellement cachee)
        text = extract_text(source, format_type)
        if not text:
            return None
            
        # 2. Traitement du texte
        cleaned_text = clean_text(text)
        normalized_text = normalize_text(cleaned_text)
        
        # 3. Filtration par taille minimale
        if len(normalized_text.strip()) <= MIN_TEXT_LENGTH:
            logger.warning(f"Texte trop court (len={len(normalized_text.strip())}) pour {source}")
            return None

        # 4. Génération de l'Embedding (Utilisation du service centralisé !)
        # Nous appelons ici la méthode mise en cache de la classe.
        embedding_vector = EMBEDDING_SERVICE.embed_documents(normalized_text)
            
        # 5. Détection de langue
        langage = "unknown"
        try:
            langage = detect(normalized_text[:1000])
        except Exception:
            pass
            
        # 6. Construction de l'objet document
        doc_data = {
            "source": source,
            "text": normalized_text.strip(),
            "embedding": embedding_vector, # Ajout de l'embedding
            "metadata": {
                "format": format_type,
                "langue": langage,
                "theme": THEME,
                "timestamp": datetime.now().isoformat(),
                "extraction_time_s": round(time.time() - start_time, 2)
            }
        }
        if not source.startswith(('http', 'https')):
            doc_data["metadata"]["title"] = Path(source).stem
        
        return doc_data

    except Exception as e:
        logger.error(f"Erreur lors de handle_text_extraction pour {source}: {e}")
        return None

# --- 5. ORCHESTRATION ET SAUVEGARDE ---

def extract_from_urls(url_list: List[str]) -> List[Dict[str, Any]]:
    """Gère l'extraction, le filtrage robots.txt et la détection de format pour les URLs."""
    corpus_segment = []
    logger.info(f"\n--- 🌐 DÉMARRAGE DE L'EXTRACTION WEB ({len(url_list)} URLs uniques) ---")
    
    for url in url_list:
        if not url.startswith(('http', 'https')): continue
        
        # Vérification robot.txt (Potentiellement cachee)
        if not is_allowed(url):
            logger.warning(f"Interdit par robots.txt (Skipped) : {url}")
            continue
            
        try:
            # Utilisation de HEAD pour une détection de format rapide
            response = requests.head(url, allow_redirects=True, timeout=10)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type")
            format_type = detect_format(content_type, url)
            
            if format_type == "unknown":
                logger.warning(f"Format non supporté pour {url}: {content_type}")
                continue

            doc_data = handle_text_extraction(url, format_type)
            
            if doc_data:
                corpus_segment.append(doc_data)
                logger.info(f"✅ Extrait ({format_type}, {len(doc_data['text'])} chars) de {url}")
            # else: le warning est loggué dans handle_text_extraction
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur HTTP/Connexion pour {url}: {e}")
        except Exception as e:
            logger.error(f"Erreur inattendue pour {url}: {e}")

    return corpus_segment


def extract_from_local_pdfs(pdf_dir: Path) -> List[Dict[str, Any]]:
    """Gère l'extraction des documents PDF locaux."""
    corpus_segment = []
    
    if not pdf_dir.is_dir():
        logger.error(f"❌ Le dossier PDF local {pdf_dir} n'existe pas. Skipping.")
        return []

    logger.info(f"\n--- 🗄️ DÉMARRAGE DE L'EXTRACTION PDF LOCALE ({pdf_dir}) ---")
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    for filename in pdf_files:
        doc_data = handle_text_extraction(str(filename), "pdf")
        
        if doc_data:
            corpus_segment.append(doc_data)
            logger.info(f"✅ Extrait ({len(doc_data['text'])} chars) de {filename.name}")
        else: 
            logger.warning('an error occur')
            
    return corpus_segment


def save_corpus(corpus: List[Dict[str, Any]]):
    """Sauvegarde les résultats finaux dans les fichiers JSON (avec embeddings) et TXT."""
    
    if not DATA_DIR.is_dir():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Sauvegarde du Corpus JSON (avec embeddings)
    with open(OUTPUT_CORPUS_FILE, "w", encoding="utf-8") as f:
        # Simplifié pour un fichier plus compact :
        json.dump(corpus, f, ensure_ascii=False, separators=(',', ':'))
        
    logger.info(f"\n🏁 CORPUS TERMINÉ. {len(corpus)} documents extraits (avec {EMBEDDING_DIMENSION}D embeddings simulés) et sauvegardés dans {OUTPUT_CORPUS_FILE.name}.")

    # 2. Sauvegarde des Sources TXT
    sources = [doc['source'] for doc in corpus]
    with open(OUTPUT_SOURCES_FILE, "w", encoding="utf-8") as f:
        f.write('\n'.join(sources) + '\n')
    logger.info(f"{len(sources)} sources sauvegardées dans {OUTPUT_SOURCES_FILE.name}.")


def run_extraction_pipeline(url_list: List[str], pdf_dir: Path):
    """Orchestre l'extraction complète."""
    
    # Assurer l'existence des dossiers nécessaires
    if not DATA_DIR.is_dir(): DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not PDF_DIR.is_dir(): PDF_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("🚀 DÉMARRAGE DU PIPELINE D'EXTRACTION RAG")
    logger.info(f"💾 Caching actif pour l'I/O et l'Embedding. Dimension Embedding: {EMBEDDING_DIMENSION}D.")
    start_time = time.time()
    
    # Nettoyer les doublons dans la liste d'URLs
    unique_urls = list(set(url_list)) 
    
    final_corpus = []
    
    # 1. Extraction Web (URLs)
    web_corpus = extract_from_urls(unique_urls)
    final_corpus.extend(web_corpus)
    
    # 2. Extraction PDF Locaux
    #pdf_corpus = extract_from_local_pdfs(pdf_dir)
    #final_corpus.extend(pdf_corpus)
    
    # 3. Sauvegarde
    if final_corpus:
        save_corpus(final_corpus)
    else:
        logger.warning("🚫 Corpus vide. Aucune donnée n'a été extraite et sauvegardée.")
        
    end_time = time.time()
    total_time = end_time - start_time
    total_docs = len(final_corpus)
    
    # Affichage du résumé
    print("\n" + "="*70)
    print("RÉSUMÉ FINAL DU PIPELINE RAG")
    print(f"Documents Totaux (avec Embeddings) : {total_docs}")
    print(f"Documents Web : {len(web_corpus)}")
   # print(f"Documents PDF Locaux : {len(pdf_corpus)}")
    print(f"Temps total d'exécution : {total_time:.2f} secondes.")
    print(f"Sortie Corpus : {OUTPUT_CORPUS_FILE.name}")
    print("="*70)


# ----------------------------------------------------------------------
# --- POINT D'ENTRÉE PRINCIPAL ---
# ----------------------------------------------------------------------
if __name__ == "__main__":
    
    # Données d'entrée (déplacées ici)
    URLS_TO_SCRAPE: List[str] = [
        "https://fr.wikipedia.org/wiki/Agriculture_au_Burkina_Faso",
        "https://afristat.org/wp-content/uploads/2022/04/NotesCours_Agri.pdf",
        "https://lefaso.net/spip.php?article141676",
        "https://www.fao.org/in-action/agrisurvey/access-to-data/burkina-faso/en",
        "https://cnrada.org/fiche-nuisibles/mil-mildiou-ou-lepre-du-mil/",
        "https://burkinafaso.opendataforafrica.org/vvowcnc/production-de-mil-et-superficies-cultiv%C3%A9es-par-province",
    ]
    
    # --- DÉMONSTRATION DE L'EMBEDDING SERVICE (DEMANDÉE) ---
    print("\n" + "~"*70)
    print(f"DÉMONSTRATION: INVOCATION DE L'EMBEDDING SERVICE (DIM: {EMBEDDING_DIMENSION})")

    sample_text = "L'agriculture est essentielle au Burkina Faso."
    
    # 1er appel: calcul et cache
    embedding_vector_1 = EMBEDDING_SERVICE.embed_documents(sample_text) 
    print(f"1er appel (Calcul et Cache): Vector (début): {embedding_vector_1[:5]}...")

    # 2ème appel: lecture depuis le cache LRU
    embedding_vector_2 = EMBEDDING_SERVICE.embed_documents(sample_text)
    print(f"2ème appel (Cache Hit): Vector (début): {embedding_vector_2[:5]}...")
    
    # Notez que le cache LRU n'affiche pas de log "Cache Hit" par défaut, 
    # mais l'appel est beaucoup plus rapide.

    print(f"Longueur du vecteur généré: {len(embedding_vector_1)}")
    print("~"*70 + "\n")
    # --------------------------------------------------------

    run_extraction_pipeline(URLS_TO_SCRAPE, PDF_DIR)