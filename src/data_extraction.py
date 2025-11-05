"""Ce module est dedie a extraire les informations venant des sites web et des documents pdf que nous avons utilise"""

from src.data_processing import clean_text,normalize_text

import time
import logging
from typing import Dict, Optional, Any
import os, json
import urllib.robotparser
from bs4 import BeautifulSoup
import wget
import requests
from langdetect import detect
from pdfminer.high_level import extract_text as extract_pdf_text
import mimetypes
from pathlib import Path
from datetime import datetime
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



# Initialisation du corpus et des sources. Ci joint , les liens utilises

urls = [
    "https://fr.wikipedia.org/wiki/Agriculture_au_Burkina_Faso",
    "https://share.google/QbJCh2ygUQQBlGUtb",
    "https://share.google/0sJ6jsMgez1sI3Ye5",
    "https://afristat.org/wp-content/uploads/2022/04/NotesCours_Agri.pdf",
    "https://afristat.org/wp-content/uploads/2022/04/NotesCours_Agri.pdf",
    "https://share.google/JFsAeoTXBfuNtqTEu",
    "https://docs.google.com/forms/d/e/1FAIpQLSduR9VwiyApSfLsXGEu6oklvzNXOkYF-S6m0IV1zu1P3TsDVw/viewform",
     "https://doi.org/10.4060/cd3185en",
    "https://doi.org/10.4060/cd4965en",
    "https://doi.org/10.4060/cd4313en",
    "https://lefaso.net/spip.php?article141676",
    "https://doi.org/10.4060/cc8166en",
     "https://mita.coraf.org/assets/files/fiches/Mita--milismi9507.pdf""https://www.cirad.fr/les-actualites-du-cirad/actualites/2023/les-mils-cereales-pour-une-agriculture-resiliente",
    "https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.nitidae.org/files/acf8fe1c/manuel_de_formation_darda.pdf&ved=2ahUKEwipgaTzvdGQAxWBXEEAHaHjC5IQFnoECGsQAQ&sqi=2&usg=AOvVaw1byCUFVDKSINjUz5jsn5KD",
    "https://doi.org/10.4060/cd7304en",
    "https://burkinafaso.opendataforafrica.org/vvowcnc/production-de-mil-et-superficies-cultiv%C3%A9es-par-province",
    "https://doi.org/10.4060/cb9479fr",
    "https://reca-niger.org/IMG/pdf/culture_du_mil_et_contraintes_niger_2019.pdf",
    "https://openknowledge.fao.org/search?f.isPartOfSeries=La%20situation%20mondiale%20de%20l%27alimentation%20et%20l%27agriculture%20%20(SOFA),equals&spc.sf=dc.date.issued&spc.sd=DESC",
    "https://doi.org/10.4060/cb1447fr",
    "https://www.autreterre.org/agriculture-burkina-faso-affaire-de-femmes/",
    "https://cnrada.org/fiche-nuisibles/mil-mildiou-ou-lepre-du-mil/",
    "https://www.agri-mutuel.com/documents/",
    "https://www.fao.org/family-farming/detail/fr/c/472595/",
    "https://commons.wikimedia.org/wiki/File%3ATree_Crops%3B_A_Permanent_Agriculture_%281929%29.pdf?utm_source=chatgpt.com",
    "https://www.usgs.gov/apps/croplands/documents?utm_source=chatgpt.com",
    "https://www.researchgate.net/publication/370865727_Stockage_post_recolte_des_cereales_au_Burkina_Faso",
    "https://www.oecd.org/fr/publications/agriculture-alimentation-et-emploi-en-afrique-de-l-ouest_56d463a9-fr.html?utm_source=chatgpt.com",
    "https://publications.gc.ca/site/fra/431673/publication.html?utm_source=chatgpt.com",
    "https://www.cahiersagricultures.fr/articles/cagri/full_html/2020/01/cagri200020s/cagri200020s.html?utm_source=chatgpt.com"
    "https://www.fair-sahel.org/content/download/4680/35605/version/1/file/Rapport+FAIR+06+-+Int%C3%A9gration+AE+Burkina+03.pdf",
    "https://lefaso.net/spip.php?article141797",
    "https://www.sidwaya.info/developpement-de-lagriculture-a-yancheng-lempreinte-de-la-modernisation-a-la-chinoise/"
    "https://fr.wikipedia.org/wiki/Agriculture_au_Burkina_Faso", 
    "https://reseau-far.com/pays/burkina-faso/",
    "https://microdata.insd.bf/index.php/catalog/83",
    "https://microdata.insd.bf/index.php/home",
    "https://www.fao.org/in-action/agrisurvey/access-to-data/burkina-faso/en",
    "https://help.fews.net/fde/v1/burkina-faso-data-book",
    "https://catalog.data.gov/dataset/burkina-faso-compact-diversified-agriculture-and-water-management"
]

corpus, sources = [], []

 # Les noms de fichiers et dossiers utilises pour stocker les donnees extraites
DATA_DIR = 'data/'
PDF_DIR = os.getenv("PDF_DIR", "data/localDocuments")

THEME = "Agriculture"

# Nous definissons quelques fonctions qui nous seront utiles lors de l'extraction

# is_allowed nous permet de verifier si le scrapping est autorise par le fichier robots.txt du site web. Elle prend en entree l'url du document a scraper et nous retourne un booleen :Autorise/Non Autorise

def is_allowed(url) -> bool:
    BASE_URL = url.split("/")[0] + "//" + url.split("/")[2]
    robots_url = BASE_URL + "/robots.txt"
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(robots_url)

    try:
        rp.read()
        return rp.can_fetch("*", url)
    except:
        return False 

# detectFormat nous permet de detecter le format du document a scraper. Comme format,nous pouvons citer:PDF,HTML...Elle prend en entree le content_type et l'url du document et nous retourne le format detecte
      
def detectFormat(content_type, url)-> str:
    if content_type:

        if "pdf" in content_type:
            return "pdf"
        elif "html" in content_type:
            return "html"
        elif "text/plain" in content_type:
            return "txt"
        
    extention = mimetypes.guess_type(url)[0]
    if extention:
        if "pdf" in extention:
            return "pdf"
        elif "html" in extention:
            return "html"
        elif "text" in extention:
            return "txt"
    return "unknown"

# Cette fonction nous permet d'extraire le texte des differents formats de documents .Elle prend en entree l'url et le format du document et nous retourne le texte extrait

def extract_text_from_url(url, formatType)-> str:
    if formatType == "html":
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator=' ', strip=True)
    
    elif formatType == "pdf":
        response = requests.get(url)
        with open("temp.pdf", "wb") as f:
            f.write(response.content)
        return extract_pdf_text("temp.pdf")
    
    elif formatType == "txt":
        response = requests.get(url)
        return response.text
    
    else:
        return ""

# La fonction handleText nous permet d'extraire le texte,le nettoyer et le normaliser. En gros ,il nous permet de faire du preprocessing sur le texte extrait 

def handleText(url,formatType):
    text = extract_text_from_url(url, formatType)
    cleaned_text_ = clean_text(text)
    normalize_text_ = normalize_text(cleaned_text_)
    return normalize_text_


# La fonction SetCorpus nous permet de constituer le corpus a partir des urls et des documents pdf locaux. 

def SetCorpus(urls,PDF_DIR):

    for url in urls:
        # On commence par verifier si le scrapping est autorise. Si oui ,on passe a l'extraction. Si non ,on passe a l'url suivant
        
        if is_allowed(url):
            logger.info(f"Autorise : {url}")
            try:
                response = requests.get(url)
                content_type = response.headers.get("Content-Type")
                formatType = detectFormat(content_type, url)
                text = handleText(url,formatType)
                # Si le texte extrait est superieur a 20 caracteres ,on l'ajoute au corpus.En fait ,c'est une simple regle empirique pour eviter d'avoir des textes trop courts ou vides dans le corpus
                if len(text.strip()) > 20:
                    langage = detect(text) # Cette ligne nous permet de detecter la langue du texte extrait
                    corpus.append({
                        "source": url,
                        "text": text,
                        "metadata": {
                            "format": formatType,
                            "langue": langage,
                            "theme": THEME
                        }
                    })
                    sources.append(url)
                else:
                    logger.warning(f"Text court ou illisivle : {url}")
            except Exception as e:
                logger.warning(f"erreurn de scrapping : {e}")
        else:
            logger.warning(f"Interdit par robot.txt : {url}")
    
    """ Maintenant ,nous passons a l'extraction a partir des documents pdf locaux.
      Nous devons parcourir le dossier contenant les pdf et pour chaque pdf ,
      nous allons extraire le texte en utilisant la fonction handleText.
      Nous devons nous assurer que la variable pdf_folder est de type Path pour pouvoir utiliser iterdir()"""
    
    PDF_DIR = Path(PDF_DIR)

    for filename in PDF_DIR.iterdir():
        if filename.suffix.lower() == ".pdf":
            try:
                text = handleText(str(filename), "pdf")  # Ensure handleText accepts str path
                text = text.strip()

                if len(text) > 20:
                    # Language detection with fallback
                    try:
                        langage, score = detect(text)
                        if score < 0.8:
                            langage = "uncertain"
                    except Exception:
                        langage = "unknown"

                    # Optional metadata enrichment
                    metadata = {
                        "format": "pdf",
                        "langue": langage,
                        "theme": THEME,
                        "length": len(text),
                        "title": filename.stem,
                        "timestamp": datetime.now().isoformat()
                    }

                    # Eviter les doublons basés sur le nom de fichier
                    if not any(doc["source"] == str(filename) for doc in corpus):
                        corpus.append({
                            "source": str(filename),
                            "text": text,
                            "metadata": metadata
                        })
                else:
                    logger.warning(f"Nous n'arrivons pas a gerer {filename.name}: le texte est trop court ({len(text)} charecteres)")

            except Exception as e:
                logger.warning(f"Erreur avec {filename.name}: {e}")
        # Sauvegarde
        output_path = os.path.join(DATA_DIR, "corpus.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)
            logger.info(f"{len(corpus)} documents extraits et sauvegardés dans {DATA_DIR}/corpus.json.")
        # Sauvegarde des sources dans source.txt
        source_path = os.path.join(DATA_DIR, "source.txt")
        with open(source_path, "w", encoding="utf-8") as f:
            for src in sources:
                f.write(src + "\n")
                logger.info(f"{len(sources)} sources sauvegardées dans {source_path}.")
    
        return 'created'

# Cette fonction nous permet de lancer l'extraction en appelant la fonction SetCorpus avec les bons parametres. 
def run_extraction():
    return SetCorpus(urls,PDF_DIR)