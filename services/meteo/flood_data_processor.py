import json
import logging
import io
import sys
import os
from pathlib import Path
from typing import Tuple, Dict, Any
from PIL import Image, UnidentifiedImageError
from shapely.geometry import shape, box
from shapely.errors import ShapelyError

# --- IMPORT CONFIGURATION ---
try:
    from ... import config
except ImportError:
    # Ce bloc de secours assure que le script peut être exécuté même si l'architecture de package est plate.
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        import config
    except ImportError:
        # Mock de secours complet si config.py est totalement absent
        class ConfigMock:
            PROCESSOR_BBOX = (-5.5, 9.4, 2.4, 15.1) # BBOX pour le Burkina Faso
            PROCESSOR_FLOOD_KEYWORDS = {"flood", "inond", "alerte", "hazard", "risque", "threshold"}
            PROCESSOR_IMG_MIN_SIZE = 1024 # Taille minimale en bytes pour les images
            PROCESSOR_IMG_THRESHOLD = 0.90 # Ratio de couleur uniforme acceptable
            RAW_DATA_DIR = "./data/raw_fanfar"
            PROCESSED_DATA_FILE = "./data/processed/burkina_flood_risks.json"
        config = ConfigMock()

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("services.flood_processor")

class FloodDataProcessor:
    """
    Service de post-traitement optimisé : 
    - Nettoyage d'images (basé sur histogramme/échantillonnage rapide)
    - Agrégation GeoJSON (filtrage spatial et sémantique)
    
    Adapté pour utiliser la configuration centralisée (config.py).
    
    Dépendances requises : Pillow (PIL) et Shapely.
    """
    
    def __init__(self, raw_data_dir: Path = None, aggregated_output_path: Path = None):
        # Utilisation des chemins de la config si non fournis
        self.raw_data_dir = Path(raw_data_dir or getattr(config, 'RAW_DATA_DIR', "./data/raw"))
        self.aggregated_output_path = Path(aggregated_output_path or getattr(config, 'PROCESSED_DATA_FILE', "./data/processed/output.json"))
        
        # Paramètres depuis config
        bbox = getattr(config, 'PROCESSOR_BBOX', (-180, -90, 180, 90))
        # Crée une boîte Shapely pour le filtrage spatial rapide (Burkina Faso BBOX)
        self.burkina_box = box(*bbox)
        self.flood_keywords = getattr(config, 'PROCESSOR_FLOOD_KEYWORDS', set())
        self.min_file_size = getattr(config, 'PROCESSOR_IMG_MIN_SIZE', 1024)
        self.img_threshold = getattr(config, 'PROCESSOR_IMG_THRESHOLD', 0.90)
        
        # Création des dossiers si inexistants
        self.aggregated_output_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.raw_data_dir.exists():
            logger.warning(f"Dossier brut introuvable : {self.raw_data_dir}")

    # --- TRAITEMENT IMAGES OPTIMISÉ ---

    def _is_irrelevant_image(self, img: Image.Image) -> bool:
        """
        Vérifie si l'image est majoritairement noire (nuit/vide) ou verte (tuile de fond standard).
        Utilise un échantillonnage rapide (thumbnail) pour l'efficacité.
        """
        img = img.convert("RGB")
        # Réduit l'image pour un échantillonnage plus rapide (100x100 pixels)
        img.thumbnail((100, 100)) 
        
        pixels = list(img.getdata())
        total_pixels = len(pixels)
        if total_pixels == 0: return True

        target_pixels = 0
        
        for r, g, b in pixels:
            # Critère NOIR (représente souvent les zones de nuit ou les zones hors carte)
            if r < 40 and g < 40 and b < 40:
                target_pixels += 1
            # Critère VERT (représente souvent les tuiles de fond simples ou les zones de forêt non pertinentes)
            elif g > 80 and g > 1.3 * r and g > 1.3 * b:
                target_pixels += 1
        
        ratio = target_pixels / total_pixels
        # L'image est considérée comme non pertinente si plus du seuil (ex: 90%) des pixels correspondent
        return ratio >= self.img_threshold

    def clean_images(self) -> Tuple[int, int]:
        """Supprime les images jugées inutiles."""
        logger.info(f"Nettoyage des images dans {self.raw_data_dir}...")
        removed = 0
        kept = 0
        valid_extensions = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}

        for p in self.raw_data_dir.glob("*"):
            if p.suffix.lower() in valid_extensions:
                # 1. Filtrage par taille
                if p.stat().st_size < self.min_file_size:
                    p.unlink()
                    removed += 1
                    continue

                # 2. Filtrage par contenu
                try:
                    with Image.open(p) as img:
                        if self._is_irrelevant_image(img):
                            p.unlink()
                            removed += 1
                        else:
                            kept += 1
                except (UnidentifiedImageError, OSError):
                    # Fichier corrompu ou illisible
                    p.unlink(missing_ok=True)
                    removed += 1
                except Exception as e:
                    logger.warning(f"Erreur image {p.name}: {e}")

            # Filtrage spécifique pour les SVG de petite taille (souvent des icônes)
            elif p.suffix.lower() == ".svg":
                if p.stat().st_size < 800:
                    p.unlink()
                    removed += 1
                else:
                    kept += 1
            
            # Si le fichier est un JSON, on l'ignore pour le moment (il sera traité après)
            elif p.suffix.lower() != ".json":
                pass # Ignore les autres types de fichiers

        logger.info(f"Nettoyage terminé. Supprimés: {removed}, Conservés: {kept}.")
        return removed, kept

    # --- TRAITEMENT GEOJSON OPTIMISÉ ---

    def aggregate_flood_data(self):
        """Agrège les GeoJSON, filtre par BBOX et mots-clés."""
        logger.info("Agrégation des données vectorielles (GeoJSON)...")
        aggregated: Dict[str, Any] = {"type": "FeatureCollection", "features": []}
        files_processed = 0
        features_kept = 0
        
        if not self.raw_data_dir.exists():
            logger.error(f"Le dossier source {self.raw_data_dir} n'existe pas.")
            return

        for p in self.raw_data_dir.glob("*.json"):
            # Évite de traiter le fichier de sortie s'il est dans le dossier source
            if p.resolve() == self.aggregated_output_path.resolve(): continue

            try:
                content = p.read_text(encoding="utf-8")
                if not content.strip(): continue
                raw = json.loads(content)
            except Exception as e:
                logger.debug(f"Impossible de lire/parser JSON {p.name}: {e}")
                continue
            
            features_list = []
            if raw.get("type") == "FeatureCollection":
                features_list = raw.get("features", [])
            elif raw.get("type") == "Feature":
                features_list = [raw]
            # Gestion d'autres formats JSON qui pourraient contenir des features (ex: API brutes)
            elif isinstance(raw, list) and all(isinstance(item, dict) and item.get("geometry") for item in raw):
                features_list = raw

            if not features_list: continue
            files_processed += 1

            for feat in features_list:
                geom = feat.get("geometry")
                props = feat.get("properties", {}) or {}
                
                if not geom: continue

                # 1. Validation Géométrique et Réparation (si nécessaire)
                try:
                    shp = shape(geom)
                    if not shp.is_valid:
                        # Tente une réparation simple de la géométrie
                        shp = shp.buffer(0) 
                except (ShapelyError, ValueError):
                    logger.debug(f"Géométrie non valide dans {p.name}, ignorée.")
                    continue

                # 2. Filtrage Spatial: Vérifie l'intersection avec la BBOX du Burkina Faso
                if not shp.intersects(self.burkina_box):
                    continue

                # 3. Filtrage Sémantique: Recherche de mots-clés dans les propriétés/ID
                props_values = " ".join([str(v) for v in props.values()]).lower()
                feature_id = str(feat.get("id", "")).lower()
                
                is_relevant = any(k in props_values for k in self.flood_keywords) or \
                              any(k in feature_id for k in self.flood_keywords)

                if is_relevant:
                    # Ajoute les métadonnées de traitement à la feature
                    props["source_file"] = p.name
                    props["processed_date"] = "2025-11-27"
                    feat["properties"] = props
                    
                    # S'assure d'utiliser la géométrie potentiellement réparée (si buffer(0) a été appelé)
                    # Note: Pour ne pas casser le GeoJSON brut, on ne réinjecte que si la géométrie était invalide
                    if not shape(geom).is_valid:
                         feat["geometry"] = json.loads(shp.to_json())
                    
                    aggregated["features"].append(feat)
                    features_kept += 1

        try:
            with open(self.aggregated_output_path, 'w', encoding='utf-8') as f:
                json.dump(aggregated, f, ensure_ascii=False, indent=2)
            logger.info(f"Succès. {features_kept} entités agrégées depuis {files_processed} fichiers.")
        except Exception as e:
            logger.error(f"Erreur écriture fichier agrégé : {e}")

# --- EXEMPLE D'UTILISATION ---
if __name__ == "__main__":
    # Ce bloc d'exécution nécessite que Pillow (PIL) et Shapely soient installés.
    print("Démarrage du Processeur de Données d'Inondation...")
    
    # Utilise les chemins par défaut définis dans config.py (ou le Mock)
    processor = FloodDataProcessor()
    
    # --- Création de dossiers (pour que l'exemple d'exécution fonctionne) ---
    if not processor.raw_data_dir.exists():
        processor.raw_data_dir.mkdir(parents=True)
        print(f"Dossier brut créé : {processor.raw_data_dir.resolve()}")
    
    # --- Exécution des étapes de nettoyage et d'agrégation ---
    removed, kept = processor.clean_images()
    processor.aggregate_flood_data()
    
    print("-" * 50)
    print("Traitement terminé.")
    print(f"Chemin de sortie agrégé : {processor.aggregated_output_path.resolve()}")