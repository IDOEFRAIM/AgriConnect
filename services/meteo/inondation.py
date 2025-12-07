# Fichier : Agents/Meteo/fanfar_floods_burkina.py (Chef d'orchestre)

# Imports des services (doivent être accessibles via le chemin)
# from services.fanfar_scraper_service import FanfarScraperService
# from services.flood_data_processor import FloodDataProcessor
from pathlib import Path
import asyncio
from services.meteo import FloodDataProcessor,FanfarScraperService

# ... (le code d'importation des modules définis ci-dessus)

AGGREGATED_PATH = Path("fanfar_output/floods_burkina_aggregated.json")
RAW_DATA_DIR = Path("fanfar_raw_output") 

async def main_pipeline(headless: bool = True):
    # 1. Initialisation des Services
    scraper = FanfarScraperService(output_dir_name=RAW_DATA_DIR.name)
    processor = FloodDataProcessor(
        raw_data_dir=RAW_DATA_DIR,
        aggregated_output_path=AGGREGATED_PATH
    )
    
    # 2. Exécution du Scraping (Acquisition des données brutes)
    print("--- Démarrage de l'acquisition FANFAR ---")
    await scraper.run_scraper_pipeline(headless=headless)
    
    # 3. Exécution du Traitement (Nettoyage et Filtrage)
    print("--- Démarrage du post-traitement ---")
    
    # Nettoyage des images (optionnel mais bon pour le stockage)
    processor.clean_images()
    
    # Filtrage et agrégation GeoJSON
    processor.aggregate_flood_data()
    
    print(f"✅ Pipeline FANFAR terminé. Données agrégées prêtes dans : {AGGREGATED_PATH.resolve()}")

if __name__ == "__main__":
    # Exécute visible la première fois pour debug, change headless=True pour production
    asyncio.run(main_pipeline(headless=False))