import logging
# Assurez-vous d'importer la classe depuis votre fichier
from services.meteo.fanfar_flood_risk import FanfarFloodRiskService, FloodRiskResult 

# Configuration du logging (pour voir les logs du service)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_fanfar")

def test_successful_extraction(location: str):
    """Teste l'extraction du risque pour une localité connue au BF."""
    logger.info(f"--- Début du test de réussite pour {location} ---")
    # Tentez headless=False au début pour observer l'interaction
    scraper = FanfarFloodRiskService(headless=True) 
    result: FloodRiskResult = scraper.get_flood_risk(location)
    
    # Vérifications
    assert result.location == location
    # L'un de ces deux champs doit être renseigné pour indiquer une extraction réussie
    if result.severity_level is None:
        logger.warning(f"Risque non détecté, mais la page a pu s'ouvrir. Brut: {result.raw_text}")
    
    print("\n✅ Résultat du Test Réussite :")
    print(result)

# Exécutez le test avec une capitale et une autre ville significative
test_successful_extraction("Ouagadougou")
# test_successful_extraction("Bobo-Dioulasso") # Ajoutez un deuxième point de contrôle