import logging
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional

# Correction de l'import: PlaywrightTimeoutError est l'alias de TimeoutError dans playwright.sync_api
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError, Page, BrowserContext, APIResponse

# Configuration des logs
logger = logging.getLogger("services.map_visual_scraper")

# --- Import et Mock des dépendances externes ---
try:
    # Tentative d'import des modules frères
    from . import core_utils 
    from .. import config
except ImportError:
    # Fallback pour exécution isolée (nécessite des mocks locaux)
    logger.warning("WARNING: Using mock configuration and isolated imports. Ensure correct module path in production.")
    
    # Mock minimal des classes requises si core_utils n'existe pas
    class MapControllerMock:
        def __init__(self, page, config): pass
        def nuke_overlays(self): logging.info("Mocking overlay removal.")
        def switch_category(self, category): logging.info(f"Mocking switch to {category}.")
        
    class CoreUtilsMock:
        MapController = MapControllerMock
        normalize_name = lambda name: name.lower().replace(' ', '_')
        
    core_utils = CoreUtilsMock()

    # Mock de configuration pour l'exécution isolée
    class ConfigMock:
        URL_MAP_VIEWER = "https://meteoburkina.bf/mapviewer/"
        VISUAL_DIR = "rapports_scraping/images_cartes"
        # Catégorie cible pour la capture visuelle (ex: Imagerie Satellite)
        VISUAL_MAP_CATEGORY = "SATELLITE_IMAGERY" 
        HEADLESS_MODE = True
        USER_AGENT = "Mozilla/5.0 (Scraper)"
        BROWSER_TIMEOUT = 90000 
        SELECTOR_MAP_CONTAINER = '#map-container' # Sélecteur du conteneur de la carte pour la capture ciblée
        IMAGE_RENDER_TIMEOUT = 8000 # Ajout de cette constante au mock (en millisecondes)
        SELECTOR_BLOCKERS = ['.modal', '.cookie-popup'] # Nécessaire pour MapController
        SELECTOR_MAP_MENU = ".datasets-menu, .map-layer-panel" # Nécessaire pour MapController
    config = ConfigMock()

# --- Le Scraper ---

class MapVisualScraper:
    """
    Service dédié au scraping visuel (capture d'écran) de couches cartographiques
    spécifiques, comme l'imagerie satellite.
    """

    def __init__(self, headless: bool = None):
        # Initialisation avec les attributs de la config
        self.headless = headless if headless is not None else getattr(config, 'HEADLESS_MODE', True)
        self.base_url = getattr(config, 'URL_MAP_VIEWER')
        self.output_dir = getattr(config, 'VISUAL_DIR')
        self.target_category = getattr(config, 'VISUAL_MAP_CATEGORY')
        self.map_selector = getattr(config, 'SELECTOR_MAP_CONTAINER')
        self.browser_timeout = getattr(config, 'BROWSER_TIMEOUT', 90000)
        self.render_timeout_ms = getattr(config, 'IMAGE_RENDER_TIMEOUT', 5000)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Dossier de sortie visuel créé: {self.output_dir}")

    def _setup_browser_context(self, p: Any) -> BrowserContext:
        """Configure et lance le contexte du navigateur avec des paramètres de robustesse."""
        browser = p.chromium.launch(
            headless=self.headless, 
            args=["--no-sandbox", "--disable-setuid-sandbox"]
        )
        context = browser.new_context(
            user_agent=getattr(config, 'USER_AGENT', "Mozilla/5.0"),
            ignore_https_errors=True,
            viewport={"width": 1280, "height": 720} # Résolution standard pour capture
        )
        return context
    
    def _save_screenshot(self, page: Page) -> Optional[str]:
        """Capture et sauvegarde l'image de la carte."""
        normalized_name = core_utils.normalize_name(self.target_category)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.output_dir, f"map_{normalized_name}_{timestamp}.png")
        
        try:
            # Capture d'écran ciblée sur le conteneur de la carte
            if self.map_selector and self.map_selector != 'body':
                map_element = page.locator(self.map_selector)
                # Vérification de visibilité avant la capture
                if not map_element.is_visible():
                    logger.warning(f"Le conteneur de carte ({self.map_selector}) n'est pas visible. Tentative de capture de toute la page.")
                    # Fallback sur la capture de toute la page si l'élément n'est pas trouvé/visible
                    page.screenshot(path=filename, full_page=False)
                else:
                    map_element.screenshot(path=filename)
            else:
                # Capture de la page entière si le sélecteur est 'body' ou non défini
                page.screenshot(path=filename, full_page=False)

            logger.info(f"💾 Capture d'écran sauvegardée pour {self.target_category} : {filename}")
            return filename
        except Exception as e:
            logger.error(f"Erreur lors de la capture ou de la sauvegarde de l'image : {e}")
            return None


    def scrape_visual(self) -> Dict[str, Any]:
        """
        Lance le scraping visuel (capture d'écran) de la couche cartographique cible.
        """
        context = None
        screenshot_path = None
        
        logger.info(f"Démarrage du scraping visuel pour la catégorie : {self.target_category}")
        
        with sync_playwright() as p:
            try:
                # 1. Configuration et Lancement du Navigateur
                context = self._setup_browser_context(p)
                page = context.new_page()
                
                logger.info(f"Connexion à la carte : {self.base_url}")
                page.goto(
                    self.base_url, 
                    # Utiliser 'networkidle' pour s'assurer que l'imagerie a chargé
                    wait_until="networkidle", 
                    timeout=self.browser_timeout
                )

                # 2. Initialisation et Contrôle de la Carte
                # Note: Le MapController a besoin de la config complète pour connaître les sélecteurs de blocage
                controller = core_utils.MapController(page, config) 
                
                # Attendre que le menu principal soit potentiellement chargé avant le nettoyage
                menu_selector = getattr(config, 'SELECTOR_MAP_MENU', 'body')
                try:
                    page.wait_for_selector(menu_selector, state="visible", timeout=10000)
                except:
                    logger.debug("Menu non trouvé ou timeout, poursuite du nettoyage.")

                controller.nuke_overlays()
                page.wait_for_timeout(2000) # Attendre que le nettoyage prenne effet

                # 3. Activation de la Couche Cible
                logger.info(f"Activation de la couche visuelle : {self.target_category}")
                controller.switch_category(self.target_category) 
                
                # Attente supplémentaire pour le rendu de l'imagerie satellite/radar
                page.wait_for_timeout(self.render_timeout_ms)

                # 4. Capture d'écran
                screenshot_path = self._save_screenshot(page)
                
                if screenshot_path:
                    return {
                        "status": "SUCCESS",
                        "message": f"Capture d'écran de la carte ({self.target_category}) réussie.",
                        "path": screenshot_path
                    }
                else:
                    return {
                        "status": "ERROR",
                        "message": "Capture d'écran échouée ou élément non trouvé.",
                        "path": None
                    }
            
            except PlaywrightTimeoutError as e:
                logger.error(f"Erreur de navigation (timeout) lors du chargement de la carte visuelle: {e}")
                return {"status": "ERROR", "message": "Navigation timeout ou chargement image trop lent.", "path": None}
            except Exception as e:
                logger.error(f"Erreur critique lors du scraping visuel : {e}", exc_info=True)
                return {"status": "ERROR", "message": f"Échec de l'exécution: {e}", "path": None}
            
            finally:
                # 5. Fermeture du navigateur
                if context:
                    try:
                        context.browser.close()
                    except:
                        pass

# Exemple d'exécution pour test
if __name__ == "__main__":
    # Assurez-vous d'avoir une URL valide si vous exécutez ce script
    # Utilisez headless=False pour voir l'exécution
    scraper = MapVisualScraper(headless=True) 
    results = scraper.scrape_visual()
    
    logger.info("\n--- Résultat du Scraping Visuel de Carte ---")
    logger.info(f"Statut: {results['status']}")
    logger.info(f"Message: {results['message']}")
    if results['path']:
        logger.info(f"Chemin du fichier: {results['path']}")