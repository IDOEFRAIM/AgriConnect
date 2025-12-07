import logging
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
# CORRECTION CRITIQUE: PlaywrightTimeoutError a été renommé en TimeoutError.
from playwright.sync_api import sync_playwright, Response, Page, TimeoutError, BrowserContext

# Configuration des logs (utilise le logger de l'orchestrateur)
logger = logging.getLogger("MapFeatureScraper")

# Définition de l'alias pour la compatibilité avec le reste du code
PlaywrightTimeoutError = TimeoutError

# --- CORRECTION CRITIQUE: Configuration par défaut en cas d'échec de l'import ---
try:
    # Tentative d'import de la configuration consolidée
    # Ceci est la structure attendue si le script est exécuté dans un framework plus large.
    from config import SCRAPER_CONFIG
    CONFIG_REFERENCE = SCRAPER_CONFIG
except ImportError:
    logger.critical("ATTENTION: config.py ou SCRAPER_CONFIG introuvable. Utilisation d'une configuration par défaut pour la démo/revue.")
    CONFIG_REFERENCE = {
        'URL_MAP_VIEWER': "https://map.example.com/viewer", # URL de substitution si la config est absente
        'RAW_DATA_DIR': "raw_map_data",
        'HEADLESS_MODE': True,
        'BROWSER_TIMEOUT': 60000,
        'USER_AGENT': "Mozilla/5.0 (Playwright)",
        'NETWORK_IDLE_TIMEOUT': 25000, 
        'SELECTOR_MAP_MENU': ".datasets-menu, .map-layer-panel, .sidebar", # Sélecteur par défaut du panneau de couches
        'WATCHLIST_MAP_CATEGORIES': ["Flood Risk", "Hydrology"],
        'SELECTOR_BLOCKERS': ['.modal', '.cookie-popup', '#overlay', '.leaflet-control-container'],
        'KEYWORDS_FANFAR_RISK': ["risk", "flood", "alert", "fanfar", "features"],
        'PROCESSOR_FLOOD_KEYWORDS': {}
    }
    # Création du répertoire de sortie pour le mock, nécessaire pour éviter une erreur I/O
    if not os.path.exists(CONFIG_REFERENCE['RAW_DATA_DIR']):
        os.makedirs(CONFIG_REFERENCE['RAW_DATA_DIR'], exist_ok=True)
# ---------------------------------------------------------------------------------


class MapNetworkInterceptor:
    """
    Espion Réseau : Capture les données vectorielles (GeoJSON).
    """
    def __init__(self, config: Dict[str, Any]):
        self.captured_features = []
        # Utilisation des mots-clés de la config pour la détection
        self.fanfar_keywords = config.get("KEYWORDS_FANFAR_RISK", ["features"])
        self.flood_keywords = config.get("PROCESSOR_FLOOD_KEYWORDS", {})

    def handle_response(self, response: Response):
        """Callback déclenché à chaque réponse du serveur."""
        try:
            # 1. Filtre sur le type MIME JSON
            if "json" not in response.headers.get("content-type", ""):
                return

            # 2. Filtre sur l'URL pour cibler les couches de données
            url = response.url.lower()
            if not any(k in url for k in ["map", "layer", "feature", "dataset", "geojson"]):
                 # Inclure les mots-clés de risque de Fanfar dans l'URL pour plus de robustesse
                 if not any(k in url for k in self.fanfar_keywords):
                     return

            data = response.json()
            
            # 3. Vérification structure GeoJSON standard (FeatureCollection)
            is_geojson = isinstance(data, dict) and data.get("type") == "FeatureCollection"
            
            # 4. Vérification alternative (structure de réponse API de risque/Fanfar)
            # On vérifie si la réponse JSON contient l'un des mots-clés
            is_risk_data = any(k in json.dumps(data).lower() for k in self.fanfar_keywords)
            
            if is_geojson:
                features = data.get("features", [])
                
                # 5. Filtrage : On n'ajoute que les GeoJSON qui contiennent des features
                if features:
                    logger.info(f"📡 GeoJSON reçu ({len(features)} objets) via : {url[-40:]}")
                    self.captured_features.extend(features)
                    
            elif is_risk_data:
                # Si ce n'est pas un GeoJSON strict, on stocke la réponse complète si elle contient des données de risque
                logger.info(f"📡 Donnée Risque API reçue via : {url[-40:]}")
                # Enregistre la réponse complète avec l'URL en tant que métadonnée
                self.captured_features.append({"metadata": url, "data": data})

        except Exception as e:
            # Souvent causé par une réponse qui n'est pas du JSON valide, on ignore silencieusement
            # logger.debug(f"Erreur de lecture de réponse JSON: {e}")
            pass

    def reset(self):
        self.captured_features = []

class MapController:
    """
    Le "Pilote" : Gère l'interface et les blocages via les sélecteurs de config.
    """
    def __init__(self, page: Page, config: Dict[str, Any]):
        self.page = page
        self.config = config
        self.menu_selector = config.get('SELECTOR_MAP_MENU', ".datasets-menu")

    def nuke_overlays(self):
        """
        SOLUTION RADICALE : Supprime via JS tous les éléments bloquants définis dans la config.
        """
        try:
            # Récupération des sélecteurs de blocage depuis la configuration
            blockers = json.dumps(self.config.get('SELECTOR_BLOCKERS', []))
            
            self.page.evaluate(f"""() => {{
                const blockers = {blockers};
                blockers.forEach(selector => {{
                    document.querySelectorAll(selector).forEach(el => el.remove());
                }});
            }}""")
        except Exception as e:
            logger.warning(f"Erreur lors du nettoyage JS des overlays : {e}")

    def switch_category(self, category_name: str):
        """
        Clique sur un onglet de la palette pour charger la couche GeoJSON associée.
        """
        logger.info(f"👉 Recherche catégorie : {category_name}")
        
        # 1. On nettoie le terrain avant de cliquer
        self.nuke_overlays()
        
        try:
            timeout = self.config.get('NETWORK_IDLE_TIMEOUT', 15000)
            
            # Cible : Texte insensible à la casse
            btn = self.page.locator(f"text=/{category_name}/i").first
            
            # 2. Tentative de Clic
            if btn.count() > 0:
                # On force le clic pour ignorer les overlays invisibles
                btn.click(force=True, timeout=5000)
                logger.info(f"Clic forcé sur '{category_name}' réussi. Attente chargement...")
                # time.sleep(3) est conservé car l'activation de la couche peut prendre du temps
                # même après l'événement de clic et avant l'événement réseau.
                time.sleep(3) 
            else:
                logger.error(f"Bouton '{category_name}' introuvable dans le menu.")
                
        except Exception as e:
            logger.error(f"Échec critique du clic sur '{category_name}'. Re-nettoyage et tentative finale. Erreur: {e}")
            self.nuke_overlays()
            try:
                # Tentative finale de clic forcé
                self.page.locator(f"text=/{category_name}/i").first.click(force=True, timeout=5000)
            except:
                pass


class MapFeatureScraper:
    """Service complet pour intercepter les données GeoJSON et autres features vectorielles."""

    def __init__(self, headless: bool = None):
        
        self.config = CONFIG_REFERENCE # Utilisation de la référence globale (mock ou importée)
            
        # Initialisation des variables de la classe
        self.headless = headless if headless is not None else self.config.get('HEADLESS_MODE', True)
        self.base_url = self.config.get('URL_MAP_VIEWER')
        self.output_dir = self.config.get('RAW_DATA_DIR')
        self.targets = self.config.get('WATCHLIST_MAP_CATEGORIES', [])
        
        if not self.base_url or not self.output_dir:
            logger.critical("Configuration manquante (URL_MAP_VIEWER ou RAW_DATA_DIR).")
            raise ValueError("Configuration de base manquante.")

        # Le répertoire a déjà été créé dans le bloc de configuration par défaut,
        # mais on assure ici qu'il existe si l'import initial a réussi.
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)


    def scrape_features(self) -> Dict[str, Any]:
        """Exécute le scraping pour toutes les catégories cibles."""
        logger.info(f"Démarrage de l'interception de features GeoJSON pour {len(self.targets)} cibles.")
        
        all_features = {}
        successful_captures = 0

        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=self.headless, 
                args=["--no-sandbox", "--disable-setuid-sandbox"]
            )
            context = browser.new_context(
                viewport={"width": 1366, "height": 768},
                user_agent=self.config.get('USER_AGENT')
            )
            page = context.new_page()

            # Navigation initiale
            try:
                page.goto(self.base_url, wait_until="domcontentloaded", timeout=self.config.get('BROWSER_TIMEOUT'))
            except PlaywrightTimeoutError as e:
                logger.error(f"Échec de l'accès au site {self.base_url} (Timeout): {e}")
                browser.close()
                return {"status": "ERROR", "message": f"Navigation échouée: Timeout lors de l'accès à l'URL. {e}"}
            except Exception as e:
                logger.error(f"Échec de l'accès au site {self.base_url}: {e}")
                browser.close()
                return {"status": "ERROR", "message": f"Navigation échouée: {e}"}

            controller = MapController(page, self.config)
            
            # --- Attente explicite du menu de la carte pour éviter les sleep() trop longs ---
            menu_selector = self.config.get('SELECTOR_MAP_MENU', ".datasets-menu, .map-layer-panel")
            try:
                page.wait_for_selector(menu_selector, state="visible", timeout=15000)
                logger.info("Menu de la carte détecté. Nettoyage initial des overlays.")
                controller.nuke_overlays()
                # Petite pause après le nettoyage pour laisser le DOM se stabiliser
                time.sleep(2) 
            except PlaywrightTimeoutError:
                logger.warning("Le menu de la carte n'est pas apparu dans les temps. Poursuite avec nettoyage forcé.")
                controller.nuke_overlays()
            except Exception as e:
                logger.warning(f"Erreur lors de l'attente du menu/nettoyage : {e}")


            for category in self.targets:
                interceptor = MapNetworkInterceptor(self.config)
                
                # 1. Attacher l'intercepteur pour la catégorie actuelle
                page.on("response", interceptor.handle_response)
                
                # 2. Changer de catégorie (déclenche le chargement réseau)
                controller.switch_category(category)
                
                # 3. Attendre la stabilité du réseau pour capturer les données
                try:
                    # Utilisation d'un état réseau stable
                    page.wait_for_load_state("networkidle", timeout=self.config.get('NETWORK_IDLE_TIMEOUT'))
                    time.sleep(3) # Attente supplémentaire pour les requêtes asynchrones/rendering
                except PlaywrightTimeoutError:
                    logger.warning(f"Timeout réseau atteint pour {category}. Les données ont pu être capturées même sans 'networkidle'.")
                except Exception as e:
                    logger.warning(f"Erreur inattendue pendant l'attente réseau pour {category}: {e}")
                
                # 4. Détacher l'intercepteur (important pour la boucle)
                page.remove_listener("response", interceptor.handle_response)
                
                # 5. Sauvegarde des données capturées
                if interceptor.captured_features:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = os.path.join(self.output_dir, f"{category.lower().replace(' ', '_')}_{timestamp}.geojson")
                    
                    # Assurer que les features sont dans une FeatureCollection valide avant la sauvegarde
                    # Filtration pour s'assurer que seuls les dictionnaires de type Feature sont inclus, 
                    # ignorant les objets de métadonnées brutes que l'intercepteur peut avoir stockés.
                    geojson_data = {
                        "type": "FeatureCollection",
                        "features": [f for f in interceptor.captured_features if isinstance(f, dict) and f.get('type') == 'Feature'], 
                        "metadata": {
                            "source_url": self.base_url,
                            "category": category,
                            "captured_items": len(interceptor.captured_features),
                            "timestamp": timestamp
                        }
                    }
                    
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(geojson_data, f, ensure_ascii=False, indent=2)
                        
                    all_features[category] = filename
                    successful_captures += 1
                    logger.info(f"💾 {len(geojson_data['features'])} features GeoJSON sauvegardées pour '{category}' dans : {filename}")
                else:
                    logger.warning(f"Aucune feature GeoJSON capturée pour '{category}'.")
            
            browser.close()

        if successful_captures > 0:
            return {
                "status": "SUCCESS",
                "message": f"{successful_captures} couches de features GeoJSON capturées.",
                "results": all_features # Dictionnaire {catégorie: chemin_fichier}
            }
        else:
            return {
                "status": "FAILURE",
                "message": "Aucune donnée GeoJSON n'a pu être capturée pour les cibles définies.",
                "results": {}
            }