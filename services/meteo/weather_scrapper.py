import logging
import json
import time
import os
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional
from playwright.sync_api import sync_playwright, Response, Page

# --- IMPORT CONFIGURATION ---
try:
    from ... import config
except ImportError:
    # Fallback pour exécution isolée
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        import config
    except ImportError:
        # Mock de secours (Miroir de config.py pour tests sans fichier)
        class ConfigMock:
            URL_MAP_VIEWER = "https://meteoburkina.bf/mapviewer/"
            CAPTURES_DIR = "rapports/captures"
            SELECTOR_MAP_MENU = ".datasets-menu"
            SELECTOR_MAP_SIDEBAR = ['.sidebar', '.left-panel', '.control-panel', '.ol-control', '.floating-panel', '#left-panel']
            SELECTOR_BLOCKERS = ['.ReactModal__Overlay', '.ReactModalPortal', '.modal-overlay', '.intro-popup']
            HEADLESS_MODE = True
            USER_AGENT = "Mozilla/5.0"
            BROWSER_TIMEOUT = 60000
            NETWORK_IDLE_TIMEOUT = 15000
            WATCHLIST_CATEGORIES = ["ALERTES", "PLUIE", "INONDATIONS", "SÉCHERESSE"]
        config = ConfigMock()

# Configuration des logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("services.map_scraper")

class MapNetworkInterceptor:
    """
    Espion Réseau : Capture les données vectorielles (GeoJSON).
    """
    def __init__(self):
        self.captured_features = []

    def handle_response(self, response: Response):
        """Callback déclenché à chaque réponse du serveur."""
        try:
            # Filtre strict sur le type MIME JSON
            if "json" not in response.headers.get("content-type", ""):
                return

            # Optimisation : On ne parse que si ça ressemble à du GeoJSON
            url = response.url.lower()
            # On cible les couches de données (map, layer, feature, dataset)
            if not any(k in url for k in ["map", "layer", "feature", "dataset"]):
                return

            try:
                # On lit tout le texte (plus de limite [:1000] pour ne rien rater)
                data = response.json()
                
                # Vérification structure GeoJSON standard
                if isinstance(data, dict) and data.get("type") == "FeatureCollection":
                    features = data.get("features", [])
                    if features:
                        logger.info(f"📡 Données reçues ({len(features)} objets) via : {url[-40:]}")
                        self.captured_features.extend(features)
            except:
                pass 
        except:
            pass

    def reset(self):
        self.captured_features = []

class MapController:
    """
    Le "Pilote" : Gère l'interface et les blocages via les sélecteurs de config.
    """
    def __init__(self, page: Page):
        self.page = page
        # Récupération du sélecteur de menu depuis la config
        self.menu_selector = getattr(config, 'SELECTOR_MAP_MENU', ".datasets-menu")

    def nuke_overlays(self):
        """
        SOLUTION RADICALE : Supprime via JS tous les éléments bloquants définis dans la config.
        """
        try:
            # Injection sûre de la liste Python vers JS
            blockers = json.dumps(getattr(config, 'SELECTOR_BLOCKERS', []))
            
            self.page.evaluate(f"""() => {{
                const blockers = {blockers};
                blockers.forEach(selector => {{
                    document.querySelectorAll(selector).forEach(el => el.remove());
                }});
            }}""")
        except Exception as e:
            logger.warning(f"Erreur lors du nettoyage JS : {e}")

    def switch_category(self, category_name: str):
        """
        Clique sur un onglet de la palette.
        Gère le cas où un popup bloque le clic.
        """
        logger.info(f"👉 Recherche catégorie : {category_name}")
        
        # 1. On nettoie le terrain avant de cliquer
        self.nuke_overlays()
        
        try:
            # Attente du menu (Timeout depuis config)
            timeout = getattr(config, 'NETWORK_IDLE_TIMEOUT', 15000)
            self.page.wait_for_selector(self.menu_selector, timeout=timeout)
            
            # Cible : Texte insensible à la casse
            btn = self.page.locator(f"text=/{category_name}/i").first
            
            # 2. Tentative de Clic
            if btn.count() > 0:
                # On force le clic pour ignorer les overlays invisibles
                btn.click(force=True, timeout=5000)
                logger.info(f"Clic forcé sur '{category_name}' réussi.")
                time.sleep(3) # Temps de chargement de la couche
            else:
                logger.error(f"Bouton '{category_name}' introuvable dans le menu.")
                
        except Exception as e:
            logger.error(f"Échec sur '{category_name}'. Tentative de sauvetage...")
            self.nuke_overlays()
            try:
                self.page.locator(f"text=/{category_name}/i").first.click(force=True)
            except:
                pass

    def clean_interface(self):
        """Masque les menus pour une capture d'écran propre."""
        try:
            # Liste des éléments à masquer (Sidebar + Menu principal)
            sidebars = getattr(config, 'SELECTOR_MAP_SIDEBAR', [])
            to_hide = sidebars + [self.menu_selector]
            
            js_array = json.dumps(to_hide)
            
            self.page.evaluate(f"""() => {{
                const selectors = {js_array};
                selectors.forEach(s => {{
                    document.querySelectorAll(s).forEach(el => el.style.display = 'none');
                }});
            }}""")
        except:
            pass

class MapVisualizerService:
    """Service complet pour explorer la carte météo."""

    def __init__(self, headless: bool = None):
        # Si headless n'est pas spécifié, on prend celui de la config
        if headless is None:
            self.headless = getattr(config, 'HEADLESS_MODE', True)
        else:
            self.headless = headless
            
        # URL depuis config
        self.base_url = getattr(config, 'URL_MAP_VIEWER', "https://meteoburkina.bf/mapviewer/")
        
        # Dossier de sortie depuis config
        self.output_dir = getattr(config, 'CAPTURES_DIR', "rapports/captures")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def capture_category(self, category: str) -> Optional[Dict[str, Any]]:
        interceptor = MapNetworkInterceptor()
        
        with sync_playwright() as p:
            # Lancement navigateur avec User-Agent config
            browser = p.chromium.launch(
                headless=self.headless, 
                args=["--no-sandbox", "--disable-setuid-sandbox"]
            )
            context = browser.new_context(
                viewport={"width": 1366, "height": 768},
                user_agent=getattr(config, 'USER_AGENT', "Mozilla/5.0")
            )
            page = context.new_page()

            page.on("response", interceptor.handle_response)

            logger.info(f"Connexion : {self.base_url}")
            try:
                timeout = getattr(config, 'BROWSER_TIMEOUT', 60000)
                page.goto(self.base_url, wait_until="domcontentloaded", timeout=timeout)
            except Exception as e:
                logger.error(f"Site HS : {e}")
                return None

            controller = MapController(page)
            
            # Séquence d'initialisation
            time.sleep(5) 
            controller.nuke_overlays() # Destruction immédiate des popups

            # Action
            interceptor.reset()
            controller.switch_category(category)
            
            # Attente Données
            logger.info("Chargement couche...")
            try:
                timeout_net = getattr(config, 'NETWORK_IDLE_TIMEOUT', 15000)
                page.wait_for_load_state("networkidle", timeout=timeout_net)
            except:
                pass 
            time.sleep(3)

            # Capture
            controller.clean_interface()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            filename = os.path.join(self.output_dir, f"carte_{category.lower()}_{timestamp}.png")
            page.screenshot(path=filename)
            logger.info(f"📸 Capture OK : {filename}")

            browser.close()

            return {
                "category": category,
                "snapshot_path": filename,
                "features_count": len(interceptor.captured_features),
                "features_sample": interceptor.captured_features[:1]
            }

if __name__ == "__main__":
    # Pour le test, on force le mode visuel (False) sauf si surchargé
    service = MapVisualizerService(headless=False)
    
    # Utilisation de la liste définie dans la config si dispo
    TARGETS = getattr(config, 'WATCHLIST_CATEGORIES', ["ALERTES"])
    
    print(f"--- Démarrage Scan Optimisé ({len(TARGETS)} cibles) ---")
    
    for cat in TARGETS:
        print(f"\n>>> Traitement : {cat}")
        try:
            res = service.capture_category(cat)
            if res:
                print(f"   ✅ Image: {res['snapshot_path']}")
                print(f"   📍 Vecteurs: {res['features_count']}")
        except Exception as e:
            print(f"   ⚠️ Erreur: {e}")