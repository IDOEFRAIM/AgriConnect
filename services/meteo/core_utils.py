import logging
import json
import re 
from typing import Any, List
from playwright.sync_api import Response, Page, TimeoutError as PlaywrightTimeoutError

# NOTE: Dans un environnement réel, ce fichier serait importé à côté de 'config'.
# On suppose ici que 'config' est disponible ou importé par les modules qui utilisent celui-ci.

logger = logging.getLogger("core_utils")

def normalize_name(name: str) -> str:
    """Nettoyage et normalisation d'une chaîne pour l'utiliser dans un nom de fichier."""
    # Supprimer les accents
    name = re.sub(r'[éèêë]', 'e', name)
    name = re.sub(r'[àâä]', 'a', name)
    name = re.sub(r'[ôö]', 'o', name)
    name = re.sub(r'[ûü]', 'u', name)
    name = re.sub(r'[îï]', 'i', name)
    # Remplacer les espaces et les caractères non alphanumériques par des underscores
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Réduire les doubles underscores
    name = re.sub(r'_{2,}', '_', name).strip('_')
    return name.lower()


class MapNetworkInterceptor:
    """
    Espion Réseau (Network Interceptor): Écoute les réponses du serveur pour capturer
    spécifiquement les données GeoJSON.
    """
    def __init__(self):
        # Liste pour stocker les objets GeoJSON (features) capturés.
        self.captured_features = []

    def handle_response(self, response: Response):
        """Callback déclenché à chaque réponse du serveur."""
        try:
            if response.status != 200:
                return

            # Vérifie si la réponse est JSON
            content_type = response.headers.get("content-type", "").lower()
            if "json" not in content_type:
                return

            url = response.url.lower()
            # Cible stricte : ne traite que les URLs suggérant des données cartographiques
            if not any(k in url for k in ["map", "layer", "feature", "dataset", "geojson"]):
                return

            try:
                data = response.json()
                
                # Vérification de la structure GeoJSON standard (FeatureCollection)
                if isinstance(data, dict) and data.get("type") == "FeatureCollection":
                    features = data.get("features", [])
                    if features:
                        logger.info(f"📡 GeoJSON reçu ({len(features)} features) : {url[-50:]}")
                        self.captured_features.extend(features)
            except json.JSONDecodeError:
                # Ignorer si ce n'est pas un JSON valide ou si c'est un JSON sans GeoJSON
                pass 
        except Exception:
            # Ignorer les erreurs de réponse non critiques
            pass

    def reset(self):
        """Réinitialise la liste des features capturées pour une nouvelle opération de scraping."""
        self.captured_features = []


class MapController:
    """
    Contrôleur Playwright: Gère l'interaction avec l'interface de la carte 
    (clic sur les menus, suppression des overlays, masquage des éléments).
    """
    def __init__(self, page: Page, config: Any):
        self.page = page
        self.config = config
        self.menu_selector = getattr(self.config, 'SELECTOR_MAP_MENU', ".datasets-menu")

    def nuke_overlays(self):
        """Supprime via JS tous les éléments bloquants (popups, modales) définis dans config.py."""
        try:
            blockers = getattr(self.config, 'SELECTOR_BLOCKERS', [])
            blockers_json = json.dumps(list(set(blockers)))
            
            # Injection de code JS pour supprimer les éléments
            self.page.evaluate(f"""() => {{
                const blockers = {blockers_json};
                blockers.forEach(selector => {{
                    document.querySelectorAll(selector).forEach(el => {{
                        if (el.parentNode) el.remove(); 
                    }});
                }});
            }}""")
            logger.info("🗑️ Overlays et popups initiaux nukés.")
        except Exception as e:
            logger.warning(f"Erreur lors du nettoyage JS des overlays : {e}")

    def switch_category(self, category_name: str):
        """Clique sur un onglet ou un bouton de catégorie spécifique."""
        logger.info(f"👉 Tentative de sélection de la catégorie : {category_name}")
        self.nuke_overlays() 
        try:
            # Utilisation d'un sélecteur basé sur le texte, forcé pour contourner les masques
            self.page.click(
                f"text=/{category_name}/i", 
                force=True, 
                timeout=5000 
            )
            logger.info(f"Clic forcé sur '{category_name}' réussi.")
            self.page.wait_for_timeout(3000) # Attente post-clic pour le déclenchement des requêtes réseau
        except PlaywrightTimeoutError:
            logger.error(f"Le bouton '{category_name}' n'a pas été trouvé ou n'est pas cliquable.")
        except Exception as e:
            logger.error(f"Échec critique lors du switch de catégorie : {e}")

    def clean_interface_for_screenshot(self):
        """Masque les menus et panneaux pour une capture d'écran propre (pour les services visuels)."""
        try:
            sidebars = getattr(self.config, 'SELECTOR_MAP_SIDEBAR', [])
            to_hide = list(set(sidebars + [self.menu_selector]))
            js_array = json.dumps(to_hide)
            
            # Injection de code JS pour masquer les éléments
            self.page.evaluate(f"""() => {{
                const selectors = {js_array};
                selectors.forEach(s => {{
                    document.querySelectorAll(s).forEach(el => {{
                        el.style.visibility = 'hidden'; 
                        el.style.pointerEvents = 'none'; 
                    }});
                }});
            }}""")
            logger.info("🎨 Interface masquée pour la capture d'écran.")
        except Exception as e:
            logger.warning(f"Erreur lors du masquage de l'interface : {e}")