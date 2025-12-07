import time
import json
import re
import os
from datetime import datetime
from typing import Optional, Dict

# --- DÉPENDANCES SIMULÉES (MOCK) ---
# Ces classes simulent les dépendances externes de votre environnement.
try:
    from flood_data_processor import FloodDataProcessor
    from playwright.sync_api import sync_playwright, Route, Response
    class Config:
        HEADLESS_MODE = True
        URL_FANFAR_PIV = "https://fanfar.eu/fr/piv/"
        KEYWORDS_FANFAR_RISK = ["severity", "return_period"]
        USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        BROWSER_TIMEOUT = 60000
        CAPTURES_DIR = "rapports/captures"
    config = Config()

    class Logger:
        def info(self, msg):
            print(f"[INFO] {msg}")
        def warning(self, msg):
            print(f"[WARN] {msg}")
        def error(self, msg):
            print(f"[ERROR] {msg}")
    logger = Logger()

    class FloodRiskResult:
        def __init__(self, location: str, risk_level: str, risk_score: int, source_type: str, details: str, station_name: str = "N/A", screenshot_path: str = None):
            self.location = location
            self.risk_level = risk_level
            self.risk_score = risk_score
            self.source_type = source_type
            self.details = details
            self.station_name = station_name
            self.screenshot_path = screenshot_path
        def __repr__(self):
            return f"FloodRiskResult(Location: {self.location}, Risk: {self.risk_level} ({self.risk_score}), Source: {self.source_type})"
            

   
except ImportError:
    print("ATTENTION: Playwright n'est pas importé. Le code ci-dessous est incomplet/non exécutable sans ces dépendances.")
    # Définir des classes mock minimales si l'import Playwright échoue
    


class FanfarUltimateScraper:
    def __init__(self, headless: bool = getattr(config, 'HEADLESS_MODE', True)):
        self.headless = headless
        # URL depuis la config
        self.target_url = getattr(config, 'URL_FANFAR_PIV', "https://fanfar.eu/fr/piv/")
        self.captured_payloads = []

    def _intercept_network(self, route: Route):
        """Bloque les images/fonts pour accélérer le chargement."""
        if route.request.resource_type in ["image", "media", "font", "stylesheet"]:
            route.abort()
        else:
            route.continue_()

    def _monitor_response(self, response: Response):
        """Espionne le trafic réseau pour trouver des données hydrologiques."""
        try:
            if response.request.resource_type in ["image", "media", "font", "stylesheet", "script"]:
                return

            url = response.url.lower()
            if "google" in url or "analytics" in url: return

            content_txt = ""
            try:
                # Tente de récupérer le texte de la réponse. Peut échouer si c'est un binaire ou une réponse vide.
                content_txt = response.text()
            except:
                return 

            # Mots-clés techniques depuis la config
            keywords = getattr(config, 'KEYWORDS_FANFAR_RISK', ["severity", "return_period"])
            
            if any(k in content_txt.lower() for k in keywords):
                is_json = False
                data_obj = None
                try:
                    data_obj = json.loads(content_txt)
                    is_json = True
                except:
                    pass

                self.captured_payloads.append({
                    "url": url,
                    "type": "JSON" if is_json else "TEXT/HTML",
                    "content": data_obj if is_json else content_txt,
                    "timestamp": time.time()
                })
        except Exception:
            # Ignorer les exceptions générales de monitoring pour ne pas bloquer le scraping
            pass

    def get_risk(self, location: str) -> FloodRiskResult:
        """Méthode principale pour obtenir le risque d'une ville."""
        self.captured_payloads = [] 
        
        with sync_playwright() as p:
            # Lancement navigateur avec User-Agent de la config
            browser = p.chromium.launch(
                headless=self.headless, 
                args=["--no-sandbox", "--disable-setuid-sandbox"]
            )
            context = browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent=getattr(config, 'USER_AGENT', "Mozilla/5.0")
            )
            page = context.new_page()

            # Attachement des gestionnaires de réseau
            page.route("**/*", self._intercept_network)
            page.on("response", self._monitor_response)

            logger.info(f"Chargement de FANFAR pour : {location}...")
            try:
                timeout = getattr(config, 'BROWSER_TIMEOUT', 60000)
                page.goto(self.target_url, wait_until="domcontentloaded", timeout=timeout)
            except Exception as e:
                logger.error(f"Le site ne répond pas : {e}")
                return FloodRiskResult(location, "Erreur Site", 0, "ERROR", str(e))

            # --- GESTION COOKIES ---
            try:
                cookie_texts = ["Accepter", "Accept", "J'accepte", "Tout accepter"]
                for txt in cookie_texts:
                    # Recherche d'un bouton ou lien contenant l'un des textes d'acceptation
                    btn = page.locator(f"button:has-text('{txt}', exact=True), a:has-text('{txt}', exact=True)")
                    if btn.count() > 0 and btn.first.is_visible():
                        btn.first.click(timeout=1000)
                        page.wait_for_timeout(500) # Petit délai pour l'animation
                        break
            except:
                pass

            # --- RECHERCHE STANDARD ---
            logger.info("Recherche de la localité...")
            station_name = "Non détectée"

            try:
                # Cherche la barre de recherche
                search_box = page.locator("input[type='search'], input[placeholder*='Search'], input[placeholder*='Recherche']").first
                search_box.fill(location)
                time.sleep(2.0) # Délai pour l'apparition des suggestions
                
                # Cherche les suggestions (Leaflet Geocoder ou autres)
                suggestions = page.locator(".leaflet-control-geocoder-alternatives li, .search-results li")
                try:
                    # Attend que la première suggestion soit visible (max 3s)
                    suggestions.first.wait_for(state="visible", timeout=3000)
                except:
                    pass

                if suggestions.count() > 0 and suggestions.first.is_visible():
                    best_match = suggestions.first
                    txt = best_match.inner_text()
                    if txt: station_name = txt
                    else: station_name = location
                    
                    logger.info(f"Sélection : {station_name}")
                    best_match.click()
                else:
                    # Si aucune suggestion visible, valide la recherche par Entrée
                    search_box.press("Enter")
                    station_name = location
                    logger.info("Validation par Entrée.")
                
                # Attente du zoom et du chargement des données (5s)
                page.wait_for_timeout(5000)
                
            except Exception as e:
                logger.warning(f"Problème recherche : {e}")

            # --- INTERACTION CARTE (TIR DE BARRAGE) ---
            logger.info("Tentative de clic sur la station...")
            vp = page.viewport_size
            # Centre de la fenêtre d'affichage
            cx, cy = vp["width"] / 2, vp["height"] / 2
            
            # Essais de clic : centre, puis légers offsets
            offsets = [(0, 0), (0, -15), (0, 15), (-15, 0), (15, 0)]
            popup_found = False
            
            for i, (ox, oy) in enumerate(offsets):
                if popup_found: break
                page.mouse.click(cx + ox, cy + oy)
                page.wait_for_timeout(2000) # Délai pour laisser le popup apparaître
                
                # Vérifie la présence du popup Leaflet standard
                if page.locator(".leaflet-popup-content").is_visible():
                    logger.info(f"Popup détecté au clic #{i+1}")
                    popup_found = True
            
            page.wait_for_timeout(2000)

            # --- ANALYSE DES RÉSULTATS BRUTS ---
            risk_level = "Inconnu"
            risk_score = 0
            source_type = "NONE"
            raw_details = ""
            screenshot_file = None
            
            # 1. Analyse Réseau (Priorité)
            api_result = self._analyze_captured_network()
            if api_result:
                risk_level = api_result['label']
                risk_score = api_result['score']
                source_type = api_result['source']
                raw_details = str(api_result['raw'])[:200]

            # 2. Analyse Visuelle (Fallback)
            ui_text = ""
            try:
                # Tente de localiser le popup
                popup = page.locator(".leaflet-popup-content").first
                if popup.is_visible():
                    # Utilisation du dossier de config si possible
                    captures_dir = getattr(config, 'CAPTURES_DIR', "rapports/captures")
                    if not os.path.exists(captures_dir):
                        os.makedirs(captures_dir, exist_ok=True)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    # Nettoie le nom de la localité pour le nom de fichier
                    clean_loc = re.sub(r'[^a-zA-Z0-9_]', '', location.replace(' ', '_')) 
                    screenshot_file = f"{captures_dir}/{clean_loc}_{timestamp}.png"
                    popup.screenshot(path=screenshot_file) # Capture le popup
                    
                    ui_text = popup.inner_text()
                    
                    # Tente d'affiner le nom de la station à partir du titre du popup
                    try:
                        title = popup.locator("h3, strong, b, .title").first
                        if title.is_visible():
                            station_name = title.inner_text()
                    except:
                        pass
            except:
                pass

            # 3. Fallback d'interprétation si rien trouvé par l'API
            if source_type == "NONE":
                # ... (Logique d'interprétation du texte de l'UI) ...
                if not ui_text:
                    try:
                        # Récupère le texte du body si rien dans le popup
                        body_txt = page.locator("body").inner_text()
                        if "skip to content" not in body_txt.lower():
                            ui_text = body_txt
                        else:
                            ui_text = "MENU_ONLY_DETECTED" # Texte trop générique
                    except:
                        pass
                
                # Interprète le texte brut de l'UI
                ui_res = self._interpret_text(location, ui_text, "UI_TEXT")
                
                # Mise à jour des résultats par fallback
                risk_level = ui_res.risk_level
                risk_score = ui_res.risk_score
                source_type = ui_res.source_type
                raw_details = ui_res.details
                
                # Si le nom de la station n'a pas été trouvé plus tôt
                if station_name == "Non détectée" and "MENU_ONLY" not in ui_text:
                    station_name = "Zone Carte (Approx)"

            browser.close()
            
            # --- INTEGRATION DU PROCESSEUR DE DONNÉES (FloodDataProcessor) ---
            # C'est ici que le processeur prend le relais pour garantir la qualité de la sortie
            
            processor = FloodDataProcessor()
            
            raw_data = {
                "location": location,
                "station_name": station_name,
                "risk_level": risk_level,
                "risk_score": risk_score,
                "source_type": source_type,
                "raw_details": raw_details,
                "screenshot_path": screenshot_file
            }
            
            # Traitement des données
            processed_data = processor.process_flood_risk(raw_data)
            
            # Mise à jour des variables finales avec les données traitées (si le processeur les a modifiées)
            # Ceci permet au processeur de standardiser les noms de stations, les niveaux, etc.
            final_risk_level = processed_data.get('risk_level', risk_level)
            final_risk_score = processed_data.get('risk_score', risk_score)
            final_station_name = processed_data.get('station_name', station_name)
            final_details = processed_data.get('raw_details', raw_details)
            final_source_type = processed_data.get('source_type', source_type)
            final_screenshot_path = processed_data.get('screenshot_path', screenshot_file)

            # Création du résultat final à partir des données traitées
            return FloodRiskResult(
                location=location,
                risk_level=final_risk_level,
                risk_score=final_risk_score,
                source_type=final_source_type,
                details=final_details,
                station_name=final_station_name,
                screenshot_path=final_screenshot_path
            )

    def _analyze_captured_network(self) -> Optional[Dict]:
        """Analyse les paquets capturés."""
        for packet in reversed(self.captured_payloads):
            content = str(packet['content']).lower()
            
            # Filtre HTML strict pour se concentrer sur les données
            if isinstance(packet['content'], str) and ("<!doctype" in content[:200] or "<html" in content[:200]):
                continue
            
            score = 0
            label = "Normal"
            found = False
            
            # Détection des niveaux de risque basés sur des mots-clés
            if "niveau de sévérité 3" in content or "severity level 3" in content or "return period > 30" in content:
                score = 3
                label = "Critique (Très Haut)"
                found = True
            elif "niveau de sévérité 2" in content or "severity level 2" in content or "return period > 5" in content:
                score = 2
                label = "Élevé"
                found = True
            elif "niveau de sévérité 1" in content or "severity level 1" in content or "return period > 2" in content:
                score = 1
                label = "Modéré"
                found = True
            elif "normal" in content or "risk-free" in content:
                score = 0
                label = "Faible / Normal"
                found = True
            
            if found:
                return {
                    "score": score, 
                    "label": label, 
                    "source": f"API_{packet['type']}", 
                    "raw": packet['content']
                }
        return None

    def _interpret_text(self, location, text, source) -> FloodRiskResult:
        """Traduit le texte brut de l'interface utilisateur en données structurées."""
        text = text.lower()
        level = "Inconnu"
        score = 0
        detail = text[:100]

        if "très haut" in text or "très élevé" in text or ">30 ans" in text:
            level = "Critique (Très Haut)"
            score = 3
        elif "haut" in text or "élevé" in text or ">5 ans" in text:
            level = "Élevé"
            score = 2
        elif "modéré" in text or ">2 ans" in text:
            level = "Modéré"
            score = 1
        elif len(text) > 20:
             if "menu_only_detected" in text or "skip to content" in text:
                 level = "Erreur (Cible manquée)"
                 score = 0
                 detail = "Le clic sur la carte n'a pas ouvert de popup de données."
             else:
                 level = "Faible / Normal"
                 score = 0 
        
        return FloodRiskResult(location, level, score, source, detail)