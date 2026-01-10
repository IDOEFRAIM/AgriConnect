import logging
import json
import time
from typing import List, Dict, Any, Optional
from playwright.sync_api import sync_playwright, Page, TimeoutError as PlaywrightTimeoutError

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("WeatherForecastService")

# URL Cible
BASE_URL = "https://meteoburkina.bf/le-climat-de-nos-villes/"
PLAYWRIGHT_TIMEOUT = 60000  # 60 secondes pour être large

class WeatherForecastService:
    """
    Service pour scraper les données météo par ville en utilisant Playwright.
    Extrait les données climatiques (températures, précipitations) depuis les graphiques Highcharts.
    """

    def __init__(self, headless: bool = True):
        self.headless = headless
        self.structured_forecasts: List[Dict[str, Any]] = []

    def _extract_highcharts_data(self, page: Page) -> Dict[str, Any]:
        """
        Tente d'extraire les données brutes des graphiques Highcharts présents sur la page.
        """
        try:
            # Script pour récupérer les données de tous les graphiques Highcharts sur la page
            data = page.evaluate("""() => {
                const charts = [];
                if (window.Highcharts && window.Highcharts.charts) {
                    window.Highcharts.charts.forEach((chart, index) => {
                        if (chart) {
                            const seriesData = chart.series.map(s => ({
                                name: s.name,
                                data: s.data.map(p => ({
                                    category: p.category,
                                    y: p.y,
                                    x: p.x
                                }))
                            }));
                            charts.push({
                                title: chart.title ? chart.title.textStr : `Graphique ${index}`,
                                subtitle: chart.subtitle ? chart.subtitle.textStr : '',
                                series: seriesData
                            });
                        }
                    });
                }
                return charts;
            }""")
            return data
        except Exception as e:
            logger.warning(f"Impossible d'extraire les données Highcharts: {e}")
            return {}

    def _scrape_forecasts(self, page: Page) -> List[Dict[str, Any]]:
        """
        Scrape les données pour chaque ville disponible dans le menu déroulant.
        """
        forecasts = []
        try:
            # 1. Attendre et localiser le menu déroulant des villes
            select_selector = "#city_select"
            try:
                page.wait_for_selector(select_selector, state="attached", timeout=15000)
            except PlaywrightTimeoutError:
                logger.error(f"Sélecteur {select_selector} introuvable. La structure de la page a peut-être changé.")
                return []
            
            # Récupérer toutes les options de villes
            options = page.locator(f"{select_selector} option").all()
            cities_to_scrape = []
            for opt in options:
                val = opt.get_attribute("value")
                label = opt.inner_text().strip()
                if val: 
                    cities_to_scrape.append((val, label))
            
            logger.info(f"Villes trouvées dans le menu : {len(cities_to_scrape)}")

            # Limiter le nombre de villes pour le test/démo si nécessaire, sinon tout scraper
            # cities_to_scrape = cities_to_scrape[:3] 

            # 2. Itérer sur chaque ville
            for city_val, city_name in cities_to_scrape:
                try:
                    logger.info(f"Scraping pour : {city_name}")
                    # Sélectionner la ville
                    page.select_option(select_selector, city_val)
                    
                    # Attendre que le contenu se mette à jour (AJAX)
                    # On attend un peu pour laisser le temps au JS de mettre à jour le graphique
                    page.wait_for_timeout(2000) 
                    
                    # Extraction des données Highcharts
                    charts_data = self._extract_highcharts_data(page)
                    
                    # Construction du résumé textuel
                    content_summary = f"Données climatiques pour {city_name}.\n"
                    if charts_data:
                        for chart in charts_data:
                            content_summary += f"Graphique: {chart.get('title', 'N/A')}\n"
                            for serie in chart.get('series', []):
                                content_summary += f"- {serie.get('name', 'Série')}: {len(serie.get('data', []))} points de données.\n"
                    else:
                        content_summary += "Aucune donnée graphique extraite.\n"

                    # Création de l'objet résultat standardisé
                    forecast_entry = {
                        "url": BASE_URL,
                        "type": "weather_data",
                        "title": f"Climat - {city_name}",
                        "content": content_summary,
                        "metadata": {
                            "city": city_name,
                            "raw_data": charts_data
                        }
                    }
                    
                    forecasts.append(forecast_entry)
                    
                except Exception as e:
                    logger.error(f"Erreur lors du traitement de {city_name}: {e}")
                    continue

            if not forecasts:
                logger.warning("Aucune donnée extraite via le menu déroulant.")
                
            return forecasts

        except Exception as e:
            logger.error(f"Erreur inattendue lors du scraping des prévisions: {e}")
            return []

    def scrape_forecast(self) -> Dict[str, Any]:
        """
        Méthode principale pour orchestrer le scraping.
        """
        self.structured_forecasts = []
        
        with sync_playwright() as p:
            # Lancement du navigateur
            browser = p.chromium.launch(headless=self.headless)
            context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
            page = context.new_page()

            logger.info(f"Navigation vers la page des prévisions: {BASE_URL}")
            try:
                page.goto(BASE_URL, wait_until="domcontentloaded", timeout=PLAYWRIGHT_TIMEOUT)
                
                # Gestion des popups éventuels (cookies, pubs, etc.)
                try:
                    page.locator(".dialog-close-button, .close-popup, #cookie-accept").click(timeout=2000)
                except:
                    pass

                self.structured_forecasts = self._scrape_forecasts(page)
                
            except Exception as e:
                logger.error(f"Erreur globale de scraping: {e}")
            finally:
                browser.close()

        if not self.structured_forecasts:
            return {
                "status": "ERROR",
                "message": "Échec de l'extraction des données météo.",
                "results": []
            }
        else:
            return {
                "status": "SUCCESS",
                "message": f"{len(self.structured_forecasts)} villes collectées.",
                "results": self.structured_forecasts
            }

def run_scrape_forecast() -> Dict[str, Any]:
    return WeatherForecastService(headless=True).scrape_forecast()

if __name__ == "__main__":
    # Test avec headless=False pour voir ce qui se passe
    service = WeatherForecastService(headless=True)
    res = service.scrape_forecast()
    print(json.dumps(res, indent=2, ensure_ascii=False))
