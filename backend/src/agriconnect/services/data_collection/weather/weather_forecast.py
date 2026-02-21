
import logging
import json
import os
import sys
import time
from typing import List, Dict, Any, Optional, Callable
from playwright.sync_api import sync_playwright, Page, TimeoutError as PlaywrightTimeoutError


# --- Orchestrateur avec config centralisée ---
class WeatherForecastOrchestrator:
    """
    Orchestrateur pour le scraping météo (WeatherForecastService).
    Permet d'exécuter le scraping, de logger, et de sauvegarder les résultats.
    """
    DEFAULT_CONFIG = {
        "BASE_URL": "https://meteoburkina.bf/le-climat-de-nos-villes/",
        "PLAYWRIGHT_TIMEOUT": 60000,  # ms
        "HEADLESS": True,
        "SELECTOR_CITY": "#city_select",
        "WAIT_CITY_SELECTOR": 15000,  # ms
        "WAIT_GRAPH_UPDATE": 2000,    # ms
        "LOG_FILE": "weather_forecast_orchestrator.log",
    }

    def __init__(self, headless: bool = True, config: dict = None):
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        if headless is not None:
            self.config["HEADLESS"] = headless
        os.makedirs(os.path.dirname(self.config["LOG_FILE"]) or '.', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config["LOG_FILE"], mode='w', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger("WeatherForecastOrchestrator")
        self.forecast_scraper = WeatherForecastService(self.config, logger=self.logger)

    def run(self) -> Dict[str, Any]:
        self.logger.info("===== DÉMARRAGE DU SCRAPING MÉTÉO =====")
        start_time = time.time()
        try:
            result = self.forecast_scraper.scrape_forecast()
            status = result.get('status', 'UNKNOWN')
            self.logger.info(f"Statut: {status}, Message: {result.get('message', 'N/A')}")
            self.logger.info(f"Nombre de villes: {len(result.get('results', []))}")
        except Exception as e:
            self.logger.error(f"Erreur critique lors du scraping météo: {e}", exc_info=True)
            result = {"status": "CRITICAL_ERROR", "message": str(e), "results": []}
        duration = time.time() - start_time
        self.logger.info(f"Durée totale: {duration:.2f} secondes.")
        self.logger.info("===== FIN DU SCRAPING MÉTÉO =====")
        return result


class WeatherForecastService:
    """
    Service pour scraper les données météo par ville en utilisant Playwright.
    Extrait les données climatiques (températures, précipitations) depuis les graphiques Highcharts.
    """
    DEFAULT_CONFIG = {
        "BASE_URL": "https://meteoburkina.bf/le-climat-de-nos-villes/",
        "PLAYWRIGHT_TIMEOUT": 60000,  # ms
        "HEADLESS": True,
        "SELECTOR_CITY": "#city_select",
        "WAIT_CITY_SELECTOR": 15000,  # ms
        "WAIT_GRAPH_UPDATE": 2000,    # ms
        "LOG_FILE": "weather_forecast_orchestrator.log",
    }
    def __init__(self, config: dict=DEFAULT_CONFIG, logger=None):
        self.config = config.copy()
        self.headless = self.config["HEADLESS"]
        self.structured_forecasts: List[Dict[str, Any]] = []
        self.logger = logger or logging.getLogger("WeatherForecastOrchestrator")

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
            self.logger.warning(f"Impossible d'extraire les données Highcharts: {e}")
            return {}

    def _scrape_forecasts(self, page: Page) -> List[Dict[str, Any]]:
        """
        Scrape les données pour chaque ville disponible dans le menu déroulant.
        """
        forecasts = []
        try:
            # 1. Attendre et localiser le menu déroulant des villes
            select_selector = self.config["SELECTOR_CITY"]
            try:
                page.wait_for_selector(select_selector, state="attached", timeout=self.config["WAIT_CITY_SELECTOR"])
            except PlaywrightTimeoutError:
                self.logger.error(f"Sélecteur {select_selector} introuvable. La structure de la page a peut-être changé.")
                return []
            # Récupérer toutes les options de villes
            options = page.locator(f"{select_selector} option").all()
            cities_to_scrape = []
            for opt in options:
                val = opt.get_attribute("value")
                label = opt.inner_text().strip()
                if val: 
                    cities_to_scrape.append((val, label))
            self.logger.info(f"Villes trouvées dans le menu : {len(cities_to_scrape)}")

            # Limiter le nombre de villes pour le test/démo si nécessaire, sinon tout scraper
            # cities_to_scrape = cities_to_scrape[:3] 

            # 2. Itérer sur chaque ville
            for city_val, city_name in cities_to_scrape:
                try:
                    self.logger.info(f"Scraping pour : {city_name}")
                    # Sélectionner la ville
                    page.select_option(select_selector, city_val)
                    
                    # Attendre que le contenu se mette à jour (AJAX)
                    # On attend un peu pour laisser le temps au JS de mettre à jour le graphique
                    page.wait_for_timeout(self.config["WAIT_GRAPH_UPDATE"]) 
                    
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
                        "url": self.config["BASE_URL"],
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
                    self.logger.error(f"Erreur lors du traitement de {city_name}: {e}")
                    continue

            if not forecasts:
                self.logger.warning("Aucune donnée extraite via le menu déroulant.")
                
            return forecasts

        except Exception as e:
            self.logger.error(f"Erreur inattendue lors du scraping des prévisions: {e}")
            return []

    def scrape_forecast(self) -> Dict[str, Any]:
        """
        Méthode principale pour orchestrer le scraping.
        """
        self.structured_forecasts = []
        

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
            page = context.new_page()

            self.logger.info(f"Navigation vers la page des prévisions: {self.config['BASE_URL']}")
            try:
                page.goto(self.config["BASE_URL"], wait_until="domcontentloaded", timeout=self.config["PLAYWRIGHT_TIMEOUT"])
                try:
                    page.locator(".dialog-close-button, .close-popup, #cookie-accept").click(timeout=2000)
                except:
                    pass
                self.structured_forecasts = self._scrape_forecasts(page)
            except Exception as e:
                self.logger.error(f"Erreur globale de scraping: {e}")
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
    def insert_to_postgres(self, data: list):
        import psycopg2
        from psycopg2.extras import Json
        
        # Utilise des variables d'environnement en prod !
        DB_PARAMS = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'dbname': os.getenv('DB_NAME', 'agriconnect'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASS', 'postgres')
        }

        query = """
            INSERT INTO weather_forecast (city, title, content, raw_data, url, type, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (city, title) 
            DO UPDATE SET 
                content = EXCLUDED.content,
                raw_data = EXCLUDED.raw_data,
                updated_at = NOW();
        """

        try:
            with psycopg2.connect(**DB_PARAMS) as conn:
                with conn.cursor() as cur:
                    # Préparation des données pour une exécution groupée (plus rapide)
                    values = [
                        (
                            entry.get('metadata', {}).get('city'),
                            entry.get('title'),
                            entry.get('content'),
                            Json(entry.get('metadata', {}).get('raw_data')),
                            entry.get('url'),
                            entry.get('type')
                        ) for entry in data
                    ]
                    # Utilisation de executemany pour la performance
                    cur.executemany(query, values)
                    conn.commit()
        except Exception as e:
            self.logger.error(f"Erreur SQL critique : {e}")
            raise # On remonte l'erreur pour que le worker sache qu'il a échoué     
          
    def execute_production_sync(self, output_json: str = "backend/sources/raw_data/weather_forecast_results.json") -> Dict[str, Any]:
        """
        Méthode pour la production : exécute le scraping, sauvegarde en JSON et insère en base PostgreSQL.
        """
        self.logger.info("Démarrage du cycle de synchronisation météo production...")
        start_time = time.time()
        results = self.run()
        duration = time.time() - start_time

        # Sauvegarde JSON
        try:
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Résultats sauvegardés dans {output_json}")
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde JSON : {e}")

        # Insertion en base PostgreSQL (à adapter selon ton schéma)
        try:
            self.insert_to_postgres(results.get("results", []))
            self.logger.info("Insertion en base PostgreSQL réussie.")
        except Exception as e:
            self.logger.error(f"Erreur lors de l'insertion en base PostgreSQL : {e}")

        if results.get("status") == "SUCCESS":
            self.logger.info(f"Synchronisation réussie en {duration:.2f}s.")
            return {
                "timestamp": time.time(),
                "count": len(results.get("results", [])),
                "data": results.get("results")
            }
        else:
            self.logger.error(f"Échec de la synchronisation : {results.get('message')}")
            return results
    
    