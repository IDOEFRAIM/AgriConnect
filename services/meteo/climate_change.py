import csv
import json
import os
import logging
from typing import List, Dict, Any, Optional
from playwright.sync_api import sync_playwright, Page, TimeoutError as PlaywrightTimeoutError

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ClimatScraper")

class ClimatScraper:
    """
    Classe pour scraper les données climatiques mensuelles (Température, Précipitation)
    des villes du Burkina Faso en utilisant Playwright pour interagir avec
    les graphiques Highcharts sur le site de Meteo Burkina.
    """
    URL_CLIMAT = "https://meteoburkina.bf/le-climat-de-nos-villes/"
    MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    def __init__(self, headless: bool = True):
        """
        Initialise le scraper.

        Args:
            headless (bool): Exécuter Playwright en mode sans tête (True par défaut).
        """
        self.headless = headless
        self.structured_data: Dict[str, Any] = {}
        logger.info("ClimatScraper initialisé (headless=%s)", self.headless)

    # --- Nettoyeurs / Utilitaires Statiques ---

    @staticmethod
    def _nettoyer_nom_ville(nom: str) -> str:
        """Nettoie et normalise le nom d'une ville."""
        return nom.strip().lower().replace("  ", " ").replace("-", " ").title()

    @staticmethod
    def _nettoyer_nom_indicateur(nom: str) -> str:
        """Nettoie et normalise le nom d'un indicateur climatique."""
        nom = nom.strip().lower()
        if "minimale" in nom:
            return "Temp. min"
        elif "maximale" in nom:
            return "Temp. max"
        elif "précipitation" in nom:
            return "Précipitations"
        return nom.title()

    @staticmethod
    def _serie_valide(data: List[Optional[float]]) -> bool:
        """Vérifie si la série de données contient au moins une valeur non-None."""
        return any(d is not None for d in data)

    def _ensure_checked(self, page: Page, selector: str) -> None:
        """S'assure qu'un élément (checkbox) est coché."""
        try:
            cb = page.query_selector(selector)
            if cb and not cb.is_checked():
                cb.click()
                page.wait_for_timeout(500)
        except PlaywrightTimeoutError:
            logger.warning("Timeout lors de l'interaction avec le sélecteur: %s", selector)
        except Exception as e:
            logger.error("Erreur lors de la vérification du checkbox: %s", e)

    # --- Logique de Scraping Playwright ---

    def _scrape_data(self) -> Dict[str, Any]:
        """Exécute la logique Playwright pour extraire les données Highcharts par ville."""
        all_data = {}

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            page = browser.new_page()
            
            try:
                page.goto(self.URL_CLIMAT, wait_until="domcontentloaded", timeout=30000)
                page.wait_for_selector("#city_select", timeout=10000)
            except PlaywrightTimeoutError:
                logger.error("Délai de chargement dépassé ou sélecteur #city_select non trouvé.")
                browser.close()
                return {}

            ville_elements = page.query_selector_all("#city_select option:not([value=''])") # Exclure l'option par défaut vide

            logger.info("%d villes détectées. Démarrage de la boucle de scraping.", len(ville_elements))

            for ville in ville_elements:
                ville_nom_raw = ville.inner_text()
                ville_nom = self._nettoyer_nom_ville(ville_nom_raw)
                ville_value = ville.get_attribute("value")

                if not ville_value:
                    continue

                logger.info("Traitement de la ville: %s", ville_nom)

                try:
                    # 1. Sélectionner la ville
                    page.select_option("#city_select", ville_value)
                    page.wait_for_timeout(1000) # Attendre que le graphique se mette à jour
                    
                    # 2. S'assurer que les indicateurs sont sélectionnés
                    self._ensure_checked(page, "text=Température minimale")
                    self._ensure_checked(page, "text=Température maximale")
                    self._ensure_checked(page, "text=Précipitation")
                    page.wait_for_timeout(500)

                    # 3. Exécuter le JS pour extraire les séries Highcharts
                    chart_data = page.evaluate("""
                        () => {
                            // Highcharts.charts est l'array global des instances de graphiques
                            const charts = Highcharts.charts;
                            return charts.map(chart => {
                                // Filtrer les graphiques non initialisés
                                if (!chart || !chart.series) return null;
                                return chart.series.map(s => ({
                                    name: s.name,
                                    // Extraction des valeurs Y (la méthode Highcharts est plus fiable)
                                    data: s.data.map(point => point.y)
                                }));
                            });
                        }
                    """)

                    # Nettoyage et aplatissement des données du graphique
                    chart_data_flat = [s for chart in chart_data if chart for s in chart]
                    all_data[ville_nom] = chart_data_flat
                
                except Exception as e:
                    logger.error("Erreur critique lors du scraping de %s: %s", ville_nom, e)

            browser.close()
            logger.info("Scraping Playwright terminé.")
            return all_data

    # --- Logique de Structuration ---

    def _structure_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Structure les données brutes par Ville -> Indicateur -> Mois."""
        structured_data = {}
        for ville, series_list in raw_data.items():
            structured_data[ville] = {}
            for serie in series_list:
                name = self._nettoyer_nom_indicateur(serie["name"])
                data = serie["data"]
                
                if not self._serie_valide(data):
                    continue

                structured_data[ville][name] = {
                    self.MONTHS[i]: (data[i] if i < len(data) and data[i] is not None else "NA")
                    for i in range(len(self.MONTHS))
                }
        logger.info("Structuration des données terminée.")
        return structured_data

    # --- Méthode Principale (Renommée) ---

    # RENOMMÉ de run_scrape() à scrape_forecast()
    def scrape_forecast(self) -> Dict[str, Any]:
        """
        Exécute le scraping complet, la structuration et stocke les résultats.
        Cette méthode est l'entrée principale attendue par l'orchestrateur.

        Returns:
            Dict[str, Any]: Les données structurées par ville.
        """
        logger.info("Démarrage de la méthode scrape_forecast...")
        raw_data = self._scrape_data()
        self.structured_data = self._structure_data(raw_data)
        
        if not self.structured_data:
            logger.warning("Aucune donnée climatique n'a été structurée.")
            
        # L'orchestrateur attend un dictionnaire de résultat standard
        if self.structured_data:
            return {
                "status": "SUCCESS", 
                "message": f"{len(self.structured_data)} villes ont été analysées.",
                "results": self.structured_data
            }
        else:
            return {
                "status": "ERROR", 
                "message": "Échec de l'extraction des données climatiques.",
                "results": {}
            }


    # --- Méthodes d'Exportation ---
    # Ces méthodes sont conservées mais non appelées dans l'exemple principal.

    def save_to_csv(self, filepath: str = "csv/climat.csv") -> None:
        """Sauvegarde les données structurées dans un fichier CSV."""
        if not self.structured_data:
            logger.warning("Aucune donnée structurée à exporter en CSV.")
            return
            
        try:
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                header = ["Ville", "Indicateur"] + self.MONTHS
                writer.writerow(header)

                for ville, indicateurs in self.structured_data.items():
                    for indicateur, valeurs in indicateurs.items():
                        # Assure que l'ordre des mois est respecté
                        row = [ville, indicateur] + [valeurs.get(m, "NA") for m in self.MONTHS] 
                        writer.writerow(row)
            
            logger.info("✅ Fichier CSV généré : %s", filepath)
        except Exception as e:
            logger.error("Erreur lors de la sauvegarde en CSV: %s", e)

    def save_to_json(self, filepath: str = "json/climat.json") -> None:
        """Sauvegarde les données structurées dans un fichier JSON."""
        if not self.structured_data:
            logger.warning("Aucune donnée structurée à exporter en JSON.")
            return

        try:
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.structured_data, f, indent=2, ensure_ascii=False)
            
            logger.info("✅ Fichier JSON généré : %s", filepath)
        except Exception as e:
            logger.error("Erreur lors de la sauvegarde en JSON: %s", e)


if __name__ == "__main__":
    # --- Exemple d'utilisation ---
    
    # 1. Créer une instance de scraper (mode headless=False pour voir l'exécution)
    scraper = ClimatScraper(headless=False)
    
    # 2. Exécuter le scraping et la structuration, et récupérer les données
    # MODIFIÉ pour utiliser la nouvelle fonction
    climat_data_result = scraper.scrape_forecast()

    climat_data = climat_data_result.get("results", {})

    if climat_data:
        logger.info("\n--- Aperçu des données (première ville) ---")
        first_city = list(climat_data.keys())[0]
        # Affichage du résultat, prêt à être utilisé par un autre module ou script
        print(json.dumps({first_city: climat_data[first_city]}, indent=2, ensure_ascii=False))

        # 3. Ne pas sauvegarder, mais ces appels seraient disponibles si nécessaire:
        # scraper.save_to_csv()
        # scraper.save_to_json()
        
        # On affiche juste un message pour confirmer le succès
        logger.info("\nScraping réussi. Les données sont disponibles dans la variable 'climat_data'.")
    else:
        logger.error("Le scraping n'a retourné aucune donnée. Vérifiez la connexion ou les sélecteurs.")