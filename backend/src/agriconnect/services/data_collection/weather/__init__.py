"""
Weather Data Collection

Collecteurs de données météorologiques de sources diverses.

Modules:
- weather_cron: Collecteur périodique avec stockage DB
- weather_forecast: API météo avec prévisions
- documents_meteo: Scraping de bulletins météo PDF/HTML

Usage:
    from backend.services.data_collection.weather import WeatherCronScraper
    
    scraper = WeatherCronScraper()
    scraper.run_for_all_locations()
"""

__version__ = "1.0.0"

__all__ = [
    "weather_cron",
    "weather_forecast",
    "documents_meteo"
]
