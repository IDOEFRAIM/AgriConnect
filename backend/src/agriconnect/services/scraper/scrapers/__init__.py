"""
Individual Scrapers Module

Scrapers spécialisés pour différents types de sources.

Scrapers disponibles:
- GoogleWorkspaceScraper: Google Drive, Docs, Forms
- PdfDownloader: PDF documents with text extraction
- FaoDoiResolver: FAO publications via DOI
- NewsScraper: News articles and blog posts
- DataPlatformScraper: Statistical platforms (INSD, FAO, FEWS)
- TechnicalResourcesExplorer: Technical agriculture resources

Interface commune:
- Tous implémentent une méthode de scraping principale
- Configuration injectable
- Error handling standardisé
- Output organisé par catégorie

Usage:
    from backend.services.scraper.scrapers import GoogleWorkspaceScraper
    
    scraper = GoogleWorkspaceScraper(output_dir="./output")
    result = scraper.scrape_url("https://drive.google.com/...")
"""

from .google_workspace_scraper import GoogleWorkspaceScraper
from .pdf_downloader import PdfDownloader
from .fao_doi_resolver import FaoDoiResolver
from .news_scraper import NewsScraper
from .data_platform_scraper import DataPlatformScraper
from .technical_resources_explorer import TechnicalResourcesExplorer

__version__ = "2.0.0"

__all__ = [
    "GoogleWorkspaceScraper",
    "PdfDownloader",
    "FaoDoiResolver",
    "NewsScraper",
    "DataPlatformScraper",
    "TechnicalResourcesExplorer"
]
