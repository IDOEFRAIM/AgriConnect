"""backend.services.scraper

Lightweight package initializer with lazy imports to avoid circular
imports when the package is imported at application startup (eg. by
FastAPI/uvicorn). Import heavy submodules only when their symbols are
accessed.

Public API remains the same (names listed in ``__all__``), but the
actual imports happen lazily via ``__getattr__``.
"""

__all__ = [
    # Main Orchestrators
    'MasterHarvester',
    'lambda_handler',
    'ScraperOrchestrator',

    # Configuration
    'ScraperConfig',
    'SourcesConfig',
    'get_config',

    # Error Handling
    'CircuitBreaker',
    'RateLimiter',

    # Checkpoint & Resource Management
    'CheckpointManager',
    'ResourceManager',

    # Individual Scrapers
    'GoogleWorkspaceScraper',
    'PdfDownloader',
    'FaoDoiResolver',
    'NewsScraper',
    'DataPlatformScraper',
    'TechnicalResourcesExplorer'
]


def __getattr__(name: str):
    """Lazy import symbols on first access to avoid circular imports."""
    if name in ('MasterHarvester', 'lambda_handler'):
        from .master_harvester import MasterHarvester, lambda_handler
        return MasterHarvester if name == 'MasterHarvester' else lambda_handler

    if name == 'ScraperOrchestrator':
        from .scraper_orchestrator import ScraperOrchestrator
        return ScraperOrchestrator

    if name in ('ScraperConfig', 'SourcesConfig', 'get_config',
                'CircuitBreaker', 'RateLimiter', 'CheckpointManager', 'ResourceManager'):
        from .core import (
            ScraperConfig,
            SourcesConfig,
            get_config,
            CircuitBreaker,
            RateLimiter,
            CheckpointManager,
            ResourceManager,
        )
        return locals()[name]

    if name in ('GoogleWorkspaceScraper', 'PdfDownloader', 'FaoDoiResolver',
                'NewsScraper', 'DataPlatformScraper', 'TechnicalResourcesExplorer'):
        from .scrapers import (
            GoogleWorkspaceScraper,
            PdfDownloader,
            FaoDoiResolver,
            NewsScraper,
            DataPlatformScraper,
            TechnicalResourcesExplorer,
        )
        return locals()[name]

    # Backwards-compatible mappings for legacy names that live in data_collection
    if name == 'DocumentScraper':
        from backend.src.agriconnect.services.data_collection.weather.documents_meteo import DocumentScraper
        return DocumentScraper

    if name == 'WeatherForecastService':
        from backend.src.agriconnect.services.data_collection.weather.weather_forecast import WeatherForecastService
        return WeatherForecastService

    if name == 'SonagessScraper':
        from backend.src.agriconnect.services.data_collection.documents.sonagess import SonagessScraper
        return SonagessScraper

    if name == 'AnamBulletinScraper':
        from backend.src.agriconnect.services.data_collection.documents.fews_pdf_harvester import AnamBulletinScraper
        return AnamBulletinScraper

    raise AttributeError(f"module {__name__} has no attribute {name}")


__version__ = "2.0.0"
__author__ = "Agribot-AI Team"
