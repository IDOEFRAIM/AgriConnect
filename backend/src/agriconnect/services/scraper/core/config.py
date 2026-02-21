"""
Configuration centralisée pour le système de scraping.

Externalisée pour faciliter les déploiements Lambda et l'environnement-specific config.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class ScraperConfig:
    """Configuration globale du scraping."""
    
    # Paths
    output_dir: Path = Path("backend/sources/raw_data")
    checkpoint_dir: Path = Path("backend/sources/checkpoints")
    log_dir: Path = Path("backend/sources/logs")
    
    # Performance
    max_workers: int = 3  # Concurrent scrapers
    request_timeout: int = 30  # seconds
    max_retries: int = 3
    retry_backoff_factor: float = 2.0  # Exponential backoff: 1s, 2s, 4s
    
    # Rate limiting
    rate_limit_calls: int = 10  # Max calls per window
    rate_limit_window: int = 60  # seconds
    
    # Lambda-specific
    lambda_timeout_buffer: int = 60  # Reserve 60s before Lambda timeout
    checkpoint_interval: int = 50  # Save checkpoint every N URLs
    
    # Retry policy
    retry_on_status_codes: List[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])
    backoff_max_delay: int = 300  # Max 5 minutes between retries
    
    # Circuit breaker
    circuit_breaker_threshold: int = 5  # Failures before opening circuit
    circuit_breaker_timeout: int = 300  # Wait 5 min before retry after circuit open
    
    # Scraper-specific settings
    pdf_max_size_mb: int = 50
    technical_explorer_max_depth: int = 2
    technical_explorer_max_pages: int = 30
    
    # User agent rotation
    user_agents: List[str] = field(default_factory=lambda: [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
    ])
    
    @classmethod
    def from_env(cls) -> 'ScraperConfig':
        """Load config from environment variables (for Lambda)."""
        return cls(
            max_workers=int(os.getenv('SCRAPER_MAX_WORKERS', '3')),
            request_timeout=int(os.getenv('SCRAPER_TIMEOUT', '30')),
            max_retries=int(os.getenv('SCRAPER_MAX_RETRIES', '3')),
            checkpoint_interval=int(os.getenv('SCRAPER_CHECKPOINT_INTERVAL', '50'))
        )


@dataclass
class SourcesConfig:
    """Configuration des sources à scraper."""
    
    # URLs catégorisées par type
    google_workspace: List[str] = field(default_factory=list)
    pdfs: List[str] = field(default_factory=list)
    fao_dois: List[str] = field(default_factory=list)
    news: List[str] = field(default_factory=list)
    data_platforms: List[str] = field(default_factory=list)
    technical_resources: List[str] = field(default_factory=list)
    
    @classmethod
    def load_default(cls) -> 'SourcesConfig':
        """Charge la configuration par défaut avec toutes les URLs."""
        return cls(
            google_workspace=[
                "https://share.google/QbJCh2ygUQQBlGUtb",
                "https://share.google/0sJ6jsMgez1sI3Ye5",
                "https://share.google/JFsAeoTXBfuNtqTEu",
                "https://docs.google.com/forms/d/e/1FAIpQLSduR9VwiyApSfLsXGEu6oklvzNXOkYF-S6m0IV1zu1P3TsDVw/viewform"
            ],
            pdfs=[
                "https://afristat.org/wp-content/uploads/2022/04/NotesCours_Agri.pdf",
                "https://mita.coraf.org/assets/files/fiches/Mita--milismi9507.pdf",
                "https://reca-niger.org/IMG/pdf/culture_du_mil_et_contraintes_niger_2019.pdf",
                "https://www.fair-sahel.org/content/download/4680/35605/version/1/file/Rapport+FAIR+06+-+Int%C3%A9gration+AE+Burkina+03.pdf"
            ],
            fao_dois=[
                "https://doi.org/10.4060/cd3185en",
                "https://doi.org/10.4060/cd4965en",
                "https://doi.org/10.4060/cd4313en",
                "https://doi.org/10.4060/cc8166en",
                "https://doi.org/10.4060/cd7304en",
                "https://doi.org/10.4060/cb9479fr",
                "https://doi.org/10.4060/cb1447fr"
            ],
            news=[
                "https://lefaso.net/spip.php?article141676",
                "https://lefaso.net/spip.php?article141797",
                "https://www.sidwaya.info/developpement-de-lagriculture-a-yancheng-lempreinte-de-la-modernisation-a-la-chinoise/",
                "https://www.cirad.fr/les-actualites-du-cirad/actualites/2023/les-mils-cereales-pour-une-agriculture-resiliente",
                "https://www.agri-mutuel.com/documents/",
                "https://fr.wikipedia.org/wiki/Agriculture_au_Burkina_Faso"
            ],
            data_platforms=[
                "https://microdata.insd.bf/index.php/catalog/83",
                "https://microdata.insd.bf/index.php/home",
                "https://www.fao.org/in-action/agrisurvey/access-to-data/burkina-faso/en",
                "https://help.fews.net/fde/v1/burkina-faso-data-book",
                "https://catalog.data.gov/dataset/burkina-faso-compact-diversified-agriculture-and-water-management"
            ],
            technical_resources=[
                # Ressources techniques agricoles
                "https://cnrada.org/fiche-nuisibles/mil-mildiou-ou-lepre-du-mil/",
                "https://www.fao.org/family-farming/detail/fr/c/472595/",
                "https://www.fao.org/4/x0490e/x0490e04.htm",  # NOUVEAU
                "https://www.usgs.gov/apps/croplands/documents",
                "https://www.oecd.org/fr/publications/agriculture-alimentation-et-emploi-en-afrique-de-l-ouest_56d463a9-fr.html",
                "https://www.cahiersagricultures.fr/articles/cagri/full_html/2020/01/cagri200020s/cagri200020s.html",
                "https://reseau-far.com/pays/burkina-faso/",
                "https://earlywarning.usgs.gov/fews/climate-workshop/page2/",  # NOUVEAU
                "https://www.chc.ucsb.edu/data/chirps",  # NOUVEAU
                
                # Plateformes de monitoring et données
                "https://agromonitoring.com/",  # NOUVEAU
                "https://smap.jpl.nasa.gov/",  # NOUVEAU
                "https://bibliocilss.pariis.net/",  # NOUVEAU
                "https://www.soilgrids.org/",  # NOUVEAU
                "https://www.taranis.com/",  # NOUVEAU
                
                # Certification et standards
                "https://certificat.ecocert.com/",  # NOUVEAU
                "https://www.agrisource.org/",  # NOUVEAU
            ]
        )
    
    def to_dict(self) -> Dict[str, List[str]]:
        """Convert to dictionary format for backward compatibility."""
        return {
            "google_workspace_links": self.google_workspace,
            "pdf_documents": self.pdfs,
            "fao_publications_doi": self.fao_dois,
            "news_and_articles": self.news,
            "statistical_platforms": self.data_platforms,
            "technical_agriculture_resources": self.technical_resources
        }


def get_config() -> ScraperConfig:
    """Helper to get configuration from environment."""
    return ScraperConfig.from_env()
