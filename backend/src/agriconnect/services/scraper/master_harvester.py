"""
Master Resource Harvester - Orchestration complète du scraping de ressources.

Ce script utilise tous les scrapers spécialisés pour extraire le maximum de contenu
depuis les sources fournies par l'utilisateur.

Workflow:
1. Catégorisation automatique des sources
2. Assignation du scraper approprié
3. Traitement par batches avec checkpoint/resume
4. Gestion d'erreurs robuste (circuit breaker, retry, rate limiting)
5. Génération de rapports détaillés
6. Catalogue centralisé de toutes les ressources

Architecture:
- Dependency Injection pour testabilité
- Configuration centralisée
- Error handling standardisé
- Checkpoint/resume pour AWS Lambda
- Logging structuré pour CloudWatch

Usage:
    python -m backend.services.scraper.master_harvester
    
Lambda:
    from backend.services.scraper.master_harvester import lambda_handler
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import time

from .core.config import ScraperConfig, SourcesConfig, get_config
from .core.error_handling import CircuitBreaker, RateLimiter, retry_with_backoff
from .core.checkpoint import CheckpointManager, CheckpointState
from .core.logging_config import setup_logging, get_logger
from .core.resource_manager import ResourceManager
from .scrapers.google_workspace_scraper import GoogleWorkspaceScraper
from .scrapers.pdf_downloader import PdfDownloader
from .scrapers.fao_doi_resolver import FaoDoiResolver
from .scrapers.news_scraper import NewsScraper
from .scrapers.data_platform_scraper import DataPlatformScraper
from .scrapers.technical_resources_explorer import TechnicalResourcesExplorer

# Initialize structured logging
setup_logging(
    log_level="INFO",
    structured=True,
    console_output=True
)
logger = get_logger(__name__, {"component": "MasterHarvester"})


class MasterHarvester:
    """
    Orchestrateur principal pour le moissonnage de toutes les ressources.
    
    Principes Clean Code:
    - Dependency Injection: Configuration et composants injectés
    - Single Responsibility: Orchestration uniquement
    - Testabilité: Tous les composants mockables
    - Idempotence: Checkpoint/resume pour Lambda
    """

    def __init__(
        self,
        config: Optional[ScraperConfig] = None,
        sources_config: Optional[SourcesConfig] = None,
        resource_manager: Optional[ResourceManager] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        rate_limiter: Optional[RateLimiter] = None
    ):
        """
        Initialize with dependency injection.
        
        Args:
            config: Scraper configuration (from env or default)
            sources_config: Sources URLs configuration
            resource_manager: Resource processing manager
            checkpoint_manager: Checkpoint for resume capability
            circuit_breaker: Circuit breaker for fault tolerance
            rate_limiter: Rate limiter to respect API limits
        """
        # Configuration
        self.config = config or get_config()
        self.sources_config = sources_config or SourcesConfig()
        
        # Output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Managers
        self.resource_manager = resource_manager or ResourceManager(
            output_dir=str(self.output_dir)
        )
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        
        # Error handling
        self.circuit_breaker = circuit_breaker or CircuitBreaker(
            failure_threshold=self.config.circuit_breaker_threshold,
            recovery_timeout=self.config.circuit_breaker_timeout
        )
        self.rate_limiter = rate_limiter or RateLimiter(
            rate_limit=self.config.rate_limit_per_second,
            burst_size=self.config.rate_limit_per_second * 2
        )
        
        # Initialize scrapers with config injection
        self.scrapers_map = self._initialize_scrapers()
        
        logger.info(
            "MasterHarvester initialized",
            extra={
                "output_dir": str(self.output_dir),
                "max_retries": self.config.max_retries,
                "rate_limit": self.config.rate_limit_per_second,
                "lambda_mode": self.config.lambda_execution
            }
        )
    
    def _initialize_scrapers(self) -> Dict:
        """Initialize all scrapers with proper configuration."""
        return {
            'google_workspace_links': GoogleWorkspaceScraper(
                output_dir=str(self.output_dir / "google_workspace")
            ),
            'pdf_documents': PdfDownloader(
                output_dir=str(self.output_dir / "pdfs")
            ),
            'fao_publications_doi': FaoDoiResolver(
                output_dir=str(self.output_dir / "fao_publications")
            ),
            'news_and_articles': NewsScraper(
                output_dir=str(self.output_dir / "news_articles")
            ),
            'statistical_platforms': DataPlatformScraper(
                output_dir=str(self.output_dir / "data_platforms")
            ),
            'technical_agriculture_resources': TechnicalResourcesExplorer(
                output_dir=str(self.output_dir / "technical_resources"),
                max_depth=2,
                max_pages=30
            )
        }

    def harvest_all(self, session_id: Optional[str] = None) -> Dict:
        """
        Lance le moissonnage complet avec checkpoint/resume.
        
        Args:
            session_id: Identifier for this scraping session (auto-generated if None)
        
        Returns:
            Statistics dictionary with execution metrics
        """
        logger.info("="*80)
        logger.info("🌾 DÉMARRAGE DU MOISSONNAGE DE RESSOURCES AGRICOLES 🌾")
        logger.info("="*80)
        
        start_time = datetime.now()
        
        # Generate session ID if not provided
        if session_id is None:
            session_id = f"harvest_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        # Load or create checkpoint
        sources_dict = self.sources_config.to_dict()
        checkpoint = self.checkpoint_manager.load_or_create(session_id, sources_dict)
        
        logger.info(
            "Session started",
            extra={
                "session_id": session_id,
                "execution_count": checkpoint.execution_count,
                "total_urls": checkpoint.total_urls,
                "pending_urls": checkpoint.total_urls - checkpoint.processed_urls
            }
        )
        
        # Process all categories with checkpoint
        stats = self._process_with_checkpoint(checkpoint, sources_dict)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        checkpoint.total_duration_seconds += duration
        
        # Save final checkpoint
        self.checkpoint_manager.save(checkpoint)
        
        # Generate report if complete
        if self.checkpoint_manager.is_complete():
            report_path = self.resource_manager.generate_report()
            logger.info(f"📄 Rapport détaillé: {report_path}")
        
        # Log summary
        self._log_summary(stats, duration, checkpoint)
        
        return stats
    
    def _process_with_checkpoint(
        self,
        checkpoint: CheckpointState,
        sources: Dict[str, List[str]]
    ) -> Dict:
        """
        Process sources with checkpoint tracking.
        
        Args:
            checkpoint: Current checkpoint state
            sources: Dictionary of category -> URLs
        
        Returns:
            Processing statistics
        """
        stats = {
            'total_sources': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'by_type': {}
        }
        
        for category, urls in sources.items():
            if category not in self.scrapers_map:
                logger.warning(f"No scraper for category: {category}")
                continue
            
            scraper = self.scrapers_map[category]
            
            # Get pending URLs for this category
            pending_urls = checkpoint.pending_urls.get(category, [])
            
            if not pending_urls:
                logger.info(f"Category {category} already completed")
                continue
            
            logger.info(
                f"Processing category: {category}",
                extra={
                    "category": category,
                    "pending_count": len(pending_urls),
                    "total_count": len(urls)
                }
            )
            
            category_stats = {
                'total': len(urls),
                'success': len(checkpoint.completed_urls.get(category, set())),
                'failed': len(checkpoint.failed_urls.get(category, set()))
            }
            
            # Process URLs in batches
            batch_size = self.config.batch_size if self.config.lambda_execution else len(pending_urls)
            
            for url in pending_urls[:batch_size]:
                # Rate limiting
                self.rate_limiter.acquire()
                
                # Circuit breaker check
                if not self.circuit_breaker.can_execute():
                    logger.warning(
                        "Circuit breaker OPEN - skipping remaining URLs",
                        extra={"category": category}
                    )
                    break
                
                # Process URL with retry
                success = self._process_url_with_retry(url, scraper, category)
                
                # Update checkpoint
                self.checkpoint_manager.mark_completed(category, url, success)
                
                # Update circuit breaker
                if success:
                    self.circuit_breaker.record_success()
                    category_stats['success'] += 1
                    stats['successful'] += 1
                else:
                    self.circuit_breaker.record_failure()
                    category_stats['failed'] += 1
                    stats['failed'] += 1
                
                stats['total_sources'] += 1
                
                # Save checkpoint after each URL (Lambda idempotence)
                if self.config.lambda_execution:
                    self.checkpoint_manager.save()
            
            stats['by_type'][category] = category_stats
        
        return stats
    
    @retry_with_backoff(max_retries=3, backoff_factor=1.0)
    def _process_url_with_retry(self, url: str, scraper, category: str) -> bool:
        """
        Process single URL with retry logic.
        
        Args:
            url: URL to process
            scraper: Scraper instance
            category: Category name
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(
                "Processing URL",
                extra={
                    "url": url,
                    "category": category,
                    "scraper": scraper.__class__.__name__
                }
            )
            
            start = time.time()
            
            # Call scraper (interface varies by scraper type)
            if hasattr(scraper, 'scrape_url'):
                result = scraper.scrape_url(url)
            elif hasattr(scraper, 'download_pdf'):
                result = scraper.download_pdf(url)
            elif hasattr(scraper, 'resolve_and_download'):
                result = scraper.resolve_and_download(url)
            else:
                logger.error(f"Unknown scraper interface: {scraper.__class__.__name__}")
                return False
            
            duration_ms = (time.time() - start) * 1000
            
            logger.info(
                "URL processed successfully",
                extra={
                    "url": url,
                    "category": category,
                    "duration_ms": round(duration_ms, 2),
                    "status": "success"
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "URL processing failed",
                extra={
                    "url": url,
                    "category": category,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "status": "failed"
                },
                exc_info=True
            )
            return False
    
    def _log_summary(self, stats: Dict, duration: float, checkpoint: CheckpointState):
        """Log execution summary."""
        logger.info("\n" + "="*80)
        logger.info("📊 RÉSUMÉ DU MOISSONNAGE")
        logger.info("="*80)
        logger.info(f"Durée totale: {duration:.1f} secondes ({duration/60:.1f} minutes)")
        logger.info(f"Sources totales: {stats['total_sources']}")
        logger.info(f"✅ Succès: {stats['successful']}")
        logger.info(f"❌ Échecs: {stats['failed']}")
        logger.info(f"⏭️  Ignorées: {stats['skipped']}")
        
        logger.info("\n📂 Détail par catégorie:")
        for category, data in stats.get('by_type', {}).items():
            rate = (data['success'] / data['total'] * 100) if data['total'] > 0 else 0
            logger.info(f"  • {category}: {data['success']}/{data['total']} ({rate:.0f}%)")
        
        # Checkpoint progress
        progress = self.checkpoint_manager.get_progress()
        logger.info(
            "\n🔄 État du checkpoint:",
            extra=progress
        )
        
        logger.info("\n" + "="*80)
        if self.checkpoint_manager.is_complete():
            logger.info("✨ MOISSONNAGE TERMINÉ ✨")
        else:
            logger.info("⏸️  MOISSONNAGE EN COURS - Reprise possible")
        logger.info("="*80)


    def harvest_category(self, category: str, session_id: Optional[str] = None) -> Dict:
        """
        Moissonne une catégorie spécifique uniquement.
        
        Args:
            category: Category name to harvest
            session_id: Session identifier (auto-generated if None)
        
        Returns:
            Statistics dictionary
        """
        sources_dict = self.sources_config.to_dict()
        
        if category not in sources_dict:
            raise ValueError(
                f"Catégorie inconnue: {category}. "
                f"Catégories disponibles: {list(sources_dict.keys())}"
            )
        
        logger.info(f"🎯 Moissonnage de la catégorie: {category}")
        
        # Create single-category sources
        filtered_sources = {category: sources_dict[category]}
        
        # Use harvest_all with filtered sources
        original_sources = self.sources_config
        self.sources_config = SourcesConfig(**filtered_sources)
        
        try:
            stats = self.harvest_all(session_id)
            logger.info(
                f"✅ Catégorie {category} terminée",
                extra={
                    "category": category,
                    "successful": stats['successful'],
                    "total": stats['total_sources']
                }
            )
            return stats
        finally:
            self.sources_config = original_sources


def main():
    """Point d'entrée principal pour exécution locale."""
    logger.info("Starting local execution")
    
    harvester = MasterHarvester()
    
    # Option 1: Tout moissonner
    stats = harvester.harvest_all()
    
    # Option 2: Moissonner une catégorie spécifique (décommenter si besoin)
    # stats = harvester.harvest_category('pdf_documents')
    
    return stats


def lambda_handler(event, context):
    """
    AWS Lambda handler pour exécution périodique.
    
    Args:
        event: Lambda event (EventBridge scheduled event)
        context: Lambda context with execution metadata
    
    Returns:
        Response with execution statistics
    """
    logger.info(
        "Lambda execution started",
        extra={
            "request_id": context.request_id,
            "function_name": context.function_name,
            "remaining_time_ms": context.get_remaining_time_in_millis()
        }
    )
    
    try:
        # Initialize harvester with Lambda mode
        config = get_config()
        config.lambda_execution = True
        
        harvester = MasterHarvester(config=config)
        
        # Session ID from event or generate
        session_id = event.get('session_id', f"lambda_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Execute harvest with checkpoint/resume
        stats = harvester.harvest_all(session_id=session_id)
        
        # Prepare response
        response = {
            "statusCode": 200,
            "body": {
                "message": "Harvest completed successfully",
                "session_id": session_id,
                "statistics": stats,
                "progress": harvester.checkpoint_manager.get_progress(),
                "execution_time_ms": context.get_remaining_time_in_millis()
            }
        }
        
        logger.info(
            "Lambda execution completed",
            extra=response["body"]
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "Lambda execution failed",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "request_id": context.request_id
            },
            exc_info=True
        )
        
        return {
            "statusCode": 500,
            "body": {
                "message": "Harvest failed",
                "error": str(e),
                "error_type": type(e).__name__
            }
        }


if __name__ == "__main__":
    main()
