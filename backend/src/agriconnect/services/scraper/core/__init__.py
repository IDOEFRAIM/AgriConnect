"""
Scraper Core Module

Composants fondamentaux pour le système de scraping production-ready.

Modules:
- config: Configuration centralisée (ScraperConfig, SourcesConfig)
- error_handling: Circuit Breaker, Rate Limiter, Retry, Dead Letter Queue
- checkpoint: Checkpoint/Resume pour AWS Lambda idempotence
- logging_config: Structured logging pour CloudWatch
- resource_manager: Gestionnaire de ressources et orchestration

Architecture:
- Dependency Injection partout
- SOLID principles
- Design patterns (Circuit Breaker, Strategy, Template Method)
- AWS Lambda compatible

Usage:
    from backend.services.scraper.core import ScraperConfig, CheckpointManager
    
    config = ScraperConfig(lambda_execution=True)
    checkpoint_mgr = CheckpointManager()
"""

from .config import ScraperConfig, SourcesConfig, get_config
from .error_handling import CircuitBreaker, RateLimiter, retry_with_backoff, DeadLetterQueue
from .checkpoint import CheckpointManager, CheckpointState
from .logging_config import setup_logging, get_logger
from .resource_manager import ResourceManager

__version__ = "2.0.0"

__all__ = [
    # Configuration
    "ScraperConfig",
    "SourcesConfig",
    "get_config",
    
    # Error Handling
    "CircuitBreaker",
    "RateLimiter",
    "retry_with_backoff",
    "DeadLetterQueue",
    
    # Checkpoint
    "CheckpointManager",
    "CheckpointState",
    
    # Logging
    "setup_logging",
    "get_logger",
    
    # Resource Management
    "ResourceManager"
]
