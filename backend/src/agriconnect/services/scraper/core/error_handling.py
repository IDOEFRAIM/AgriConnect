"""
Gestion robuste des erreurs et retry logic pour le scraping.

Implémente:
- Retry avec exponential backoff
- Circuit breaker pattern
- Rate limiting
- Dead letter queue pour échecs persistants
"""
import time
import logging
from typing import Callable, Any, Optional, TypeVar, List
from functools import wraps
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class RetryStats:
    """Statistiques de retry pour monitoring."""
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0


class CircuitBreaker:
    """
    Circuit breaker pattern pour éviter les cascading failures.
    
    States:
    - CLOSED: Normal operation
    - OPEN: Failures threshold exceeded, stop trying
    - HALF_OPEN: Testing if service recovered
    """
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.timeout = timeout  # seconds before trying again after circuit opens
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker HALF_OPEN, testing recovery")
            else:
                raise CircuitBreakerOpen(
                    f"Circuit breaker OPEN. Wait {self.timeout}s after {self.last_failure_time}"
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time passed to retry."""
        if self.last_failure_time is None:
            return True
        return datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout)
    
    def _on_success(self):
        """Reset circuit breaker on success."""
        self.failure_count = 0
        self.state = "CLOSED"
        logger.debug("Circuit breaker reset to CLOSED")
    
    def _on_failure(self):
        """Increment failure count and potentially open circuit."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(
                f"Circuit breaker OPEN after {self.failure_count} consecutive failures"
            )


class RateLimiter:
    """
    Token bucket rate limiter.
    
    Prevents overwhelming target servers.
    """
    
    def __init__(self, calls: int = 10, window: int = 60):
        """
        Args:
            calls: Max number of calls per window
            window: Time window in seconds
        """
        self.calls = calls
        self.window = window
        self.timestamps: deque = deque(maxlen=calls)
    
    def acquire(self):
        """Wait if rate limit would be exceeded."""
        now = time.time()
        
        # Remove old timestamps outside window
        while self.timestamps and self.timestamps[0] < now - self.window:
            self.timestamps.popleft()
        
        if len(self.timestamps) >= self.calls:
            # Rate limit exceeded, wait
            sleep_time = self.window - (now - self.timestamps[0])
            if sleep_time > 0:
                logger.debug(f"Rate limit reached, sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
        
        self.timestamps.append(time.time())


class DeadLetterQueue:
    """
    Stocke les URLs qui ont échoué après tous les retries.
    
    Permet une analyse post-mortem et retry manuel.
    """
    
    def __init__(self, output_file: str = "backend/sources/failed_urls.json"):
        self.output_file = output_file
        self.failed_items: List[dict] = []
    
    def add(self, url: str, error: str, category: str, attempts: int):
        """Ajoute un élément échoué."""
        self.failed_items.append({
            "url": url,
            "error": str(error),
            "category": category,
            "attempts": attempts,
            "timestamp": datetime.now().isoformat()
        })
        logger.error(f"Added to DLQ: {url} - {error}")
    
    def save(self):
        """Persiste la DLQ sur disque."""
        import json
        from pathlib import Path
        
        if not self.failed_items:
            return
        
        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.failed_items, f, indent=2, ensure_ascii=False)
        
        logger.info(f"DLQ saved: {len(self.failed_items)} failed items in {self.output_file}")


def retry_with_backoff(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    max_delay: int = 300,
    retry_on_exceptions: tuple = (Exception,),
    retry_on_result: Optional[Callable[[Any], bool]] = None
):
    """
    Decorator pour retry avec exponential backoff.
    
    Args:
        max_retries: Nombre maximum de tentatives
        backoff_factor: Facteur d'augmentation du délai (1s, 2s, 4s, 8s...)
        max_delay: Délai maximum entre retries
        retry_on_exceptions: Exceptions qui déclenchent un retry
        retry_on_result: Function qui teste si le résultat mérite un retry
    
    Example:
        @retry_with_backoff(max_retries=3, backoff_factor=2)
        def scrape_url(url):
            return requests.get(url)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            stats = RetryStats()
            
            for attempt in range(1, max_retries + 1):
                stats.total_attempts += 1
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Check if result should trigger retry
                    if retry_on_result and retry_on_result(result):
                        if attempt < max_retries:
                            delay = min(backoff_factor ** (attempt - 1), max_delay)
                            logger.warning(
                                f"{func.__name__} attempt {attempt}/{max_retries} - "
                                f"Invalid result, retrying in {delay}s"
                            )
                            time.sleep(delay)
                            continue
                    
                    # Success
                    stats.successful_attempts += 1
                    stats.last_success = datetime.now()
                    stats.consecutive_failures = 0
                    
                    if attempt > 1:
                        logger.info(
                            f"{func.__name__} succeeded after {attempt} attempts"
                        )
                    
                    return result
                
                except retry_on_exceptions as e:
                    stats.failed_attempts += 1
                    stats.last_failure = datetime.now()
                    stats.consecutive_failures += 1
                    
                    if attempt == max_retries:
                        logger.error(
                            f"{func.__name__} failed after {max_retries} attempts: {e}"
                        )
                        raise
                    
                    # Calculate backoff delay
                    delay = min(backoff_factor ** (attempt - 1), max_delay)
                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{max_retries} failed: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
            
            # Should never reach here
            raise RuntimeError(f"{func.__name__} exhausted all retries")
        
        return wrapper
    return decorator


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    pass
