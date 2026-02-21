"""
Système de checkpointing pour reprendre le scraping après interruption.

Essentiel pour AWS Lambda avec timeout limité et exécutions périodiques.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    """État complet d'une session de scraping."""
    
    session_id: str
    started_at: str
    last_updated: str
    
    # URLs par catégorie
    pending_urls: Dict[str, List[str]] = field(default_factory=dict)
    completed_urls: Dict[str, Set[str]] = field(default_factory=dict)
    failed_urls: Dict[str, Set[str]] = field(default_factory=dict)
    
    # Statistiques
    total_urls: int = 0
    processed_urls: int = 0
    successful_urls: int = 0
    failed_urls_count: int = 0
    
    # Métadonnées
    execution_count: int = 0  # Nombre d'exécutions Lambda
    total_duration_seconds: float = 0.0
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "last_updated": self.last_updated,
            "pending_urls": self.pending_urls,
            "completed_urls": {k: list(v) for k, v in self.completed_urls.items()},
            "failed_urls": {k: list(v) for k, v in self.failed_urls.items()},
            "total_urls": self.total_urls,
            "processed_urls": self.processed_urls,
            "successful_urls": self.successful_urls,
            "failed_urls_count": self.failed_urls_count,
            "execution_count": self.execution_count,
            "total_duration_seconds": self.total_duration_seconds
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CheckpointState':
        """Deserialize from dictionary."""
        # Convert lists back to sets for completed/failed URLs
        completed = {k: set(v) for k, v in data.get("completed_urls", {}).items()}
        failed = {k: set(v) for k, v in data.get("failed_urls", {}).items()}
        
        return cls(
            session_id=data["session_id"],
            started_at=data["started_at"],
            last_updated=data["last_updated"],
            pending_urls=data.get("pending_urls", {}),
            completed_urls=completed,
            failed_urls=failed,
            total_urls=data.get("total_urls", 0),
            processed_urls=data.get("processed_urls", 0),
            successful_urls=data.get("successful_urls", 0),
            failed_urls_count=data.get("failed_urls_count", 0),
            execution_count=data.get("execution_count", 0),
            total_duration_seconds=data.get("total_duration_seconds", 0.0)
        )


class CheckpointManager:
    """
    Gère les checkpoints pour reprendre le scraping après interruption.
    
    Pattern:
    1. Load existing checkpoint or create new
    2. Process URLs batch by batch
    3. Save checkpoint after each batch
    4. On next execution, resume from checkpoint
    """
    
    def __init__(self, checkpoint_dir: Path = Path("backend/sources/checkpoints")):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.current_state: Optional[CheckpointState] = None
    
    def get_checkpoint_path(self, session_id: str) -> Path:
        """Get path to checkpoint file."""
        return self.checkpoint_dir / f"checkpoint_{session_id}.json"
    
    def load_or_create(self, session_id: str, sources: Dict[str, List[str]]) -> CheckpointState:
        """
        Load existing checkpoint or create new session.
        
        Args:
            session_id: Unique identifier for this scraping session
            sources: Dictionary of category -> URLs
        
        Returns:
            CheckpointState with pending URLs
        """
        checkpoint_path = self.get_checkpoint_path(session_id)
        
        if checkpoint_path.exists():
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            return self.load(session_id)
        else:
            logger.info(f"Starting new session: {session_id}")
            return self.create_new(session_id, sources)
    
    def create_new(self, session_id: str, sources: Dict[str, List[str]]) -> CheckpointState:
        """Create new checkpoint state."""
        now = datetime.now().isoformat()
        
        # Initialize pending URLs by category
        pending_urls = {category: list(urls) for category, urls in sources.items()}
        total_urls = sum(len(urls) for urls in pending_urls.values())
        
        state = CheckpointState(
            session_id=session_id,
            started_at=now,
            last_updated=now,
            pending_urls=pending_urls,
            completed_urls={category: set() for category in sources.keys()},
            failed_urls={category: set() for category in sources.keys()},
            total_urls=total_urls,
            execution_count=1
        )
        
        self.current_state = state
        self.save(state)
        return state
    
    def load(self, session_id: str) -> CheckpointState:
        """Load existing checkpoint."""
        checkpoint_path = self.get_checkpoint_path(session_id)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        state = CheckpointState.from_dict(data)
        state.execution_count += 1
        state.last_updated = datetime.now().isoformat()
        
        self.current_state = state
        return state
    
    def save(self, state: Optional[CheckpointState] = None):
        """Save checkpoint to disk."""
        if state is None:
            state = self.current_state
        
        if state is None:
            logger.warning("No checkpoint state to save")
            return
        
        checkpoint_path = self.get_checkpoint_path(state.session_id)
        
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(state.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Checkpoint saved: {checkpoint_path}")
    
    def mark_completed(self, category: str, url: str, success: bool = True):
        """Mark URL as completed (success or failure)."""
        if self.current_state is None:
            logger.warning("No active checkpoint state")
            return
        
        # Remove from pending
        if category in self.current_state.pending_urls:
            if url in self.current_state.pending_urls[category]:
                self.current_state.pending_urls[category].remove(url)
        
        # Add to completed or failed
        if success:
            self.current_state.completed_urls[category].add(url)
            self.current_state.successful_urls += 1
        else:
            self.current_state.failed_urls[category].add(url)
            self.current_state.failed_urls_count += 1
        
        self.current_state.processed_urls += 1
        self.current_state.last_updated = datetime.now().isoformat()
    
    def get_pending_batch(self, category: str, batch_size: int = 10) -> List[str]:
        """Get next batch of URLs to process for a category."""
        if self.current_state is None:
            return []
        
        pending = self.current_state.pending_urls.get(category, [])
        batch = pending[:batch_size]
        
        return batch
    
    def is_complete(self) -> bool:
        """Check if all URLs have been processed."""
        if self.current_state is None:
            return True
        
        total_pending = sum(
            len(urls) for urls in self.current_state.pending_urls.values()
        )
        return total_pending == 0
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress statistics."""
        if self.current_state is None:
            return {}
        
        return {
            "total_urls": self.current_state.total_urls,
            "processed": self.current_state.processed_urls,
            "successful": self.current_state.successful_urls,
            "failed": self.current_state.failed_urls_count,
            "pending": self.current_state.total_urls - self.current_state.processed_urls,
            "progress_percent": round(
                (self.current_state.processed_urls / self.current_state.total_urls * 100)
                if self.current_state.total_urls > 0 else 0,
                2
            ),
            "executions": self.current_state.execution_count,
            "duration_seconds": self.current_state.total_duration_seconds
        }
    
    def cleanup_old_checkpoints(self, keep_days: int = 7):
        """Remove checkpoints older than specified days."""
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        
        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_*.json"):
            try:
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                
                last_updated = datetime.fromisoformat(data.get("last_updated", ""))
                
                if last_updated < cutoff_date:
                    checkpoint_file.unlink()
                    logger.info(f"Cleaned up old checkpoint: {checkpoint_file.name}")
            
            except Exception as e:
                logger.warning(f"Failed to clean up {checkpoint_file}: {e}")
