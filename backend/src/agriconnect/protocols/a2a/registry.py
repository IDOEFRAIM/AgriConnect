import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from enum import Enum

logger = logging.getLogger("A2A.Registry")

# ═══════════════════════════════════════════════════════════
# ENUMS & MODELS
# ═══════════════════════════════════════════════════════════

class AgentStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    BUSY = "busy"
    ERROR = "error"

class AgentDomain(str, Enum):
    DIAGNOSIS = "diagnosis"
    MARKET = "market"
    MARKETPLACE = "marketplace"
    WEATHER = "weather"
    FORMATION = "formation"
    SOIL = "soil"
    FINANCE = "finance"
    EXTERNAL = "external"

@dataclass
class AgentCard:
    agent_id: str = ""
    name: str = ""
    description: str = ""
    domain: AgentDomain = AgentDomain.EXTERNAL
    intents: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    zones: List[str] = field(default_factory=list)
    crops: List[str] = field(default_factory=list)
    endpoint: str = ""
    protocol: str = "internal"
    version: str = "1.0"
    status: AgentStatus = AgentStatus.ACTIVE
    avg_response_ms: int = 500
    registered_at: str = ""
    last_heartbeat: str = ""

    def __post_init__(self):
        if not self.agent_id: self.agent_id = str(uuid.uuid4())[:8]
        if not self.registered_at: self.registered_at = datetime.now(timezone.utc).isoformat()
        # Normalisation automatique
        self.intents = [i.strip().upper() for i in self.intents]
        # Store zones in UPPERCASE for consistent topic matching
        self.zones = [z.strip().upper() for z in self.zones]
        self.crops = [c.strip().lower() for c in self.crops]

    def supports_zone(self, zone: str) -> bool:
        z = zone.strip().lower()
        return "all" in self.zones or z in self.zones

    def supports_crop(self, crop: str) -> bool:
        c = crop.strip().lower()
        return "all" in self.crops or c in self.crops

# ═══════════════════════════════════════════════════════════
# REGISTRY CORE
# ═══════════════════════════════════════════════════════════

class A2ARegistry:
    def __init__(self):
        self._agents: Dict[str, AgentCard] = {}
        # Doubles index pour une recherche O(1)
        self._intent_index: Dict[str, Set[str]] = {}
        self._zone_index: Dict[str, Set[str]] = {}
        logger.info("🔌 A2A Registry initialisé")

    def register(self, card: AgentCard) -> str:
        """Enregistre un agent et met à jour les index de recherche rapide."""
        self._agents[card.agent_id] = card

        # Indexation Intentions
        for intent in card.intents:
            self._intent_index.setdefault(intent.upper(), set()).add(card.agent_id)
        
        # Indexation Zones
        for zone in card.zones:
            self._zone_index.setdefault(zone.upper(), set()).add(card.agent_id)

        logger.info("✅ Agent [%s] enregistré (%s)", card.name, card.agent_id)
        return card.agent_id

    def unregister(self, agent_id: str):
        """Supprime un agent et nettoie proprement tous les index."""
        if agent_id in self._agents:
            del self._agents[agent_id]
            for index in [self._intent_index, self._zone_index]:
                for ids in index.values():
                    ids.discard(agent_id)
            logger.info("❌ Agent [%s] retiré", agent_id)

    def discover(
        self,
        intent: Optional[str] = None,
        zone: Optional[str] = None,
        crop: Optional[str] = None,
        domain: Optional[AgentDomain] = None,
    ) -> List[AgentCard]:
        """
        Moteur de recherche hybride.
        Utilise les index pour filtrer massivement, puis affine les résultats.
        """
        # 1. Intersection des index (Recherche ultra-rapide)
        candidates_ids = None

        if intent:
            intent_ids = self._intent_index.get(intent.upper(), set())
            candidates_ids = intent_ids if candidates_ids is None else candidates_ids & intent_ids

        if zone and zone.lower() != "all":
            zone_ids = self._zone_index.get(zone.lower(), set()) | self._zone_index.get("all", set())
            candidates_ids = zone_ids if candidates_ids is None else candidates_ids & zone_ids

        # 2. Récupération des objets
        if candidates_ids is not None:
            candidates = [self._agents[aid] for aid in candidates_ids if aid in self._agents]
        else:
            candidates = list(self._agents.values())

        # 3. Filtrage fin (Cultures, Domaine, Status)
        results = [
            a for a in candidates 
            if a.status == AgentStatus.ACTIVE
            and (not domain or a.domain == domain)
            and (not crop or a.supports_crop(crop))
        ]

        # 4. Tri par performance
        results.sort(key=lambda a: a.avg_response_ms)
        return results

    def heartbeat(self, agent_id: str, status: AgentStatus = AgentStatus.ACTIVE):
        """Update last seen."""
        if agent_id in self._agents:
            self._agents[agent_id].status = status
            self._agents[agent_id].last_heartbeat = datetime.now(timezone.utc).isoformat()

    def stats(self) -> Dict[str, Any]:
        return {
            "total": len(self._agents),
            "active": sum(1 for a in self._agents.values() if a.status == AgentStatus.ACTIVE),
            "intents": len(self._intent_index),
            "zones": len(self._zone_index)
        }