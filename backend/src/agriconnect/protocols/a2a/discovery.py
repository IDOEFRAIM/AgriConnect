"""
A2A Discovery — Service de découverte d'agents pour AgriConnect.
=================================================================
Ce service est le point d'entrée unique pour toute communication A2A.
"""

import logging
from typing import Any, Dict, List, Optional

# Imports internes (assure-toi que ces chemins correspondent à ta structure finale)
from .registry import A2ARegistry, AgentCard, AgentDomain, AgentStatus
from .messaging import A2AChannel, A2AMessage, MessageType

# NOTE: internal_agents importé en lazy dans register_internal_agents()
# pour éviter un import circulaire core → a2a → core.

logger = logging.getLogger("A2A.Discovery")

class A2ADiscovery:
    """
    Service de découverte et routing inter-agents.
    Gère le cycle de vie : Enregistrement -> Découverte -> Routage.
    """

    def __init__(self, registry: Optional[A2ARegistry] = None, channel: Optional[A2AChannel] = None):
        self.registry = registry or A2ARegistry()
        self.channel = channel or A2AChannel()
        logger.info("🔌 A2A Discovery Service initialisé")

    # ═══════════════════════════════════════════════════════════
    # ENREGISTREMENT
    # ═══════════════════════════════════════════════════════════

    def register_internal_agents(self):
        """
        Enregistre les agents internes définis dans la configuration core.
        Configure automatiquement les abonnements aux messages.
        """
        # Import en lazy pour briser le cycle core ↔ a2a
        from backend.src.agriconnect.core.agent_registry import internal_agents

        for card in internal_agents:
            self._register_and_subscribe(card)
        
        logger.info("✅ %d agents internes enregistrés et abonnés", len(internal_agents))

    def register_external_agent(self, card: AgentCard) -> str:
        """Enregistre un agent externe (ex: formation, sentinelle)."""
        card.protocol = card.protocol or "http"
        agent_id = self._register_and_subscribe(card)
        logger.info("🔗 Agent externe enregistré: %s (%s)", card.name, card.endpoint)
        return agent_id

    def _register_and_subscribe(self, card: AgentCard) -> str:
        """Logique privée pour lier l'enregistrement et l'abonnement aux topics."""
        agent_id = self.registry.register(card)

        for intent in card.intents:
            intent_key = intent.strip().upper()
            
            # Abonnement au topic de base (ex: "CHECK_PRICE")
            self.channel.subscribe(agent_id, intent_key)
            
            # Abonnement spécifique par zone
            for zone in card.zones:
                zone_key = zone.strip().upper()
                if zone_key != "ALL":
                    # ex: "CHECK_PRICE_BOBO"
                    self.channel.subscribe(agent_id, f"{intent_key}_{zone_key}")
                else:
                    # Pour les agents globaux, on crée un topic spécifique
                    self.channel.subscribe(agent_id, f"{intent_key}_GLOBAL")
        
        return agent_id

    # ═══════════════════════════════════════════════════════════
    # DISCOVERY (RECHERCHE)
    # ═══════════════════════════════════════════════════════════

    def find_agents(
        self,
        intent: Optional[str] = None,
        zone: Optional[str] = None,
        crop: Optional[str] = None,
        domain: Optional[AgentDomain] = None,
    ) -> List[AgentCard]:
        """Trouve les agents capables de traiter une requête spécifique."""
        # Normalisation des inputs pour la recherche
        clean_intent = intent.strip().upper() if intent else None
        clean_zone = zone.strip().upper() if zone else None
        
        return self.registry.discover(
            intent=clean_intent, 
            zone=clean_zone, 
            crop=crop, 
            domain=domain
        )

    # ═══════════════════════════════════════════════════════════
    # ROUTING (ENVOI)
    # ═══════════════════════════════════════════════════════════

    def route_message(
        self,
        sender: str,
        intent: str,
        payload: Dict[str, Any],
        zone: str = "",
        crop: str = "",
        receiver: Optional[str] = None,
        priority: int = 0,
    ) -> Dict[str, Any]:
        """
        Route un message. Si 'receiver' est vide, utilise le discovery 
        pour trouver le meilleur agent automatiquement.
        """
        intent_key = intent.strip().upper()
        
        message = A2AMessage(
            sender_id=sender,
            intent=intent_key,
            payload=payload,
            zone=zone.strip().upper(),
            crop=crop,
            priority=priority,
        )

        # Cas 1 : Destinataire forcé (ex: réponse directe à un handshake)
        if receiver:
            message.receiver_id = receiver
            msg_id = self.channel.send(message)
            return {"message_id": msg_id, "delivered_to": [receiver], "status": "ok"}

        # Cas 2 : Routage intelligent par intention
        agents = self.find_agents(intent=intent_key, zone=zone, crop=crop)
        if not agents:
            logger.warning("🚫 Aucun agent trouvé pour %s dans la zone %s", intent_key, zone)
            return {"message_id": message.message_id, "delivered_to": [], "status": "no_agent"}

        # On prend le premier (le plus pertinent selon le registre)
        best_agent = agents[0]
        message.receiver_id = best_agent.agent_id
        msg_id = self.channel.send(message)

        return {
            "message_id": msg_id,
            "delivered_to": [best_agent.agent_id],
            "agent_name": best_agent.name,
            "status": "ok",
        }

    def broadcast_offer(
        self,
        sender: str,
        intent: str,
        payload: Dict[str, Any],
        zone: str = "",
        crop: str = "",
    ) -> Dict[str, Any]:
        """Diffuse une offre à tous les agents abonnés au topic."""
        intent_key = intent.strip().upper()
        zone_key = zone.strip().upper()
        
        message = A2AMessage(
            sender_id=sender,
            intent=intent_key,
            payload=payload,
            zone=zone_key,
            crop=crop,
        )

        # On construit le topic cible
        topic = f"{intent_key}_{zone_key}" if zone_key and zone_key != "ALL" else intent_key
        
        # Le channel gère la distribution aux abonnés du topic
        delivered = self.channel.broadcast(message, topic=topic)

        return {
            "message_id": message.message_id,
            "topic": topic,
            "delivered_to": delivered,
            "count": len(delivered),
            "status": "ok",
        }

    # ═══════════════════════════════════════════════════════════
    # TRADING & MONITORING
    # ═══════════════════════════════════════════════════════════

    def initiate_trade(self, seller_id: str, buyer_id: str, offer: Dict[str, Any]) -> str:
        """Déclenche un protocole de négociation sécurisé entre deux agents."""
        message = A2AMessage(
            sender_id=seller_id,
            receiver_id=buyer_id,
            intent="TRADE_INITIATE",
            payload=offer,
        )
        return self.channel.initiate_handshake(message)

    def status(self) -> Dict[str, Any]:
        """Donne une vue globale de la santé du réseau A2A."""
        return {
            "registry_stats": self.registry.stats(),
            "channel_stats": self.channel.stats(),
            "uptime_status": "healthy"
        }