"""
A2A Discovery 2.0 — onserver le partie discovery,savoir ce qu il font
======================================================================

Every routing decision records candidates, scores, and winner
reasoning into the message's TraceEnvelope.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from backend.src.agriconnect.core.agent_registry import internal_agents

from .registry import A2ARegistry, AgentCard, AgentDomain, AgentStatus
from .messaging import A2AChannel, A2AMessage, MessageType
from backend.src.agriconnect.protocols.core import (
    CorrelationCtx,
    TraceCategory,
    TraceEnvelope,
)

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

    # Small util: canonicalize topic/intent/zone keys in one place
    @staticmethod
    def normalize_key(val: Optional[str]) -> Optional[str]:
        if val is None:
            return None
        return val.strip().upper()
    def _build_message(
        self,
        sender: str,
        intent_key: str,
        payload: Dict[str, Any],
        zone: str,
        crop: str,
        priority: int,
        trace_envelope: Optional[TraceEnvelope],
        correlation: Optional[CorrelationCtx],
    ) -> A2AMessage:
        return A2AMessage(
            sender_id=sender,
            intent=intent_key,
            payload=payload,
            zone=zone.strip().upper(),
            crop=crop,
            priority=priority,
            trace_envelope=trace_envelope,
            correlation=correlation or CorrelationCtx(),
        )

    def register_internal_agents(self):
        """Enregistre les agents internes définis dans la configuration core."""
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
        agent_id = self.registry.register(card)
        for intent in card.intents:
            intent_key = intent.strip().upper()
            self.channel.subscribe(agent_id, intent_key)
            for zone in card.zones:
                zone_key = zone.strip().upper()
                if zone_key != "ALL":
                    self.channel.subscribe(agent_id, f"{intent_key}_{zone_key}")
                else:
                    self.channel.subscribe(agent_id, f"{intent_key}_GLOBAL")
        return agent_id

    def find_agents(
        self,
        intent: Optional[str] = None,
        zone: Optional[str] = None,
        crop: Optional[str] = None,
        domain: Optional[AgentDomain] = None,
    ) -> List[AgentCard]:
        clean_intent = self.normalize_key(intent)
        clean_zone = self.normalize_key(zone)
        return self.registry.discover(intent=clean_intent, zone=clean_zone, crop=crop, domain=domain)

    def route_message(self, sender: str, intent: str, payload: Dict[str, Any], zone: str = "", crop: str = "", receiver: Optional[str] = None, priority: int = 0, trace_envelope: Optional[TraceEnvelope] = None, correlation: Optional[CorrelationCtx] = None) -> Dict[str, Any]:
        """Route un message. If `receiver` is provided, do a direct route; otherwise use discovery."""
        t0 = time.monotonic()
        intent_key = self.normalize_key(intent) or ""

        message = self._build_message(sender, intent_key, payload, zone, crop, priority, trace_envelope, correlation)

        if receiver:
            return self._route_direct(message, receiver, intent_key, trace_envelope, t0)

        return self._route_by_intent(message, intent_key, zone, crop, payload, trace_envelope, t0)

    def _route_direct(self, message, receiver, intent_key, trace_envelope, t0):
        message.receiver_id = receiver
        ack = self.channel.send(message)
        if trace_envelope:
            trace_envelope.record(
                TraceCategory.ROUTING,
                "A2ADiscovery",
                "route_direct",
                input_summary={"intent": intent_key, "receiver": receiver},
                output_summary={"ack": getattr(ack, 'ack_status', None)},
                reasoning=f"Direct route to {receiver}",
                duration_ms=(time.monotonic() - t0) * 1000,
            )
        return {"message_id": getattr(ack, 'message_id', message.message_id), "delivered_to": [receiver], "status": "ok"}

    def _route_by_intent(self, message, intent_key, zone, crop, payload, trace_envelope, t0):
        scored = self.registry.discover_scored(intent=intent_key, zone=zone, crop=crop)
        agents = self.find_agents(intent=intent_key, zone=zone, crop=crop)

        if not agents:
            logger.warning("🚫 Aucun agent trouvé pour %s dans la zone %s", intent_key, zone)
            if trace_envelope:
                trace_envelope.record(
                    TraceCategory.DISCOVERY,
                    "A2ADiscovery",
                    "no_candidates",
                    input_summary={"intent": intent_key, "zone": zone, "crop": crop},
                    output_summary={"candidates": 0},
                    reasoning="No agent found matching intent/zone/crop",
                    duration_ms=(time.monotonic() - t0) * 1000,
                )
            return {"message_id": message.message_id, "delivered_to": [], "status": "no_agent"}

        return self._attempt_send_to_agents(message, agents, scored, intent_key, zone, crop, payload, trace_envelope, t0)

    def _validate_payload(self, agent: AgentCard, payload: Dict[str, Any]) -> Optional[str]:
        schema = getattr(agent, "input_schema", None)
        if not schema:
            return None
        required = schema.get("required") or []
        missing = [r for r in required if r not in payload]
        if missing:
            return f"missing_required_fields: {missing}"
        return None

    def _attempt_send_to_agents(self, message, agents, scored, intent_key, zone, crop, payload, trace_envelope, t0):
        last_ack = None
        delivered_to: List[str] = []
        attempted: List[str] = []
        for agent in agents:
            attempted.append(agent.agent_id)

            if agent.status != AgentStatus.ACTIVE:
                if trace_envelope:
                    trace_envelope.record(
                        TraceCategory.DISCOVERY,
                        "A2ADiscovery",
                        "skip_inactive",
                        input_summary={"agent": agent.agent_id},
                        output_summary={"status": agent.status.value},
                        reasoning=f"Skipped {agent.agent_id} due to status={agent.status.value}",
                    )
                continue

            payload_err = self._validate_payload(agent, payload)
            if payload_err:
                if trace_envelope:
                    trace_envelope.record(
                        TraceCategory.DISCOVERY,
                        "A2ADiscovery",
                        "invalid_payload",
                        input_summary={"agent": agent.agent_id, "required_missing": payload_err},
                        output_summary={"status": "invalid_payload"},
                        reasoning=f"Payload validation failed for {agent.agent_id}: {payload_err}",
                    )
                continue

            message.receiver_id = agent.agent_id
            ack = self.channel.send(message)
            last_ack = ack

            if trace_envelope:
                trace_envelope.record(
                    TraceCategory.DISCOVERY,
                    "A2ADiscovery",
                    "attempt_send",
                    input_summary={"intent": intent_key, "candidate": agent.agent_id},
                    output_summary={"ack": getattr(ack, 'ack_status', None)},
                    reasoning=(f"Attempted send to {agent.agent_id}; ack={getattr(ack, 'ack_status', None)}"),
                    duration_ms=(time.monotonic() - t0) * 1000,
                )

            status = getattr(ack, 'ack_status', None)
            status_val = status.value if hasattr(status, 'value') else str(status)
            if status_val == 'accepted':
                delivered_to.append(agent.agent_id)
                if trace_envelope:
                    trace_envelope.record(
                        TraceCategory.DISCOVERY,
                        "A2ADiscovery",
                        "route_by_intent",
                        input_summary={"intent": intent_key, "zone": zone, "crop": crop},
                        output_summary={
                            "candidates": scored,
                            "winner": agent.agent_id,
                            "winner_name": agent.name,
                        },
                        reasoning=(
                            f"Selected {agent.name} ({agent.agent_id}) "
                            f"from {len(scored)} candidates"
                        ),
                        duration_ms=(time.monotonic() - t0) * 1000,
                    )
                return {"message_id": getattr(ack, 'message_id', message.message_id), "delivered_to": delivered_to, "agent_name": agent.name, "status": "ok"}

            # non-accepted: record and continue
            if trace_envelope:
                trace_envelope.record(
                    TraceCategory.DISCOVERY,
                    "A2ADiscovery",
                    "candidate_failed",
                    input_summary={"agent": agent.agent_id},
                    output_summary={"ack": getattr(ack, 'ack_status', None)},
                    reasoning=f"Candidate {agent.agent_id} failed to accept message; trying next",
                )
            continue

        if trace_envelope:
            trace_envelope.record(
                TraceCategory.DISCOVERY,
                "A2ADiscovery",
                "undeliverable",
                input_summary={"intent": intent_key, "attempted": attempted},
                output_summary={"last_ack": getattr(last_ack, 'ack_status', None) if last_ack else None},
                reasoning="No candidate accepted the message",
                duration_ms=(time.monotonic() - t0) * 1000,
            )

        return {"message_id": message.message_id, "delivered_to": delivered_to, "status": "undeliverable"}

    def broadcast_offer(
        self,
        sender: str,
        intent: str,
        payload: Dict[str, Any],
        zone: str = "",
        crop: str = "",
        trace_envelope: Optional[TraceEnvelope] = None,
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
            trace_envelope=trace_envelope,
        )

        topic = f"{intent_key}_{zone_key}" if zone_key and zone_key != "ALL" else intent_key
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