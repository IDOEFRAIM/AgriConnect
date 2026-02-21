"""
Tests unitaires — Protocoles A2A (messaging, registry, discovery).
"""

import pytest
from unittest.mock import MagicMock, patch


class TestA2ARegistry:
    """Tests du registre d'agents A2A."""

    def test_register_agent(self):
        from backend.src.agriconnect.protocols.a2a.registry import A2ARegistry, AgentCard, AgentDomain

        reg = A2ARegistry()
        card = AgentCard(
            name="TestAgent",
            domain=AgentDomain.FORMATION,
            capabilities=["training"],
            endpoint="http://localhost:8000",
            zones=["BOBO", "OUAGA"],
        )
        agent_id = reg.register(card)
        assert agent_id is not None
        assert len(agent_id) > 0

    def test_discover_by_domain(self):
        from backend.src.agriconnect.protocols.a2a.registry import A2ARegistry, AgentCard, AgentDomain

        reg = A2ARegistry()
        card = AgentCard(
            name="FormationAgent",
            domain=AgentDomain.FORMATION,
            capabilities=["training"],
            endpoint="http://localhost",
            zones=["all"],
        )
        reg.register(card)
        found = reg.discover(domain=AgentDomain.FORMATION)
        assert len(found) >= 1
        assert any(c.name == "FormationAgent" for c in found)

    def test_discover_by_intent(self):
        from backend.src.agriconnect.protocols.a2a.registry import A2ARegistry, AgentCard, AgentDomain

        reg = A2ARegistry()
        card = AgentCard(
            name="PriceAgent",
            domain=AgentDomain.MARKET,
            intents=["CHECK_PRICE", "SELL_OFFER"],
            zones=["all"],
        )
        reg.register(card)
        found = reg.discover(intent="CHECK_PRICE")
        assert len(found) >= 1
        assert found[0].name == "PriceAgent"

    def test_register_and_unregister(self):
        from backend.src.agriconnect.protocols.a2a.registry import A2ARegistry, AgentCard, AgentDomain

        reg = A2ARegistry()
        card = AgentCard(name="TempAgent", domain=AgentDomain.SOIL)
        agent_id = reg.register(card)
        assert len(reg.discover()) == 1
        reg.unregister(agent_id)
        assert len(reg.discover()) == 0

    def test_heartbeat_updates_status(self):
        from backend.src.agriconnect.protocols.a2a.registry import (
            A2ARegistry, AgentCard, AgentDomain, AgentStatus,
        )

        reg = A2ARegistry()
        card = AgentCard(name="HeartbeatAgent", domain=AgentDomain.WEATHER)
        agent_id = reg.register(card)
        reg.heartbeat(agent_id, AgentStatus.BUSY)
        assert reg._agents[agent_id].status == AgentStatus.BUSY

    def test_stats(self):
        from backend.src.agriconnect.protocols.a2a.registry import A2ARegistry, AgentCard, AgentDomain

        reg = A2ARegistry()
        reg.register(AgentCard(name="A", domain=AgentDomain.FORMATION, intents=["LEARN"]))
        reg.register(AgentCard(name="B", domain=AgentDomain.MARKET, intents=["SELL"]))
        stats = reg.stats()
        assert stats["total"] == 2
        assert stats["active"] == 2


class TestA2AMessaging:
    """Tests du canal de messagerie A2A."""

    def test_create_message(self):
        from backend.src.agriconnect.protocols.a2a.messaging import A2AMessage, MessageType

        msg = A2AMessage(
            sender_id="agent_1",
            receiver_id="agent_2",
            message_type=MessageType.REQUEST,
            intent="CHECK_PRICE",
            payload={"question": "Quel est le prix du maïs ?"},
        )
        assert msg.sender_id == "agent_1"
        assert msg.receiver_id == "agent_2"
        assert msg.message_type == MessageType.REQUEST
        assert msg.intent == "CHECK_PRICE"

    def test_message_validation_ok(self):
        from backend.src.agriconnect.protocols.a2a.messaging import A2AMessage, MessageType

        msg = A2AMessage(
            sender_id="a1",
            receiver_id="a2",
            intent="CHECK_PRICE",
        )
        result = msg.validate()
        assert result == {"status": "ok"}

    def test_message_validation_missing_sender(self):
        from backend.src.agriconnect.protocols.a2a.messaging import A2AMessage

        msg = A2AMessage(receiver_id="a2", intent="SELL")
        result = msg.validate()
        assert "error" in result

    def test_copy_for_receiver(self):
        from backend.src.agriconnect.protocols.a2a.messaging import A2AMessage, MessageType

        msg = A2AMessage(
            sender_id="broadcaster",
            receiver_id="all",
            message_type=MessageType.BROADCAST,
            intent="ALERT",
            payload={"alert": "Sécheresse"},
        )
        copy = msg.copy_for_receiver("agent_42")
        assert copy.receiver_id == "agent_42"
        assert copy.sender_id == "broadcaster"
        assert copy.payload == msg.payload
        assert copy is not msg
        assert copy.message_id != msg.message_id

    def test_channel_send_receive(self):
        from backend.src.agriconnect.protocols.a2a.messaging import A2AChannel, A2AMessage, MessageType

        channel = A2AChannel()

        msg = A2AMessage(
            sender_id="sentinel",
            receiver_id="agent_1",
            intent="FLOOD_ALERT",
            message_type=MessageType.REQUEST,
            payload={"type": "FLOOD"},
        )
        channel.send(msg)

        inbox = channel.receive("agent_1")
        assert len(inbox) == 1
        assert inbox[0].payload["type"] == "FLOOD"

    def test_send_requires_receiver_id(self):
        from backend.src.agriconnect.protocols.a2a.messaging import A2AChannel, A2AMessage

        channel = A2AChannel()
        msg = A2AMessage(sender_id="a1", intent="SELL")  # no receiver_id
        with pytest.raises(ValueError, match="receiver_id"):
            channel.send(msg)

    def test_broadcast_creates_copies(self):
        from backend.src.agriconnect.protocols.a2a.messaging import A2AChannel, A2AMessage, MessageType

        channel = A2AChannel()
        channel.subscribe("a1", "ALERTS")
        channel.subscribe("a2", "ALERTS")

        msg = A2AMessage(
            sender_id="broadcaster",
            intent="ALERT_WEATHER",
            payload={"data": "test"},
        )
        delivered = channel.broadcast(msg, topic="ALERTS")

        assert "a1" in delivered
        assert "a2" in delivered

        inbox_a1 = channel.receive("a1")
        inbox_a2 = channel.receive("a2")
        assert len(inbox_a1) == 1
        assert len(inbox_a2) == 1
        assert inbox_a1[0].receiver_id == "a1"
        assert inbox_a2[0].receiver_id == "a2"
        assert inbox_a1[0].message_type == MessageType.BROADCAST

    def test_broadcast_excludes_sender(self):
        from backend.src.agriconnect.protocols.a2a.messaging import A2AChannel, A2AMessage

        channel = A2AChannel()
        channel.subscribe("broadcaster", "NEWS")
        channel.subscribe("listener", "NEWS")

        msg = A2AMessage(sender_id="broadcaster", intent="NEWS")
        delivered = channel.broadcast(msg, topic="NEWS")
        assert "broadcaster" not in delivered
        assert "listener" in delivered

    def test_handshake_requires_receiver_id(self):
        from backend.src.agriconnect.protocols.a2a.messaging import A2AChannel, A2AMessage

        channel = A2AChannel()
        msg = A2AMessage(sender_id="a1", intent="NEGOTIATE")
        with pytest.raises(ValueError, match="receiver_id"):
            channel.initiate_handshake(msg)

    def test_handshake_full_cycle(self):
        from backend.src.agriconnect.protocols.a2a.messaging import (
            A2AChannel, A2AMessage, HandshakeStatus, MessageType,
        )

        channel = A2AChannel()
        msg = A2AMessage(
            sender_id="buyer",
            receiver_id="seller",
            intent="NEGOTIATE_PRICE",
            payload={"price": 250},
        )
        hs_id = channel.initiate_handshake(msg)
        assert hs_id == msg.message_id

        # Seller accepts
        resp = channel.respond_handshake(hs_id, "seller", HandshakeStatus.ACCEPTED, {"price": 240})
        assert resp.handshake_status == HandshakeStatus.ACCEPTED
        assert resp.sender_id == "seller"

    def test_idempotency_dedup(self):
        from backend.src.agriconnect.protocols.a2a.messaging import A2AChannel, A2AMessage

        channel = A2AChannel()
        msg = A2AMessage(sender_id="a1", receiver_id="a2", intent="SELL")
        channel.send(msg)
        channel.send(msg)  # same idempotency key → deduplicated
        inbox = channel.receive("a2")
        assert len(inbox) == 1

    def test_channel_stats(self):
        from backend.src.agriconnect.protocols.a2a.messaging import A2AChannel, A2AMessage

        channel = A2AChannel()
        channel.send(A2AMessage(sender_id="a", receiver_id="b", intent="X"))
        stats = channel.stats()
        assert stats["total_sent"] == 1
        assert stats["pending"] >= 1


class TestA2ADiscovery:
    """Tests du service de découverte A2A."""

    def test_discovery_init(self):
        from backend.src.agriconnect.protocols.a2a import A2ADiscovery

        discovery = A2ADiscovery()
        assert discovery.registry is not None
        assert discovery.channel is not None

    def test_register_internal_agents(self):
        from backend.src.agriconnect.protocols.a2a import A2ADiscovery

        discovery = A2ADiscovery()
        discovery.register_internal_agents()
        stats = discovery.registry.stats()
        # At least the 5 agents from agent_registry.py
        assert stats["total"] >= 5
        assert stats["active"] >= 5

    def test_discover_agent_by_intent(self):
        from backend.src.agriconnect.protocols.a2a import A2ADiscovery

        discovery = A2ADiscovery()
        discovery.register_internal_agents()
        found = discovery.registry.discover(intent="CHECK_PRICE")
        assert len(found) >= 1
        assert any(c.name == "MarketCoach" for c in found)
