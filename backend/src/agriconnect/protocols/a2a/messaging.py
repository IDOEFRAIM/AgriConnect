import json
import logging
import uuid
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional, Set
from enum import Enum
from collections import defaultdict

logger = logging.getLogger("A2A.Messaging")

class MessageType(str, Enum):
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    HANDSHAKE = "handshake"
    HEARTBEAT = "heartbeat"
    ACK = "ack"

class HandshakeStatus(str, Enum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    COUNTER = "counter_offer"
    COMPLETED = "completed"

@dataclass
class A2AMessage:
    message_id: str = ""
    message_type: MessageType = MessageType.REQUEST
    sender_id: str = ""
    receiver_id: str = ""
    intent: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    handshake_status: Optional[HandshakeStatus] = None
    reference_id: str = ""
    zone: str = ""
    crop: str = ""
    priority: int = 0
    ttl: int = 3600
    created_at: str = ""
    expires_at: str = ""
    idempotency_key: str = ""
    schema_version: str = "1.0"

    def __post_init__(self):
        now = datetime.now(timezone.utc)
        if not self.message_id:
            self.message_id = str(uuid.uuid4())[:12]
        if not self.created_at:
            self.created_at = now.isoformat()
        if not self.expires_at:
            self.expires_at = (now + timedelta(seconds=self.ttl)).isoformat()
        if not self.idempotency_key:
            raw = f"{self.sender_id}:{self.intent}:{self.created_at}:{json.dumps(self.payload, sort_keys=True, default=str)}"
            self.idempotency_key = hashlib.sha256(raw.encode()).hexdigest()[:16]
        
        self.intent = self.intent.upper()
        self.zone = self.zone.upper()

    def validate(self) -> Dict[str, Any]:
        errors = []
        if not self.sender_id: errors.append("sender_id requis")
        if not self.intent: errors.append("intent requis")
        if self.message_type == MessageType.HANDSHAKE and not self.handshake_status:
            errors.append("handshake_status requis pour HANDSHAKE")
        if self.message_type == MessageType.RESPONSE and not self.reference_id:
            errors.append("reference_id requis pour RESPONSE")
        
        return {"status": "ok"} if not errors else {"error": "INVALID_MESSAGE", "details": errors}

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["message_type"] = self.message_type.value
        if self.handshake_status:
            d["handshake_status"] = self.handshake_status.value
        return d

    def copy_for_receiver(self, receiver_id: str, override_type: Optional[MessageType] = None) -> "A2AMessage":
        """
        Create a safe copy of this message targeted to `receiver_id`.

        Preserves Enum types and avoids round-tripping through dict/asdict.
        Optionally override the message_type (e.g. BROADCAST).
        """
        return A2AMessage(
            message_id=str(uuid.uuid4())[:12],
            message_type=override_type or self.message_type,
            sender_id=self.sender_id,
            receiver_id=receiver_id,
            intent=self.intent,
            payload=self.payload.copy() if isinstance(self.payload, dict) else self.payload,
            handshake_status=self.handshake_status,
            reference_id=self.reference_id,
            zone=self.zone,
            crop=self.crop,
            priority=self.priority,
            ttl=self.ttl,
            created_at=self.created_at,
            expires_at=self.expires_at,
            idempotency_key=self.idempotency_key,
            schema_version=self.schema_version,
        )

    def create_response(self, payload: Dict[str, Any], status: str = "ok") -> "A2AMessage":
        return A2AMessage(
            message_type=MessageType.RESPONSE,
            sender_id=self.receiver_id,
            receiver_id=self.sender_id,
            intent=self.intent,
            payload={**payload, "status": status},
            reference_id=self.message_id,
            zone=self.zone,
            crop=self.crop,
        )



class A2AChannel:
    def __init__(self):
        self._queues: Dict[str, List[A2AMessage]] = defaultdict(list)
        self._subscriptions: Dict[str, Set[str]] = defaultdict(set)
        self._handlers: Dict[str, Callable] = {}
        self._handshakes: Dict[str, List[A2AMessage]] = defaultdict(list)
        self._message_log: List[A2AMessage] = []
        self._seen_idempotency: Set[str] = set()
        logger.info("🔌 A2A Channel initialisé")

    def send(self, message: A2AMessage) -> str:
        validation = message.validate()
        if "error" in validation:
            logger.warning(f"A2A message rejeté: {validation}")
            return ""

        if not message.receiver_id:
            raise ValueError("receiver_id requis pour send()")

        if message.idempotency_key in self._seen_idempotency:
            return message.message_id
        
        self._seen_idempotency.add(message.idempotency_key)
        self._queues[message.receiver_id].append(message)
        self._message_log.append(message)

        handler = self._handlers.get(message.receiver_id)
        if handler:
            try:
                handler(message)
            except Exception as e:
                logger.error(f"Erreur handler {message.receiver_id}: {e}")

        return message.message_id

    def broadcast(self, message: A2AMessage, topic: Optional[str] = None) -> List[str]:
        """Broadcast a message to all subscribers of a topic.

        This function does NOT mutate the original message. It constructs
        explicit `A2AMessage` copies to preserve Enum types and avoid
        serialization issues when recreating dataclasses from dicts.
        """
        targets = set()

        if topic:
            targets.update(self._subscriptions.get(topic.upper(), []))

        if message.intent:
            if message.zone:
                auto_topic = f"{message.intent}_{message.zone}".upper()
                targets.update(self._subscriptions.get(auto_topic, []))
            targets.update(self._subscriptions.get(message.intent.upper(), []))

        delivered_to: List[str] = []
        for agent_id in targets:
            if agent_id == message.sender_id:
                continue

            # Build a safe copy preserving Enum types
            payload_copy = message.payload.copy() if isinstance(message.payload, dict) else message.payload

            msg_copy = A2AMessage(
                message_id=str(uuid.uuid4())[:12],
                message_type=MessageType.BROADCAST,
                sender_id=message.sender_id,
                receiver_id=agent_id,
                intent=message.intent,
                payload=payload_copy,
                handshake_status=message.handshake_status,
                reference_id=message.reference_id,
                zone=message.zone,
                crop=message.crop,
                priority=message.priority,
                ttl=message.ttl,
                created_at=message.created_at,
                expires_at=message.expires_at,
                idempotency_key=message.idempotency_key,
                schema_version=message.schema_version,
            )

            self._queues[agent_id].append(msg_copy)
            delivered_to.append(agent_id)

        # Log the original message (no mutation)
        self._message_log.append(message)
        return delivered_to

    def receive(self, agent_id: str, limit: int = 10) -> List[A2AMessage]:
        messages = self._queues[agent_id][:limit]
        self._queues[agent_id] = self._queues[agent_id][limit:]
        return messages

    def subscribe(self, agent_id: str, topic: str):
        self._subscriptions[topic.upper()].add(agent_id)
        logger.info(f"📌 {agent_id} abonné à {topic.upper()}")

    def unsubscribe(self, agent_id: str, topic: str):
        self._subscriptions[topic.upper()].discard(agent_id)

    def register_handler(self, agent_id: str, handler: Callable):
        self._handlers[agent_id] = handler

    def initiate_handshake(self, message: A2AMessage, max_turns: int = 5, timeout_sec: int = 300) -> str:
        if not message.receiver_id:
            raise ValueError("receiver_id requis pour initiate_handshake()")

        message.message_type = MessageType.HANDSHAKE
        message.handshake_status = HandshakeStatus.PROPOSED
        
        message.payload["_control"] = {
            "max_turns": max_turns,
            "turns_count": 0,
            "timeout_at": (datetime.now(timezone.utc).timestamp() + timeout_sec)
        }
        
        handshake_id = message.message_id
        self._handshakes[handshake_id].append(message)
        self.send(message)
        return handshake_id

    def respond_handshake(
        self,
        handshake_id: str,
        responder_id: str,
        status: HandshakeStatus,
        payload: Optional[Dict[str, Any]] = None
    ) -> A2AMessage:
        history = self._handshakes.get(handshake_id)
        if not history:
            raise ValueError(f"Handshake {handshake_id} inconnu")

        last_msg = history[-1]
        control = last_msg.payload.get("_control", {})
        
        if datetime.now(timezone.utc).timestamp() > control.get("timeout_at", 0):
            return self._finalize_handshake(handshake_id, responder_id, HandshakeStatus.REJECTED, {"reason": "TIMEOUT"})

        if control.get("turns_count", 0) >= control.get("max_turns", 5):
            return self._finalize_handshake(handshake_id, responder_id, HandshakeStatus.REJECTED, {"reason": "MAX_TURNS"})

        new_payload = payload or {}
        new_payload["_control"] = {**control, "turns_count": control["turns_count"] + 1}

        response = A2AMessage(
            message_type=MessageType.HANDSHAKE,
            sender_id=responder_id,
            receiver_id=last_msg.sender_id,
            intent=last_msg.intent,
            payload=new_payload,
            handshake_status=status,
            reference_id=handshake_id,
            zone=last_msg.zone,
            crop=last_msg.crop
        )

        self._handshakes[handshake_id].append(response)
        self.send(response)
        return response

    def _finalize_handshake(self, hid: str, rid: str, status: HandshakeStatus, payload: dict):
        last_msg = self._handshakes[hid][-1]
        final_msg = A2AMessage(
            message_type=MessageType.HANDSHAKE,
            sender_id=rid,
            receiver_id=last_msg.sender_id,
            intent=last_msg.intent,
            payload=payload,
            handshake_status=status,
            reference_id=hid
        )
        self._handshakes[hid].append(final_msg)
        self.send(final_msg)
        return final_msg

    def stats(self) -> Dict[str, Any]:
        return {
            "pending": sum(len(q) for q in self._queues.values()),
            "subs": sum(len(s) for s in self._subscriptions.values()),
            "total_sent": len(self._message_log),
            "active_handshakes": len([h for h in self._handshakes.values() if h[-1].handshake_status not in [HandshakeStatus.COMPLETED, HandshakeStatus.REJECTED]])
        }