"""
Protocols AgriConnect 2.0 — Architecture Tri-Protocoles (Hardened)
==================================================================

core          → Shared primitives: TraceEnvelope, CorrelationCtx, AsyncResult, HandshakeRecord
observability → ProtocolTracer: structured decision tracing with DB persistence
MCP           → Système nerveux : accès données, RAG, météo (semantic cache)
A2A           → Réseau social : discovery, négociation, matching (async ACK, FSM)
AG-UI         → Visage universel : rendu adaptatif avec capability negotiation
"""

from .core import (
    CorrelationCtx,
    TraceEnvelope,
    TraceStep,
    TraceCategory,
    AsyncResult,
    AckStatus,
    HandshakeRecord,
    HandshakeFSMError,
    HSState,
    CachePolicy,
    ClientCapabilities,
)
from .observability import ProtocolTracer

__all__ = [
    "mcp", "a2a", "ag_ui",
    # Core primitives
    "CorrelationCtx", "TraceEnvelope", "TraceStep", "TraceCategory",
    "AsyncResult", "AckStatus",
    "HandshakeRecord", "HandshakeFSMError", "HSState",
    "CachePolicy", "ClientCapabilities",
    # Observability
    "ProtocolTracer",
]
