"""
MCP RAG Server 2.0 — Observable HyDE + Reranking Pipeline.
============================================================

Upgrades from v1:
  - Records HyDE query generation in TraceEnvelope
  - Records similarity scores per document
  - Records reranking decisions and final selection
  - Full provenance chain: query → HyDE → vector search → rerank → result
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from backend.src.agriconnect.protocols.core import (
    TraceCategory,
    TraceEnvelope,
)

logger = logging.getLogger("MCP.RAG")


class MCPRagServer:
    """
    Serveur MCP exposant le RAG comme un outil standardisé.
    
    Outils exposés :
      - search_agronomy_docs(query, region, level, top_k)
        → Recherche dans la base documentaire agronomique (HyDe + reranking)
      
      - get_doc_sources(query)
        → Retourne les sources utilisées (traçabilité)
    
    Ressources exposées :
      - agri://rag/stats → Statistiques du corpus (nombre docs, dernière mise à jour)
    """

    def __init__(self, retriever=None):
        """
        Args:
            retriever: Instance AgileRetriever existante (réutilisée, pas recréée)
        """
        self._retriever = retriever
        self._tools = {}
        self._resources = {}
        self._register_tools()
        self._register_resources()
        logger.info("🔌 MCP RAG Server initialisé")

    def _lazy_retriever(self):
        """Initialisation paresseuse du retriever (coûteux en mémoire)."""
        if self._retriever is None:
            try:
                from backend.src.agriconnect.rag.retriever import AgileRetriever
                self._retriever = AgileRetriever()
                logger.info("📚 RAG retriever chargé (lazy init)")
            except Exception as e:
                logger.error("RAG retriever unavailable: %s", e)
        return self._retriever

    # ═══════════════════════════════════════════════════════════
    # REGISTRATION
    # ═══════════════════════════════════════════════════════════

    def _register_tools(self):
        self._tools = {
            "search_agronomy_docs": {
                "name": "search_agronomy_docs",
                "description": (
                    "Recherche dans la base de connaissances agronomiques "
                    "d'Afrique de l'Ouest (Burkina Faso, Sahel). "
                    "Utilise HyDe + reranking pour une pertinence maximale."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Question agronomique"},
                        "region": {"type": "string", "description": "Région (ex: Burkina, Sahel)", "default": "Burkina"},
                        "level": {
                            "type": "string",
                            "description": "Niveau de l'agriculteur",
                            "enum": ["debutant", "intermediaire", "expert"],
                            "default": "debutant",
                        },
                        "top_k": {"type": "integer", "description": "Nombre de résultats", "default": 3},
                    },
                    "required": ["query"],
                },
                "handler": self._search_docs,
            },
            "get_doc_sources": {
                "name": "get_doc_sources",
                "description": "Retourne les sources documentaires utilisées pour une recherche (traçabilité)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                },
                "handler": self._get_sources,
            },
        }

    def _register_resources(self):
        self._resources = {
            "agri://rag/stats": {
                "name": "RAG Statistics",
                "description": "Statistiques du corpus documentaire",
                "mime_type": "application/json",
                "handler": self._read_stats,
            },
        }

    # ═══════════════════════════════════════════════════════════
    # INTERFACE MCP
    # ═══════════════════════════════════════════════════════════

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {"name": t["name"], "description": t["description"], "inputSchema": t["input_schema"]}
            for t in self._tools.values()
        ]

    def list_resources(self) -> List[Dict[str, Any]]:
        return [
            {"uri": uri, "name": r["name"], "description": r["description"], "mimeType": r["mime_type"]}
            for uri, r in self._resources.items()
        ]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        tool = self._tools.get(name)
        if not tool:
            return {"error": f"Outil RAG inconnu: {name}", "status": "not_found"}
        try:
            result = tool["handler"](arguments)
            return {
                "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
                "status": "ok",
            }
        except Exception as e:
            logger.error("MCP RAG call_tool error (%s): %s", name, e)
            return {"error": str(e), "status": "error"}

    def read_resource(self, uri: str, params: Dict = None) -> Dict[str, Any]:
        resource = self._resources.get(uri)
        if not resource:
            return {"error": f"Ressource RAG inconnue: {uri}", "status": "not_found"}
        try:
            data = resource["handler"](params or {})
            return {
                "contents": [{"uri": uri, "mimeType": "application/json", "text": json.dumps(data, ensure_ascii=False)}],
                "status": "ok",
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}

    # ═══════════════════════════════════════════════════════════
    # HANDLERS
    # ═══════════════════════════════════════════════════════════

    def _search_docs(self, arguments: Dict[str, Any], trace_envelope: Optional[TraceEnvelope] = None) -> Dict[str, Any]:
        """Recherche agronomique via HyDe + reranking with trace recording."""
        t0 = time.monotonic()
        retriever = self._lazy_retriever()
        if not retriever:
            return {"context": "", "sources": [], "error": "RAG indisponible"}

        query = arguments["query"]
        level = arguments.get("level", "debutant")
        top_k = arguments.get("top_k", 3)

        tone_map = {"debutant": "simple", "intermediaire": "standard", "expert": "technique"}
        tone = tone_map.get(level, "standard")

        try:
            # 1. HyDe : génère un document hypothétique
            t_hyde = time.monotonic()
            hyde_doc = retriever.generate_hyde_doc(query, tone=tone)
            hyde_ms = (time.monotonic() - t_hyde) * 1000

            if trace_envelope:
                trace_envelope.record(
                    TraceCategory.MCP_RAG,
                    "MCPRagServer",
                    "hyde_generate",
                    input_summary={"query": query[:100], "tone": tone},
                    output_summary={"hyde_doc_len": len(hyde_doc)},
                    reasoning=f"HyDE doc generated ({len(hyde_doc)} chars, tone={tone})",
                    duration_ms=hyde_ms,
                )

            # 2. Recherche vectorielle
            t_vec = time.monotonic()
            from llama_index.core import QueryBundle
            query_bundle = QueryBundle(query_str=hyde_doc)

            if retriever.vector_retriever:
                nodes = retriever.vector_retriever.retrieve(query_bundle)
            else:
                nodes = []
            vec_ms = (time.monotonic() - t_vec) * 1000

            raw_scores = [
                {"rank": i + 1, "score": round(n.score, 4) if n.score else None}
                for i, n in enumerate(nodes[:10])
            ]
            if trace_envelope:
                trace_envelope.record(
                    TraceCategory.MCP_RAG,
                    "MCPRagServer",
                    "vector_search",
                    input_summary={"hyde_doc_len": len(hyde_doc)},
                    output_summary={"raw_results": len(nodes), "scores": raw_scores},
                    reasoning=f"Vector search returned {len(nodes)} candidates",
                    duration_ms=vec_ms,
                )

            # 3. Reranking
            t_rerank = time.monotonic()
            nodes = retriever.rerank(query, nodes, top_k=top_k)
            rerank_ms = (time.monotonic() - t_rerank) * 1000

            # 4. Formatage des résultats
            context_parts = []
            sources = []
            reranked_scores = []
            for i, node in enumerate(nodes):
                text = node.node.get_content() if hasattr(node, "node") else str(node)
                meta = node.node.metadata if hasattr(node, "node") else {}
                context_parts.append(f"[Doc {i+1}] {text[:500]}")
                score_val = round(node.score, 4) if node.score else None
                sources.append({
                    "rank": i + 1,
                    "source": meta.get("source", "Inconnu"),
                    "type": meta.get("type", "Document"),
                    "score": score_val,
                })
                reranked_scores.append({"rank": i + 1, "score": score_val, "source": meta.get("source", "?")})

            total_ms = (time.monotonic() - t0) * 1000

            if trace_envelope:
                trace_envelope.record(
                    TraceCategory.MCP_RAG,
                    "MCPRagServer",
                    "rerank_and_format",
                    input_summary={"top_k": top_k, "pre_rerank_count": len(nodes)},
                    output_summary={"final_count": len(sources), "reranked": reranked_scores},
                    reasoning=(
                        f"Reranked to top-{top_k}: "
                        + ", ".join(f"{s['source']}({s['score']})" for s in reranked_scores)
                    ),
                    duration_ms=rerank_ms,
                )

            return {
                "context": "\n\n".join(context_parts),
                "sources": sources,
                "query_used": query,
                "hyde_active": True,
                "results_count": len(nodes),
                "pipeline_ms": round(total_ms, 1),
            }

        except Exception as e:
            logger.error("RAG search error: %s", e)
            if trace_envelope:
                trace_envelope.record(
                    TraceCategory.MCP_RAG,
                    "MCPRagServer",
                    "search_error",
                    input_summary={"query": query[:100]},
                    output_summary={"error": str(e)},
                    reasoning=f"RAG search failed: {e}",
                    duration_ms=(time.monotonic() - t0) * 1000,
                )
            return {"context": "", "sources": [], "error": str(e)}

    def _get_sources(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Retourne les sources documentaires pour traçabilité."""
        result = self._search_docs(arguments)
        return {"sources": result.get("sources", []), "query": arguments.get("query")}

    def _read_stats(self, params: Dict) -> Dict[str, Any]:
        """Statistiques du corpus RAG."""
        retriever = self._lazy_retriever()
        stats = {"status": "unknown", "total_docs": 0}
        if retriever and retriever.index:
            try:
                if hasattr(retriever.index, "_vector_store") and hasattr(retriever.index._vector_store, "client"):
                    stats["total_docs"] = retriever.index._vector_store.client.ntotal
                    stats["status"] = "ready"
                else:
                    stats["status"] = "ready"
            except Exception:
                stats["status"] = "error"
        else:
            stats["status"] = "unavailable"
        return stats
