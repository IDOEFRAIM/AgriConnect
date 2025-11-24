"""
prompt_builder.py - Intelligent Prompt Builder for RAG
Améliorations v2:
- Token budgeting précis avec tiktoken
- Stratégies de truncation multi-niveaux
- Templates personnalisables par agent
- Gestion avancée des citations et provenance
- Formatage markdown/structured/plain
- Compression intelligente du contexte
- Validation et sanitization robuste
- Métriques détaillées et monitoring
- Support multi-agent avec fusion intelligente
- Cache de tokenization pour performance
- Stratégies de priorité avancées (score, recency, diversity)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
import logging
import math
import hashlib
import json
import re
from enum import Enum
from functools import lru_cache

_logger = logging.getLogger("prompt_builder")

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None
    TIKTOKEN_AVAILABLE = False
    _logger.warning("tiktoken not available. Install: pip install tiktoken")


# ==============================================================================
# ENUMS & CONFIGURATION
# ==============================================================================

class TruncateStrategy(Enum):
    """Stratégies de troncature avec priorités différentes"""
    DROP_LOW_SCORE = "drop_low_score"          # Supprime docs/snippets score bas
    TRUNCATE_SNIPPETS = "truncate_snippets"    # Tronque les snippets individuellement
    TRUNCATE_DOCS = "truncate_docs"            # Tronque les documents
    SMART_COMPRESS = "smart_compress"          # Compression intelligente multi-passes
    HIERARCHICAL = "hierarchical"              # Garde structure, réduit détails


class SnippetPriority(Enum):
    """Priorités pour sélection des snippets"""
    SCORE = "score"              # Par score de similarité
    RECENCY = "recency"          # Par date (plus récent)
    LENGTH = "length"            # Par longueur (plus long = plus d'info)
    DIVERSITY = "diversity"      # Diversité des sources
    BALANCED = "balanced"        # Mix score + diversité


class OutputFormat(Enum):
    """Formats de sortie"""
    PLAIN = "plain"              # Texte simple
    MARKDOWN = "markdown"        # Format markdown enrichi
    STRUCTURED = "structured"    # Format structuré (JSON-like)
    XML = "xml"                  # Format XML


@dataclass
class PromptBuilderConfig:
    """Configuration pour le prompt builder"""
    # Token budget
    model_max_tokens: int = 4096
    response_tokens_reserved: int = 512
    context_tokens_target: int = 2048
    allow_overflow: bool = False  # Permet dépassement si critique
    
    # Formatting
    default_snippet_separator: str = "\n\n"
    citation_label_template: str = "[{idx}]"
    output_format: OutputFormat = OutputFormat.MARKDOWN
    include_headers: bool = True
    include_separators: bool = True
    
    # Selection
    max_snippets: int = 10
    max_docs: int = 8
    snippet_priority: SnippetPriority = SnippetPriority.BALANCED
    min_score_threshold: float = 0.0  # Filtre score minimum
    
    # Truncation
    token_estimator: str = "tiktoken"  # "tiktoken" or "approx"
    truncate_strategy: TruncateStrategy = TruncateStrategy.SMART_COMPRESS
    min_snippet_length: int = 50
    max_snippet_length: int = 500
    preserve_complete_sentences: bool = True
    
    # Features
    include_provenance: bool = True
    include_scores: bool = True
    include_metadata: bool = True
    deduplicate_snippets: bool = True
    compress_whitespace: bool = True
    normalize_citations: bool = True
    
    # Advanced
    cache_tokens: bool = True  # Cache token counts
    strict_mode: bool = False  # Mode strict (erreur si dépassement)
    custom_template: Optional[str] = None
    
    # Templates par agent
    agent_templates: Dict[str, str] = field(default_factory=lambda: {
        "synthesis": """You are a synthesis agent. Analyze the query and supporting context to produce a comprehensive, well-sourced answer.

Guidelines:
- Cite sources inline using labels like {citation_format}
- Synthesize information from multiple sources when possible
- Identify patterns and connections across sources
- If evidence is insufficient, state: "Insufficient evidence"
- Be concise but thorough
- Prioritize recent and high-confidence sources

Query: {query}

Context:
{context}

Synthesized Answer:""",
        
        "qa": """You are a QA agent. Answer the question using only the provided context.

Rules:
- Cite sources for all factual claims using {citation_format}
- If you cannot answer from the context, say "Insufficient evidence in provided sources"
- Be precise and factual
- Quote directly when appropriate
- Do not extrapolate beyond what's stated

Question: {query}

Context:
{context}

Answer:""",

        "research": """You are a research agent. Conduct deep analysis of the provided context.

Instructions:
- Cite all sources using {citation_format}
- Compare and contrast information across sources
- Identify consensus vs. disagreement
- Note any limitations or gaps in the evidence
- Provide confidence levels for claims

Research Query: {query}

Context:
{context}

Research Analysis:""",

        "summarize": """You are a summarization agent. Create a concise summary of the key information.

Guidelines:
- Extract main points from context
- Maintain factual accuracy
- Cite sources using {citation_format}
- Organize by themes or chronology
- Keep summary focused and relevant

Topic: {query}

Context:
{context}

Summary:""",
        
        "default": """Use the provided context to answer the query.

Instructions:
- Cite sources using {citation_format}
- If uncertain, say "Insufficient evidence"
- Be accurate and concise
- Stay within the scope of provided context

Query: {query}

Context:
{context}

Answer:"""
    })


# ==============================================================================
# PROMPT BUILDER
# ==============================================================================

class PromptBuilder:
    """
    Constructeur de prompts intelligent pour RAG.
    
    Features principales:
    - Token budgeting précis avec tiktoken
    - Citations formatées et normalisées
    - Truncation intelligente multi-stratégies
    - Support multi-agent avec fusion
    - Métriques détaillées et monitoring
    - Cache de tokenization
    - Validation robuste
    """

    def __init__(self, cfg: Optional[PromptBuilderConfig] = None):
        self.cfg = cfg or PromptBuilderConfig()
        self._tokenizer = None
        self._token_cache = {}
        self._metrics = {
            "prompts_built": 0,
            "truncations": 0,
            "total_tokens": 0,
            "avg_tokens": 0,
            "max_tokens": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Initialize tokenizer
        if TIKTOKEN_AVAILABLE and self.cfg.token_estimator == "tiktoken":
            try:
                # cl100k_base for GPT-4, GPT-3.5-turbo
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
                _logger.info("Using tiktoken for token estimation")
            except Exception as e:
                _logger.warning(f"Failed to load tiktoken: {e}")
                self._tokenizer = None

    # ==================== PUBLIC API ====================

    def build_prompt(
        self,
        agent: str,
        query: str,
        augmented: Dict[str, Any],
        custom_instructions: Optional[str] = None
    ) -> str:
        """
        Construit un prompt pour un agent.
        
        Args:
            agent: Nom de l'agent (synthesis, qa, research, summarize, default)
            query: Requête utilisateur
            augmented: Réponse augmentée avec docs, snippets, citations
            custom_instructions: Instructions additionnelles optionnelles
        
        Returns:
            Prompt formaté et budgeté
        """
        # Validate inputs
        self._validate_inputs(query, augmented)
        
        # Build context
        context = self._build_context(query, augmented)
        
        # Get template
        if self.cfg.custom_template:
            template = self.cfg.custom_template
        else:
            template = self.cfg.agent_templates.get(
                agent,
                self.cfg.agent_templates.get("default")
            )
        
        # Format citation style
        citation_format = self.cfg.citation_label_template.format(idx="N")
        
        # Build prompt
        prompt = template.format(
            query=query,
            context=context,
            citation_format=citation_format
        )
        
        # Add custom instructions if provided
        if custom_instructions:
            prompt = f"{prompt}\n\nAdditional Instructions:\n{custom_instructions}"
        
        # Apply token budget
        prompt = self._apply_token_budget(prompt)
        
        # Update metrics
        tokens = self._estimate_tokens(prompt)
        self._update_metrics(tokens)
        
        return prompt

    def build_multi_agent_prompt(
        self,
        query: str,
        agent_outputs: Dict[str, Any],
        augmented: Dict[str, Any],
        fusion_strategy: str = "synthesis"
    ) -> str:
        """
        Combine les sorties de plusieurs agents avec fusion intelligente.
        
        Args:
            query: Requête utilisateur
            agent_outputs: Dict[agent_name -> output]
            augmented: Contexte augmenté
            fusion_strategy: Stratégie de fusion (synthesis, consensus, hierarchical)
        
        Returns:
            Prompt de synthèse combiné
        """
        # Agent outputs section
        agent_parts = []
        for name, output in agent_outputs.items():
            text = output.get("text") if isinstance(output, dict) else str(output)
            meta = output.get("metadata", {}) if isinstance(output, dict) else {}
            confidence = output.get("confidence", "N/A") if isinstance(output, dict) else "N/A"
            
            header = f"### Agent: {name}"
            if meta:
                meta_str = json.dumps(meta, ensure_ascii=False, indent=2)
                agent_parts.append(f"{header}\nConfidence: {confidence}\nMetadata: {meta_str}\n{text}")
            else:
                agent_parts.append(f"{header}\nConfidence: {confidence}\n{text}")
        
        agents_context = self.cfg.default_snippet_separator.join(agent_parts)
        
        # Augmented context
        augmented_context = self._build_context(query, augmented)
        
        # Combine based on fusion strategy
        if fusion_strategy == "hierarchical":
            combined_context = f"""
## Primary Agent Outputs (High Priority)
{agents_context}

---

## Supporting Context (Reference Material)
{augmented_context}
""".strip()
        elif fusion_strategy == "consensus":
            combined_context = f"""
## Multiple Agent Perspectives
{agents_context}

## Task: Identify consensus and resolve conflicts using supporting context below

## Supporting Context
{augmented_context}
""".strip()
        else:  # synthesis
            combined_context = f"""
## Agent Outputs
{agents_context}

## Supporting Context
{augmented_context}
""".strip()
        
        # Use synthesis template
        template = self.cfg.agent_templates.get(
            "synthesis",
            self.cfg.agent_templates.get("default")
        )
        
        citation_format = self.cfg.citation_label_template.format(idx="N")
        
        prompt = template.format(
            query=query,
            context=combined_context,
            citation_format=citation_format
        )
        
        prompt = self._apply_token_budget(prompt)
        
        self._metrics["prompts_built"] += 1
        
        return prompt

    # ==================== CONTEXT BUILDING ====================

    def _build_context(
        self,
        query: str,
        augmented: Dict[str, Any]
    ) -> str:
        """Construit le contexte à partir de la réponse augmentée"""
        docs = augmented.get("docs", []) or []
        snippets = augmented.get("snippets", []) or []
        citations = augmented.get("citations", []) or []
        
        # Filter by score threshold
        if self.cfg.min_score_threshold > 0:
            docs = [d for d in docs if d.get("score", 0) >= self.cfg.min_score_threshold]
            snippets = [s for s in snippets if s.get("score", 0) >= self.cfg.min_score_threshold]
        
        # Select and sort snippets
        snippets_selected = self._select_snippets(snippets)
        
        # Select docs
        docs_selected = self._select_docs(docs)
        
        # Build citation map
        citation_map = self._build_citation_map(citations, docs_selected)
        
        # Format context based on output format
        if self.cfg.output_format == OutputFormat.MARKDOWN:
            context = self._format_markdown(
                docs_selected, snippets_selected, citation_map
            )
        elif self.cfg.output_format == OutputFormat.STRUCTURED:
            context = self._format_structured(
                docs_selected, snippets_selected, citation_map
            )
        elif self.cfg.output_format == OutputFormat.XML:
            context = self._format_xml(
                docs_selected, snippets_selected, citation_map
            )
        else:  # PLAIN
            context = self._format_plain(
                docs_selected, snippets_selected, citation_map
            )
        
        if not context.strip():
            context = "No supporting documents found."
        
        return context

    def _select_snippets(
        self,
        snippets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Sélectionne et ordonne les snippets avec stratégie avancée"""
        if not snippets:
            return []
        
        # Deduplication
        if self.cfg.deduplicate_snippets:
            snippets = self._deduplicate_snippets(snippets)
        
        # Sort by priority
        priority = self.cfg.snippet_priority
        
        if priority == SnippetPriority.SCORE:
            snippets_sorted = sorted(
                snippets,
                key=lambda s: s.get("score", 0.0),
                reverse=True
            )
        elif priority == SnippetPriority.RECENCY:
            snippets_sorted = sorted(
                snippets,
                key=lambda s: s.get("meta", {}).get("date", ""),
                reverse=True
            )
        elif priority == SnippetPriority.LENGTH:
            snippets_sorted = sorted(
                snippets,
                key=lambda s: len(s.get("text", "")),
                reverse=True
            )
        elif priority == SnippetPriority.DIVERSITY:
            snippets_sorted = self._diversify_snippets(snippets)
        elif priority == SnippetPriority.BALANCED:
            snippets_sorted = self._balanced_selection(snippets)
        else:
            snippets_sorted = snippets
        
        # Select top N
        selected = snippets_sorted[:self.cfg.max_snippets]
        
        return selected

    def _select_docs(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sélectionne les documents par score"""
        if not docs:
            return []
        
        docs_sorted = sorted(
            docs,
            key=lambda d: d.get("score", 0.0),
            reverse=True
        )
        
        return docs_sorted[:self.cfg.max_docs]

    def _deduplicate_snippets(
        self,
        snippets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Déduplique les snippets similaires par hashing"""
        seen_hashes = set()
        unique = []
        
        for snippet in snippets:
            text = snippet.get("text", "")
            # Hash normalisé (lowercase, whitespace)
            normalized = re.sub(r'\s+', ' ', text.lower().strip())
            text_hash = hashlib.md5(normalized.encode()).hexdigest()
            
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique.append(snippet)
        
        return unique

    def _diversify_snippets(
        self,
        snippets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Sélectionne des snippets diversifiés par source"""
        # Group by doc_id
        by_doc = {}
        for s in snippets:
            doc_id = s.get("doc_id")
            if doc_id not in by_doc:
                by_doc[doc_id] = []
            by_doc[doc_id].append(s)
        
        # Take best from each doc alternately
        result = []
        max_per_doc = max(1, self.cfg.max_snippets // len(by_doc)) if by_doc else 1
        
        for doc_id, doc_snippets in by_doc.items():
            sorted_snippets = sorted(
                doc_snippets,
                key=lambda s: s.get("score", 0.0),
                reverse=True
            )
            result.extend(sorted_snippets[:max_per_doc])
        
        # Sort final list by score
        result.sort(key=lambda s: s.get("score", 0.0), reverse=True)
        
        return result[:self.cfg.max_snippets]

    def _balanced_selection(
        self,
        snippets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Sélection équilibrée: score + diversité"""
        # First, get diverse set
        diverse = self._diversify_snippets(snippets)
        
        # Then, ensure top scores are included
        by_score = sorted(
            snippets,
            key=lambda s: s.get("score", 0.0),
            reverse=True
        )
        
        # Merge: top 50% by score, 50% by diversity
        top_n = self.cfg.max_snippets // 2
        
        result_ids = set()
        result = []
        
        # Add top scores
        for s in by_score[:top_n]:
            sid = id(s)
            if sid not in result_ids:
                result.append(s)
                result_ids.add(sid)
        
        # Add diverse snippets
        for s in diverse:
            if len(result) >= self.cfg.max_snippets:
                break
            sid = id(s)
            if sid not in result_ids:
                result.append(s)
                result_ids.add(sid)
        
        return result

    def _build_citation_map(
        self,
        citations: List[Dict[str, Any]],
        docs: List[Dict[str, Any]]
    ) -> Dict[Any, str]:
        """Construit la map ID -> label de citation normalisée"""
        citation_map = {}
        
        # From citations
        for i, cit in enumerate(citations, start=1):
            cit_id = cit.get("id") or cit.get("doc_id")
            label = self.cfg.citation_label_template.format(idx=i)
            citation_map[cit_id] = label
        
        # Fill from docs if not in citations
        idx = len(citations) + 1
        for doc in docs:
            doc_id = doc.get("id")
            if doc_id not in citation_map:
                label = self.cfg.citation_label_template.format(idx=idx)
                citation_map[doc_id] = label
                idx += 1
        
        return citation_map

    # ==================== FORMATTING ====================

    def _format_markdown(
        self,
        docs: List[Dict[str, Any]],
        snippets: List[Dict[str, Any]],
        citation_map: Dict[Any, str]
    ) -> str:
        """Formate en Markdown enrichi"""
        parts = []
        
        # Documents section
        if docs and self.cfg.include_headers:
            parts.append("### 📄 Source Documents\n")
            
        for doc in docs:
            doc_id = doc.get("id")
            label = citation_map.get(doc_id, "[?]")
            score = doc.get("score", 0.0)
            
            # Metadata
            meta = doc.get("meta", {})
            prov = self._format_provenance(meta)
            
            # Header with score badge
            score_badge = "🟢" if score > 0.8 else "🟡" if score > 0.5 else "🔴"
            header = f"**{label}** {score_badge} (relevance: {score:.3f})"
            if prov:
                header += f"\n{prov}"
            
            # Text excerpt
            text = doc.get("text", "")
            excerpt = self._truncate_text(text, 300)
            
            parts.append(f"{header}\n{excerpt}\n")
        
        # Snippets section
        if snippets:
            if self.cfg.include_headers:
                parts.append("### 🔍 Relevant Excerpts\n")
                
            for snippet in snippets:
                doc_id = snippet.get("doc_id")
                label = citation_map.get(doc_id, "[?]")
                score = snippet.get("score", 0.0)
                text = snippet.get("text", "")
                
                snippet_text = self._truncate_text(
                    text,
                    self.cfg.max_snippet_length
                )
                
                score_badge = "🟢" if score > 0.8 else "🟡" if score > 0.5 else "🔴"
                parts.append(
                    f"{label} {score_badge} (relevance: {score:.3f})\n> {snippet_text}\n"
                )
        
        return "\n".join(parts)

    def _format_structured(
        self,
        docs: List[Dict[str, Any]],
        snippets: List[Dict[str, Any]],
        citation_map: Dict[Any, str]
    ) -> str:
        """Formate en structure JSON-like"""
        output = {"documents": [], "snippets": []}
        
        for doc in docs:
            doc_id = doc.get("id")
            label = citation_map.get(doc_id, "[?]")
            
            doc_info = {
                "label": label,
                "id": doc_id,
                "score": round(doc.get("score", 0.0), 3),
                "text": self._truncate_text(doc.get("text", ""), 300)
            }
            
            if self.cfg.include_metadata:
                doc_info["metadata"] = doc.get("meta", {})
            
            output["documents"].append(doc_info)
        
        for snippet in snippets:
            snippet_info = {
                "label": citation_map.get(snippet.get("doc_id"), "[?]"),
                "doc_id": snippet.get("doc_id"),
                "score": round(snippet.get("score", 0.0), 3),
                "text": self._truncate_text(
                    snippet.get("text", ""),
                    self.cfg.max_snippet_length
                )
            }
            output["snippets"].append(snippet_info)
        
        return json.dumps(output, ensure_ascii=False, indent=2)

    def _format_xml(
        self,
        docs: List[Dict[str, Any]],
        snippets: List[Dict[str, Any]],
        citation_map: Dict[Any, str]
    ) -> str:
        """Formate en XML"""
        parts = ["<context>"]
        
        if docs:
            parts.append("  <documents>")
            for doc in docs:
                doc_id = doc.get("id")
                label = citation_map.get(doc_id, "[?]")
                score = doc.get("score", 0.0)
                text = self._truncate_text(doc.get("text", ""), 300)
                
                parts.append(f'    <document id="{doc_id}" label="{label}" score="{score:.3f}">')
                parts.append(f"      <text>{self._escape_xml(text)}</text>")
                parts.append("    </document>")
            parts.append("  </documents>")
        
        if snippets:
            parts.append("  <snippets>")
            for snippet in snippets:
                doc_id = snippet.get("doc_id")
                label = citation_map.get(doc_id, "[?]")
                score = snippet.get("score", 0.0)
                text = self._truncate_text(snippet.get("text", ""), self.cfg.max_snippet_length)
                
                parts.append(f'    <snippet doc_id="{doc_id}" label="{label}" score="{score:.3f}">')
                parts.append(f"      <text>{self._escape_xml(text)}</text>")
                parts.append("    </snippet>")
            parts.append("  </snippets>")
        
        parts.append("</context>")
        return "\n".join(parts)

    def _format_plain(
        self,
        docs: List[Dict[str, Any]],
        snippets: List[Dict[str, Any]],
        citation_map: Dict[Any, str]
    ) -> str:
        """Formate en texte simple"""
        parts = []
        
        for doc in docs:
            doc_id = doc.get("id")
            label = citation_map.get(doc_id, "[?]")
            score = doc.get("score", 0.0)
            
            meta = doc.get("meta", {})
            prov = self._format_provenance(meta)
            
            header = f"{label} doc_id={doc_id} score={score:.4f}"
            if prov:
                header += f" {prov}"
            
            text = self._truncate_text(doc.get("text", ""), 300)
            parts.append(f"{header}\n{text}")
        
        for snippet in snippets:
            doc_id = snippet.get("doc_id")
            label = citation_map.get(doc_id, "[?]")
            score = snippet.get("score", 0.0)
            
            text = self._truncate_text(
                snippet.get("text", ""),
                self.cfg.max_snippet_length
            )
            
            parts.append(f"{label} (score={score:.4f})\n{text}")
        
        return self.cfg.default_snippet_separator.join(parts)

    def _format_provenance(self, meta: Dict[str, Any]) -> str:
        """Formate les métadonnées de provenance"""
        if not meta or not self.cfg.include_provenance:
            return ""
        
        parts = []
        for key in ["source", "url", "date", "author", "title"]:
            if key in meta and meta[key]:
                parts.append(f"{key}:{meta[key]}")
        
        return f"[{', '.join(parts)}]" if parts else ""

    def _escape_xml(self, text: str) -> str:
        """Escape XML special characters"""
        return (text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&apos;"))

    # ==================== TOKEN BUDGETING ====================

    def _apply_token_budget(self, prompt: str) -> str:
        """Applique le budget de tokens avec validation"""
        max_allowed = self.cfg.model_max_tokens - self.cfg.response_tokens_reserved
        current_tokens = self._estimate_tokens(prompt)
        
        if current_tokens <= max_allowed:
            return prompt
        
        if self.cfg.strict_mode and not self.cfg.allow_overflow:
            raise ValueError(
                f"Prompt exceeds token budget: {current_tokens} > {max_allowed} tokens. "
                f"Enable allow_overflow or use a different truncate_strategy."
            )
        
        self._metrics["truncations"] += 1
        _logger.warning(
            f"Prompt exceeds budget: {current_tokens} > {max_allowed} tokens. "
            f"Applying {self.cfg.truncate_strategy.value} strategy."
        )
        
        strategy = self.cfg.truncate_strategy
        
        if strategy == TruncateStrategy.SMART_COMPRESS:
            return self._smart_compress(prompt, max_allowed)
        elif strategy == TruncateStrategy.TRUNCATE_SNIPPETS:
            return self._truncate_snippets_strategy(prompt, max_allowed)
        elif strategy == TruncateStrategy.TRUNCATE_DOCS:
            return self._truncate_docs_strategy(prompt, max_allowed)
        elif strategy == TruncateStrategy.HIERARCHICAL:
            return self._hierarchical_truncate(prompt, max_allowed)
        else:  # DROP_LOW_SCORE
            return self._drop_low_score_strategy(prompt, max_allowed)

    def _smart_compress(self, prompt: str, max_tokens: int) -> str:
        """Compression intelligente multi-passes du prompt"""
        # Pass 1: Remove extra whitespace
        if self.cfg.compress_whitespace:
            prompt = re.sub(r'\n{3,}', '\n\n', prompt)
            prompt = re.sub(r' {2,}', ' ', prompt)
        
        current = self._estimate_tokens(prompt)
        if current <= max_tokens:
            return prompt
        
        # Pass 2: Truncate long snippets
        lines = prompt.splitlines()
        compressed = []
        budget_used = 0
        
        for line in lines:
            line_tokens = self._estimate_tokens(line)
            
            # Si la ligne est trop longue, la tronquer intelligemment
            if line_tokens > 200:  # Threshold for long lines
                remaining = max_tokens - budget_used
                if remaining > 50:
                    if self.cfg.preserve_complete_sentences:
                        line = self._truncate_to_sentence(line, min(remaining, 200))
                    else:
                        line = self._truncate_text_by_tokens(line, min(remaining, 200))
                    line += " [...]"
                    line_tokens = self._estimate_tokens(line)
            
            if budget_used + line_tokens > max_tokens:
                compressed.append("[... content truncated to fit token budget ...]")
                break
            
            compressed.append(line)
            budget_used += line_tokens
        
        return "\n".join(compressed)

    def _hierarchical_truncate(self, prompt: str, max_tokens: int) -> str:
        """Truncation hiérarchique: garde structure, réduit détails"""
        # Split into sections
        sections = re.split(r'\n#{1,3}\s+', prompt)
        
        result = []
        budget_used = 0
        
        for i, section in enumerate(sections):
            section_tokens = self._estimate_tokens(section)
                        
            # Si la section dépasse le budget restant
            if budget_used + section_tokens > max_tokens:
                # Essayer de garder seulement l'en-tête (première ligne)
                header = section.split('\n')[0]
                header_tokens = self._estimate_tokens(header)
                
                if budget_used + header_tokens + 10 <= max_tokens:
                    result.append(f"{header}\n[... details truncated ...]")
                    budget_used += header_tokens + 5
                break
            
            result.append(section)
            budget_used += section_tokens
            
        return "\n".join(result)

    def _drop_low_score_strategy(self, prompt: str, max_tokens: int) -> str:
        """Stratégie: retire les blocs entiers en commençant par la fin (supposés moins pertinents)"""
        # On suppose que le prompt est construit avec les éléments les plus importants en premier
        # ou que _build_context a déjà trié par score.
        
        current_tokens = self._estimate_tokens(prompt)
        if current_tokens <= max_tokens:
            return prompt
            
        # Séparer par le séparateur par défaut
        parts = prompt.split(self.cfg.default_snippet_separator)
        
        # Garder l'instruction système/query (généralement le début et la fin du prompt)
        # Cette heuristique dépend du template, ici on tente une approche itérative simple
        
        while len(parts) > 1 and current_tokens > max_tokens:
            # On retire l'avant-dernier élément (souvent le dernier snippet de contexte)
            # On suppose que le dernier élément est "Answer:" ou l'instruction finale
            removed_part = parts.pop(-2) 
            current_tokens -= self._estimate_tokens(removed_part)
            
        return self.cfg.default_snippet_separator.join(parts)

    def _truncate_snippets_strategy(self, prompt: str, max_tokens: int) -> str:
        """Stratégie: raccourcit tous les snippets progressivement"""
        lines = prompt.splitlines()
        new_lines = []
        current_tokens = 0
        
        # On alloue un budget proportionnel
        estimated_lines = len(lines)
        avg_budget_per_line = max(10, max_tokens // estimated_lines)
        
        for line in lines:
            line_tokens = self._estimate_tokens(line)
            
            if current_tokens + line_tokens > max_tokens:
                # Si on dépasse, on tronque la ligne courante agressivement
                remaining = max(0, max_tokens - current_tokens - 5)
                if remaining > 10:
                    truncated = self._truncate_text_by_tokens(line, remaining)
                    new_lines.append(truncated + "...")
                break
            
            new_lines.append(line)
            current_tokens += line_tokens
            
        return "\n".join(new_lines)

    def _truncate_docs_strategy(self, prompt: str, max_tokens: int) -> str:
        """Stratégie: tronque drastiquement les documents, garde les méta-données"""
        # Similaire à drop_low_score mais essaie de garder les titres
        return self._hierarchical_truncate(prompt, max_tokens)

    # ==================== TEXT UTILS ====================

    def _truncate_to_sentence(self, text: str, max_tokens: int) -> str:
        """Tronque le texte en essayant de finir sur une phrase complète"""
        if not text:
            return ""
            
        estimated_chars = max_tokens * 4  # Approx 4 chars/token
        if len(text) <= estimated_chars:
            return text
            
        # Coupe brute
        truncated = text[:estimated_chars]
        
        # Cherche la dernière ponctuation de fin de phrase
        last_punct = -1
        for p in ['.', '!', '?', '\n']:
            pos = truncated.rfind(p)
            if pos > last_punct:
                last_punct = pos
                
        if last_punct > estimated_chars * 0.5:  # Si on ne perd pas trop de texte
            return truncated[:last_punct+1]
            
        return truncated

    def _truncate_text(self, text: str, max_length: int) -> str:
        """Tronque le texte (helper pour le formatage)"""
        if not text:
            return ""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."

    def _truncate_text_by_tokens(self, text: str, max_tokens: int) -> str:
        """Tronque le texte précisément par nombre de tokens"""
        if not text:
            return ""
            
        if self._tokenizer:
            try:
                tokens = self._tokenizer.encode(text)
                if len(tokens) <= max_tokens:
                    return text
                return self._tokenizer.decode(tokens[:max_tokens])
            except Exception:
                pass
                
        # Fallback approximation
        return text[:max_tokens * 4]

    # ==================== TOKEN ESTIMATION & METRICS ====================

    def _estimate_tokens(self, text: str) -> int:
        """
        Estime le nombre de tokens.
        Utilise le cache si activé et tiktoken si disponible.
        """
        if not text:
            return 0
            
        # Check cache
        if self.cfg.cache_tokens:
            # Use simple hash for cache key
            text_hash = hash(text)
            if text_hash in self._token_cache:
                self._metrics["cache_hits"] += 1
                return self._token_cache[text_hash]
            self._metrics["cache_misses"] += 1

        count = 0
        if self._tokenizer:
            try:
                count = len(self._tokenizer.encode(text))
            except Exception:
                # Fallback if encoding fails
                count = len(text) // 4
        else:
            # Approx: 1 token ~= 4 chars in English
            count = len(text) // 4
            
        # Update cache
        if self.cfg.cache_tokens:
            self._token_cache[text_hash] = count
            
        return max(1, count)

    def _update_metrics(self, token_count: int) -> None:
        """Met à jour les statistiques d'utilisation"""
        self._metrics["total_tokens"] += token_count
        self._metrics["max_tokens"] = max(self._metrics["max_tokens"], token_count)
        
        count = self._metrics["prompts_built"] or 1
        self._metrics["avg_tokens"] = self._metrics["total_tokens"] / count

    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques de performance"""
        return self._metrics.copy()

    def _validate_inputs(self, query: str, augmented: Dict[str, Any]) -> None:
        """Validation défensive des entrées"""
        if not query or not isinstance(query, str):
            _logger.warning("Empty or invalid query provided")
            
        if not isinstance(augmented, dict):
            raise ValueError("Augmented context must be a dictionary")
            
        # Ensure required keys exist
        for key in ["docs", "snippets", "citations"]:
            if key not in augmented:
                augmented[key] = []

    def clear_cache(self) -> None:
        """Vide le cache de tokens"""
        self._token_cache.clear()
        self._metrics["cache_hits"] = 0
        self._metrics["cache_misses"] = 0

        'ig'