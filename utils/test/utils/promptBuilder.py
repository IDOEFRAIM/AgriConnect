# rag/prompt_builder.py
from __future__ import annotations
import json
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple
import textwrap
import functools

# Prompt Builder: modular, agent-aware, reusable prompt assembly for RAG generator stage.
# Designed to support 6 agents (e.g., meteo, tendance, irrigation, climatology, alerts, general).
# Pure Python, no I/O. Inject templates, few-shot examples, and formatting policies.

# -------------------------
# Types and dataclasses
# -------------------------
@dataclass
class PromptConfig:
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 0.95
    stop_sequences: Tuple[str, ...] = ("</s>",)
    safety_instructions: Optional[str] = None
    citation_style: str = "inline"  # "inline" or "brackets" or "footnote"
    include_snippets: bool = True
    include_citations: bool = True
    max_context_chars: int = 4000  # truncate context if too long
    answer_style: str = "concise"  # "concise", "detailed", "bullet_points"
    language: str = "fr"  # default language
    persona: Optional[str] = None  # optional persona string
    examples_limit: int = 3  # number of few-shot examples to include

@dataclass
class FewShotExample:
    name: str
    input_query: str
    augmented: Dict[str, Any]  # expected augmented structure (docs/snippets/citations)
    output: str  # desired model output for the example
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentSpec:
    name: str
    description: str
    default_config: PromptConfig
    system_instructions: str  # high-level system prompt for the agent
    templates: Dict[str, str] = field(default_factory=dict)  # named templates
    few_shot: List[FewShotExample] = field(default_factory=list)

# -------------------------
# Default agent specs
# -------------------------
def _default_meteo_spec() -> AgentSpec:
    cfg = PromptConfig(
        max_tokens=300,
        temperature=0.0,
        answer_style="concise",
        language="fr",
        persona="assistant météo professionnel, factuel et clair",
        citation_style="inline",
        include_snippets=True,
    )
    sys_inst = (
        "Tu es un assistant météo professionnel. Réponds de façon factuelle, "
        "précise et concise. Priorise les informations temporelles (valid_from, valid_to), "
        "la région, et les paramètres clés (température, vent, précipitations). "
        "Si les informations sont incertaines, indique le degré de confiance."
    )
    templates = {
        "base": (
            "{system_instructions}\n\n"
            "Question: {query}\n\n"
            "Contexte récupéré (documents pertinents):\n{context}\n\n"
            "Instructions:\n"
            "- Réponds en {language}.\n"
            "- Donne une synthèse claire et structurée.\n"
            "- Fournis des citations pour chaque affirmation importante.\n"
            "- Si tu ne sais pas, dis 'Je ne sais pas'.\n\n"
            "Réponse:"
        )
    }
    return AgentSpec(name="meteo", description="Météo et prévisions", default_config=cfg, system_instructions=sys_inst, templates=templates)

def _default_tendance_spec() -> AgentSpec:
    cfg = PromptConfig(max_tokens=400, temperature=0.2, answer_style="detailed", language="fr", persona="analyste de tendances", citation_style="brackets")
    sys_inst = "Tu es un analyste de tendances. Synthétise signaux, explique causes probables et propose actions concrètes."
    templates = {
        "base": (
            "{system_instructions}\n\n"
            "Question: {query}\n\n"
            "Contexte:\n{context}\n\n"
            "Instructions:\n"
            "- Explique les tendances observées et leur signification.\n"
            "- Donne des recommandations actionnables.\n"
            "- Cite les sources utilisées.\n\n"
            "Réponse:"
        )
    }
    return AgentSpec(name="tendance", description="Tendances et analyses", default_config=cfg, system_instructions=sys_inst, templates=templates)

def _default_irrigation_spec() -> AgentSpec:
    cfg = PromptConfig(max_tokens=350, temperature=0.0, answer_style="concise", language="fr", persona="expert irrigation", citation_style="inline")
    sys_inst = "Tu es un expert en irrigation agricole. Fournis recommandations pratiques basées sur prévisions et observations."
    templates = {
        "base": (
            "{system_instructions}\n\n"
            "Question: {query}\n\n"
            "Contexte (observations et prévisions):\n{context}\n\n"
            "Instructions:\n"
            "- Propose actions concrètes (quantité d'eau, timing).\n"
            "- Priorise sécurité des cultures et économie d'eau.\n"
            "- Indique incertitudes et citations.\n\n"
            "Réponse:"
        )
    }
    return AgentSpec(name="irrigation", description="Irrigation et recommandations agricoles", default_config=cfg, system_instructions=sys_inst, templates=templates)

def _default_general_spec() -> AgentSpec:
    cfg = PromptConfig(max_tokens=512, temperature=0.0, answer_style="detailed", language="fr", persona="assistant généraliste", citation_style="inline")
    sys_inst = "Tu es un assistant utile et factuel. Réponds clairement et cite les sources."
    templates = {"base": "{system_instructions}\n\nQuestion: {query}\n\nContexte:\n{context}\n\nRéponse:"}
    return AgentSpec(name="general", description="Assistant général", default_config=cfg, system_instructions=sys_inst, templates=templates)

# Additional agent specs placeholders (alerts, climatology)
def _default_alerts_spec() -> AgentSpec:
    cfg = PromptConfig(max_tokens=200, temperature=0.0, answer_style="concise", language="fr", persona="système d'alerte", citation_style="inline")
    sys_inst = "Tu es un système d'alerte. Priorise clarté, urgence et actions immédiates."
    templates = {"base": "{system_instructions}\n\nQuestion: {query}\n\nContexte:\n{context}\n\nRéponse:"}
    return AgentSpec(name="alerts", description="Alertes et urgences", default_config=cfg, system_instructions=sys_inst, templates=templates)

def _default_climatology_spec() -> AgentSpec:
    cfg = PromptConfig(max_tokens=600, temperature=0.1, answer_style="detailed", language="fr", persona="scientifique climat", citation_style="brackets")
    sys_inst = "Tu es un expert en climatologie. Fournis analyses basées sur données et références, en expliquant incertitudes."
    templates = {"base": "{system_instructions}\n\nQuestion: {query}\n\nContexte:\n{context}\n\nRéponse:"}
    return AgentSpec(name="climatology", description="Climatologie et analyses longues", default_config=cfg, system_instructions=sys_inst, templates=templates)

# Registry of agents
_DEFAULT_AGENT_SPECS: Dict[str, AgentSpec] = {
    "meteo": _default_meteo_spec(),
    "tendance": _default_tendance_spec(),
    "irrigation": _default_irrigation_spec(),
    "general": _default_general_spec(),
    "alerts": _default_alerts_spec(),
    "climatology": _default_climatology_spec(),
}

# -------------------------
# Prompt assembly utilities
# -------------------------
def _truncate_context(context_text: str, max_chars: int) -> str:
    if len(context_text) <= max_chars:
        return context_text
    # naive truncation preserving start and end
    head = context_text[: max_chars // 2]
    tail = context_text[- (max_chars // 2) :]
    return head + "\n\n... [truncated] ...\n\n" + tail

def _format_doc_for_prompt(doc: Dict[str, Any], cfg: PromptConfig) -> str:
    """
    Format a single retrieved document for inclusion in the prompt.
    Expected doc keys: id, text, meta, score, snippet(s), citation label
    """
    parts: List[str] = []
    doc_id = doc.get("id") or doc.get("doc_id") or doc.get("chunk_id") or "<no-id>"
    score = doc.get("score")
    meta = doc.get("meta") or {}
    title = meta.get("title") or meta.get("headline") or meta.get("source") or ""
    header = f"- Document {doc_id}"
    if title:
        header += f" | {title}"
    if score is not None:
        header += f" | score={score:.3f}"
    parts.append(header)
    if cfg.include_snippets and doc.get("snippets"):
        for s in doc.get("snippets", [])[:3]:
            span = s.get("text") if isinstance(s, dict) else str(s)
            parts.append(f"  • Snippet: {span}")
    else:
        text = doc.get("text", "")
        if text:
            snippet = textwrap.shorten(text.replace("\n", " "), width=300, placeholder="...")
            parts.append(f"  • Extrait: {snippet}")
    if cfg.include_citations and doc.get("citation"):
        if cfg.citation_style == "inline":
            parts.append(f"  • Source: {doc.get('citation')}")
        elif cfg.citation_style == "brackets":
            parts.append(f"  • Source [{doc.get('citation')}]")
        else:
            parts.append(f"  • Source: {doc.get('citation')}")
    # include minimal meta keys
    meta_keys = ["published_at", "forecast_time", "valid_from", "valid_to", "region"]
    meta_info = []
    for k in meta_keys:
        if meta.get(k):
            meta_info.append(f"{k}={meta.get(k)}")
    if meta_info:
        parts.append("  • Meta: " + "; ".join(meta_info))
    return "\n".join(parts)

def _assemble_context(docs: Iterable[Dict[str, Any]], cfg: PromptConfig) -> str:
    lines: List[str] = []
    for d in docs:
        lines.append(_format_doc_for_prompt(d, cfg))
    ctx = "\n\n".join(lines)
    return _truncate_context(ctx, cfg.max_context_chars)

# -------------------------
# PromptBuilder class
# -------------------------
class PromptBuilder:
    def __init__(self, agent_specs: Optional[Dict[str, AgentSpec]] = None):
        self.agent_specs = agent_specs or _DEFAULT_AGENT_SPECS

    def get_agent_spec(self, agent_name: str) -> AgentSpec:
        if agent_name in self.agent_specs:
            return self.agent_specs[agent_name]
        # fallback to general
        return self.agent_specs["general"]

    def build_prompt(
        self,
        query: str,
        augmented: Optional[Dict[str, Any]] = None,
        agent: str = "general",
        cfg_overrides: Optional[Dict[str, Any]] = None,
        include_examples: bool = True,
    ) -> Dict[str, Any]:
        """
        Build a prompt payload ready to send to a generator LLM.
        Returns dict: {"prompt": str, "config": PromptConfig, "metadata": {...}}
        augmented: expected shape {
            "docs": [ {id, text, meta, score, snippets, citation}, ... ],
            "snippets": [...],
            "citations": [...]
        }
        """
        spec = self.get_agent_spec(agent)
        cfg = spec.default_config
        if cfg_overrides:
            cfg = _merge_config(cfg, cfg_overrides)

        # assemble context from augmented docs
        docs = (augmented or {}).get("docs", []) if augmented else []
        context = _assemble_context(docs, cfg) if docs else "Aucun document pertinent trouvé."

        # build examples if requested
        examples_text = ""
        if include_examples and spec.few_shot:
            exs = spec.few_shot[: cfg.examples_limit]
            ex_lines = []
            for ex in exs:
                ex_ctx = _assemble_context(ex.augmented.get("docs", []), cfg) if ex.augmented else ""
                ex_lines.append("=== Exemple: " + ex.name + " ===")
                ex_lines.append("Question: " + ex.input_query)
                if ex_ctx:
                    ex_lines.append("Contexte:\n" + ex_ctx)
                ex_lines.append("Réponse attendue:\n" + ex.output.strip())
                ex_lines.append("")  # spacer
            examples_text = "\n\n".join(ex_lines)

        # choose template
        template = spec.templates.get("base") or "{system_instructions}\n\nQuestion: {query}\n\nContexte:\n{context}\n\nRéponse:"
        prompt_text = template.format(
            system_instructions=spec.system_instructions,
            query=query,
            context=context,
            language=cfg.language,
        )

        # attach safety and persona
        safety = cfg.safety_instructions or ""
        persona = cfg.persona or spec.default_config.persona or ""
        header_parts = []
        if persona:
            header_parts.append(f"Persona: {persona}")
        if safety:
            header_parts.append(f"Safety: {safety}")
        header = "\n".join(header_parts)
        if header:
            prompt_text = header + "\n\n" + prompt_text

        # append examples and final instruction block
        final_instructions = _build_final_instructions(cfg)
        full_prompt_parts = [prompt_text]
        if examples_text:
            full_prompt_parts.append("Exemples:\n" + examples_text)
        full_prompt_parts.append(final_instructions)
        full_prompt = "\n\n".join(full_prompt_parts)

        # metadata for caller
        metadata = {
            "agent": agent,
            "cfg": asdict(cfg),
            "docs_count": len(docs),
            "timestamp": int(time.time()),
            "query_hash": hashlib.sha256(query.encode("utf-8")).hexdigest()[:12],
        }
        return {"prompt": full_prompt, "config": cfg, "metadata": metadata}

# -------------------------
# Helpers
# -------------------------
def _merge_config(base: PromptConfig, overrides: Dict[str, Any]) -> PromptConfig:
    d = asdict(base)
    d.update(overrides or {})
    # ensure types for tuples
    if "stop_sequences" in d and isinstance(d["stop_sequences"], list):
        d["stop_sequences"] = tuple(d["stop_sequences"])
    return PromptConfig(**d)

def _build_final_instructions(cfg: PromptConfig) -> str:
    parts: List[str] = []
    parts.append("Consignes de formatage:")
    if cfg.answer_style == "concise":
        parts.append("- Réponds en 3 à 6 phrases, claires et directes.")
    elif cfg.answer_style == "bullet_points":
        parts.append("- Fournis la réponse sous forme de points numérotés ou à puces.")
    else:
        parts.append("- Fournis une réponse détaillée, structurée en sections si nécessaire.")
    if cfg.include_citations:
        parts.append("- Pour chaque affirmation factuelle importante, indique la source entre crochets ou en ligne selon le style.")
    parts.append("- N'invente pas d'informations; si l'information manque, indique clairement l'incertitude.")
    parts.append("- Respecte la langue demandée: " + cfg.language)
    return "\n".join(parts)

# -------------------------
# Example usage helpers (pure)
# -------------------------
def build_prompt_for_generator(
    query: str,
    docs: Optional[List[Dict[str, Any]]] = None,
    agent: str = "general",
    cfg_overrides: Optional[Dict[str, Any]] = None,
    include_examples: bool = True,
) -> Dict[str, Any]:
    pb = PromptBuilder()
    augmented = {"docs": docs or []}
    return pb.build_prompt(query=query, augmented=augmented, agent=agent, cfg_overrides=cfg_overrides, include_examples=include_examples)

# -------------------------
# Minimal self-test examples (no I/O)
# -------------------------
if __name__ == "__main__":
    # quick smoke test
    sample_docs = [
        {"id": "d1", "text": "Prévision: pluie légère demain matin.", "meta": {"region": "Casablanca", "forecast_time": "2025-11-25T06:00:00Z"}, "score": 0.92, "snippets": [{"text": "pluie légère demain matin"}], "citation": "DOC1"},
        {"id": "d2", "text": "Bulletin: vent fort attendu sur la côte.", "meta": {"region": "Casablanca", "published_at": "2025-11-24T12:00:00Z"}, "score": 0.78, "snippets": [{"text": "vent fort attendu"}], "citation": "DOC2"},
    ]
    p = build_prompt_for_generator("Quel temps demain à Casablanca ?", docs=sample_docs, agent="meteo")
    print(p["prompt"][:1200])