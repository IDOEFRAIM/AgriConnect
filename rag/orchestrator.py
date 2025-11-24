from __future__ import annotations
import asyncio
import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Callable, Union

# Tentative d'import pour la propagation de contexte (si logger_module est présent)
try:
    from logger_module import set_request_context
except ImportError:
    set_request_context = None

_logger = logging.getLogger("orchestrator")

# ---------- Types ----------
Doc = Dict[str, Any]
AugmentedResponse = Dict[str, Any]
GenerationResult = Dict[str, Any]

# ---------- Config ----------
@dataclass
class OrchestratorConfig:
    agent_order: Sequence[str] = ("domain", "qa", "synthesis")
    round_timeout_s: float = 8.0
    max_retries: int = 2
    concurrency_limit: int = 8
    top_k: int = 50
    rerank_top_n: int = 20
    request_timeout_s: float = 30.0
    backoff_base_s: float = 0.5
    circuit_breaker_duration_s: float = 10.0
    circuit_breaker_threshold: int = 5  # Nombre d'échecs consécutifs avant ouverture
    semaphore_per_agent: int = 4
    metrics_hook: Optional[Callable[[Dict[str, Any]], None]] = None  # Hook pour monitoring externe

# ---------- Orchestrator ----------
class Orchestrator:
    """
    Orchestrateur robuste coordonnant le pipeline RAG : Retriever -> Augmenter -> Agents -> Generator.
    
    Philosophie Industrielle :
    - Résilience : Retries exponentiels, Circuit Breaker, Timeouts stricts.
    - Observabilité : Diagnostics détaillés, Hooks de métriques, Logs structurés.
    - Concurrence : Sémaphores globaux et par agent pour éviter la surcharge.
    """

    def __init__(
        self,
        cfg: OrchestratorConfig,
        agents: Dict[str, Any],
        retriever: Any,
        augmenter: Any,
        reranker: Optional[Any],
        prompt_builder: Any,
        generator: Any,
        evaluator: Optional[Any] = None,
        logger: Optional[Any] = None,
    ):
        self.cfg = cfg
        self.agents = agents
        self.retriever = retriever
        self.augmenter = augmenter
        self.reranker = reranker
        self.prompt_builder = prompt_builder
        self.generator = generator
        self.evaluator = evaluator
        self.logger = logger or _logger
        
        # Concurrency control
        self._sem = asyncio.Semaphore(cfg.concurrency_limit)
        self._agent_semaphores: Dict[str, asyncio.Semaphore] = {
            name: asyncio.Semaphore(cfg.semaphore_per_agent) for name in agents.keys()
        }
        
        # Circuit Breaker state
        self._circuit_open_until: Optional[float] = None
        self._failure_count: int = 0
        self._circuit_lock = asyncio.Lock()
        
        # Internal metrics
        self._metrics = {
            "requests_total": 0,
            "requests_failed": 0,
            "circuit_trips": 0
        }

    # Public entrypoint
    async def handle_query(self, query: str, ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Point d'entrée principal. Exécute le flux RAG complet.
        """
        # 1. Validation et Initialisation
        if not query or not isinstance(query, str):
            return {"error": "invalid_input", "message": "Query must be a non-empty string"}

        start_total = time.time()
        request_id = (ctx or {}).get("request_id") or str(uuid.uuid4())
        user_id = (ctx or {}).get("user_id") or "anonymous"
        
        # Enrichissement du contexte pour les logs structurés
        ctx = {**(ctx or {}), "request_id": request_id, "user_id": user_id}
        if set_request_context:
            set_request_context(ctx)

        self._metrics["requests_total"] += 1
        diagnostics: Dict[str, Any] = {"stages": [], "request_id": request_id}

        # 2. Vérification Circuit Breaker
        if await self._circuit_open():
            self._emit_metric("circuit_breaker_reject", 1)
            return {
                "error": "circuit_open", 
                "message": "Service temporairement indisponible (circuit ouvert)",
                "diagnostics": diagnostics
            }

        try:
            # --- STAGE 1: Domain Analysis (Optional) ---
            filters = {}
            domain_agent = self.agents.get("domain")
            if domain_agent and hasattr(domain_agent, "decide_filters"):
                try:
                    t0 = time.time()
                    filters = await self._call_with_retries(domain_agent.decide_filters, query, ctx, stage="domain")
                    latency = time.time() - t0
                    diagnostics["stages"].append({"stage": "domain", "latency_s": latency, "filters": filters})
                    self._emit_metric("stage_latency", latency, tags={"stage": "domain"})
                except Exception as e:
                    self._log_warn("domain agent failed (non-critical)", request_id, e)
                    diagnostics["stages"].append({"stage": "domain", "error": str(e)})

            # --- STAGE 2: Retrieval & Augmentation ---
            try:
                t0 = time.time()
                augmented: AugmentedResponse = await self._call_with_retries(
                    self.augmenter.get_augmented, query, filters, ctx, stage="augment"
                )
                latency = time.time() - t0
                doc_count = len(augmented.get("docs", []))
                diagnostics["stages"].append({
                    "stage": "augment", 
                    "latency_s": latency, 
                    "candidates": doc_count
                })
                self._emit_metric("stage_latency", latency, tags={"stage": "augment"})
                self._emit_metric("retrieved_docs", doc_count)
                
            except Exception as e:
                self._log_exception("augmenter failed (critical)", request_id, e)
                self._metrics["requests_failed"] += 1
                return {"error": "augment_failed", "message": str(e), "diagnostics": diagnostics}

            # --- STAGE 3: Parallel Agent Generation ---
            agent_tasks: Dict[str, asyncio.Task] = {}
            for name in self.cfg.agent_order:
                agent = self.agents.get(name)
                if not agent:
                    continue
                
                # Préparation du contexte spécifique à l'agent (post-retrieve hook)
                try:
                    t0 = time.time()
                    docs_for_agent = await self._maybe_call(
                        agent.post_retrieve if hasattr(agent, "post_retrieve") else None, 
                        query, augmented.get("docs", []), ctx
                    )
                    # Si post_retrieve renvoie None ou échoue, utiliser tous les docs
                    if docs_for_agent is None:
                        docs_for_agent = augmented.get("docs", [])
                        
                    diagnostics["stages"].append({
                        "stage": f"{name}.post_retrieve", 
                        "latency_s": time.time() - t0
                    })
                except Exception as e:
                    self._log_warn(f"{name}.post_retrieve failed", request_id, e)
                    docs_for_agent = augmented.get("docs", [])

                # Construction du prompt
                prompt_text = self.prompt_builder.build_prompt(
                    name, 
                    query, 
                    {
                        "docs": docs_for_agent, 
                        "snippets": augmented.get("snippets", []), 
                        "citations": augmented.get("citations", [])
                    }
                )
                
                # Lancement de la tâche asynchrone
                agent_tasks[name] = asyncio.create_task(
                    self._run_agent_generate(name, agent, prompt_text, ctx, docs_for_agent)
                )

            # Collecte des résultats avec timeout global pour cette étape
            agent_results: Dict[str, Any] = {}
            if agent_tasks:
                try:
                    t0 = time.time()
                    res_map = await asyncio.wait_for(
                        self._gather_tasks(agent_tasks), 
                        timeout=self.cfg.round_timeout_s
                    )
                    agent_results.update(res_map)
                    
                    latency = time.time() - t0
                    diagnostics["stages"].append({
                        "stage": "agents_round", 
                        "latency_s": latency, 
                        "agents_count": len(agent_results)
                    })
                    self._emit_metric("stage_latency", latency, tags={"stage": "agents_round"})
                    
                except asyncio.TimeoutError:
                    self._log_warn("agent round timed out", request_id, "timeout")
                    self._emit_metric("agent_round_timeout", 1)
                    
                    # Récupération partielle des tâches terminées
                    for n, t in agent_tasks.items():
                        if t.done():
                            try:
                                agent_results[n] = t.result()
                            except Exception as e:
                                agent_results[n] = {"error": str(e)}
                        else:
                            t.cancel() # Annuler les tâches en cours
                            agent_results[n] = {"error": "timeout"}
                    
                    diagnostics["stages"].append({"stage": "agents_round", "error": "timeout"})

            # --- STAGE 4: Synthesis ---
            final_answer: Dict[str, Any] = {}
            synthesis_agent = self.agents.get("synthesis")
            
            if synthesis_agent and hasattr(synthesis_agent, "generate"):
                try:
                    t0 = time.time()
                    multi_prompt = self.prompt_builder.build_multi_agent_prompt(
                        query, 
                        agent_results, 
                        {
                            "docs": augmented.get("docs", []), 
                            "snippets": augmented.get("snippets", []), 
                            "citations": augmented.get("citations", [])
                        }
                    )
                    final_answer = await self._call_with_retries(
                        synthesis_agent.generate, multi_prompt, ctx, stage="synthesis"
                    )
                    latency = time.time() - t0
                    diagnostics["stages"].append({"stage": "synthesis", "latency_s": latency})
                    self._emit_metric("stage_latency", latency, tags={"stage": "synthesis"})
                    
                except Exception as e:
                    self._log_exception("synthesis failed", request_id, e)
                    final_answer = self._fallback_select(agent_results)
                    diagnostics["stages"].append({"stage": "synthesis", "error": str(e), "fallback": True})
            else:
                final_answer = self._fallback_select(agent_results)

            # --- STAGE 5: Evaluation (Fire & Forget / Best Effort) ---
            if self.evaluator:
                # On ne bloque pas la réponse utilisateur pour l'évaluation
                asyncio.create_task(self._run_evaluation(query, final_answer, augmented, request_id, diagnostics))

            # Succès global
            await self._reset_circuit_failure_count()
            total_latency = time.time() - start_total
            diagnostics["total_latency_s"] = total_latency
            self._emit_metric("request_latency", total_latency)
            
            return {
                "answer": final_answer, 
                "agent_outputs": agent_results, 
                "augmented": augmented, 
                "diagnostics": diagnostics
            }

        except Exception as e:
            self._metrics["requests_failed"] += 1
            self._log_exception("Global orchestrator failure", request_id, e)
            return {"error": "internal_error", "message": str(e), "diagnostics": diagnostics}

    # ---------- Helpers internes ----------

    async def _run_agent_generate(
        self, name: str, agent: Any, prompt: str, ctx: Dict[str, Any], docs: List[Doc]
    ) -> Dict[str, Any]:
        """Exécute la génération d'un agent avec sémaphore spécifique."""
        sem = self._agent_semaphores.get(name, self._sem)
        async with sem:
            try:
                t0 = time.time()
                # Prefer agent.generate
                if hasattr(agent, "generate"):
                    out = await self._call_with_retries(agent.generate, prompt, ctx, stage=f"agent_{name}")
                else:
                    # Fallback to shared generator adapter
                    out = await self._call_with_retries(self.generator.generate, prompt, {"ctx": ctx}, stage=f"agent_{name}")
                
                latency = time.time() - t0
                
                # Standardization de la sortie
                if not isinstance(out, dict):
                    out = {"text": str(out)}
                
                out.setdefault("metadata", {})
                out["metadata"]["latency_s"] = latency
                # Snapshot des docs utilisés pour traçabilité
                out["metadata"]["docs_snapshot"] = [d.get("id") for d in docs[:5]]
                
                return out
            except Exception as e:
                self._log_exception(f"agent {name} generate failed", ctx.get("request_id"), e)
                return {"error": str(e)}

    async def _gather_tasks(self, tasks: Dict[str, asyncio.Task]) -> Dict[str, Any]:
        """Attend toutes les tâches et collecte les résultats/erreurs."""
        if not tasks:
            return {}
        
        done, _ = await asyncio.wait(list(tasks.values()), return_when=asyncio.ALL_COMPLETED)
        res = {}
        for name, task in tasks.items():
            try:
                res[name] = task.result()
            except asyncio.CancelledError:
                res[name] = {"error": "cancelled"}
            except Exception as e:
                res[name] = {"error": str(e)}
        return res

    async def _call_with_retries(self, fn: Callable, *args, **kwargs):
        """Appelle une fonction avec retry exponentiel et jitter."""
        last_exc = None
        stage = kwargs.pop("stage", "unknown")
        
        for attempt in range(self.cfg.max_retries + 1):
            try:
                res = fn(*args, **kwargs)
                if asyncio.iscoroutine(res):
                    return await asyncio.wait_for(res, timeout=self.cfg.request_timeout_s)
                return res
            except Exception as e:
                last_exc = e
                is_last_attempt = attempt == self.cfg.max_retries
                
                if not is_last_attempt:
                    # Exponential backoff with jitter (0.05 * attempt)
                    backoff = self.cfg.backoff_base_s * (2 ** attempt) + (0.05 * attempt)
                    self._log_warn(
                        f"call {getattr(fn, '__name__', str(fn))} failed (attempt {attempt+1}/{self.cfg.max_retries+1})", 
                        kwargs.get("ctx", {}).get("request_id"), 
                        e
                    )
                    await asyncio.sleep(backoff)
                else:
                    self._log_warn(f"call {getattr(fn, '__name__', str(fn))} final failure", kwargs.get("ctx", {}).get("request_id"), e)

        # Si tous les retries échouent, incrémenter le compteur de pannes du circuit breaker
        await self._record_failure()
        raise last_exc

    async def _maybe_call(self, fn: Optional[Callable], *args, **kwargs):
        """Appelle une fonction optionnelle de manière sûre."""
        if fn is None:
            return args[0] if args else None
        try:
            res = fn(*args, **kwargs)
            if asyncio.iscoroutine(res):
                return await asyncio.wait_for(res, timeout=self.cfg.request_timeout_s)
            return res
        except Exception as e:
            self._log_warn(f"maybe_call failed for {getattr(fn, '__name__', str(fn))}", kwargs.get("ctx", {}).get("request_id"), e)
            return args[0] if args else None

    async def _run_evaluation(self, query: str, final_answer: Any, augmented: Dict, request_id: str, diagnostics: Dict):
        """Exécute l'évaluation en arrière-plan."""
        if not self.evaluator:
            return
        try:
            t0 = time.time()
            response_text = final_answer.get("text") if isinstance(final_answer, dict) else str(final_answer)
            
            eval_report = await self._maybe_call(
                self.evaluator.evaluate_single, 
                {"query": query, "response": response_text, "docs": augmented.get("docs", [])}
            )
            
            # On ne peut pas modifier 'diagnostics' du scope parent thread-safe facilement si la requête est déjà retournée,
            # mais on peut logger ou envoyer la métrique.
            latency = time.time() - t0
            self._emit_metric("evaluation_latency", latency)
            
            # Log du rapport d'évaluation
            if eval_report:
                self.logger.info("Evaluation completed", extra={"request_id": request_id, "report": eval_report})
                
        except Exception as e:
            self._log_warn("background evaluation failed", request_id, e)

    def _fallback_select(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Sélectionne une réponse de repli si la synthèse échoue."""
        # 1. Essayer de trouver une réponse textuelle valide d'un agent
        for name, out in results.items():
            if isinstance(out, dict) and out.get("text") and not out.get("error"):
                return {
                    "text": out["text"], 
                    "metadata": {**out.get("metadata", {}), "fallback_source": name}
                }
        
        # 2. Sinon, agréger tout ce qu'on a
        texts = []
        for name, out in results.items():
            if isinstance(out, dict) and out.get("text"):
                texts.append(f"Agent {name}: {out['text']}")
        
        if texts:
            return {"text": "\n\n".join(texts)[:2000], "metadata": {"fallback": True, "method": "aggregation"}}
            
        return {"text": "Service unavailable due to internal errors.", "metadata": {"fallback": True, "error": "no_agent_output"}}

    # ---------- Circuit Breaker Logic ----------

    async def _record_failure(self) -> None:
        """Enregistre une défaillance critique."""
        async with self._circuit_lock:
            self._failure_count += 1
            if self._failure_count >= self.cfg.circuit_breaker_threshold:
                self._circuit_open_until = time.time() + self.cfg.circuit_breaker_duration_s
                self._metrics["circuit_trips"] += 1
                self.logger.error(
                    f"Circuit breaker TRIPPED. Open for {self.cfg.circuit_breaker_duration_s}s. Failures: {self._failure_count}"
                )

    async def _reset_circuit_failure_count(self) -> None:
        """Réinitialise le compteur de défaillances en cas de succès."""
        if self._failure_count > 0:
            async with self._circuit_lock:
                self._failure_count = 0

    async def _circuit_open(self) -> bool:
        """Vérifie si le circuit est ouvert."""
        async with self._circuit_lock:
            if self._circuit_open_until is None:
                return False
            
            if time.time() > self._circuit_open_until:
                self.logger.info("Circuit breaker recovering. Testing service availability.")
                self._circuit_open_until = None
                self._failure_count = 0 # Reset on recovery attempt (or use half-open logic)
                return False
            
            return True

    # ---------- Logging & Metrics Helpers ----------

    def _log_warn(self, msg: str, request_id: Optional[str], exc: Any) -> None:
        try:
            self.logger.warning(msg, extra={"request_id": request_id, "error": str(exc)})
        except Exception:
            _logger.warning(msg)

    def _log_exception(self, msg: str, request_id: Optional[str], exc: Any) -> None:
        try:
            self.logger.exception(msg, extra={"request_id": request_id, "error": str(exc)})
        except Exception:
            _logger.exception(msg)

    def _emit_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        if self.cfg.metrics_hook:
            try:
                metric = {"name": name, "value": value, "timestamp": time.time()}
                if tags:
                    metric.update(tags)
                self.cfg.metrics_hook(metric)
            except Exception:
                pass

    # ---------- Lifecycle ----------
    
    def get_metrics(self) -> Dict[str, int]:
        """Retourne les métriques internes."""
        return self._metrics.copy()

    async def health(self) -> Dict[str, Any]:
        try:
            ok = True
            parts = {"orchestrator": "ok"}
            
            # Probe components
            if hasattr(self.retriever, "health"):
                try:
                    r = await self._maybe_call(self.retriever.health)
                    parts["retriever"] = r
                    if isinstance(r, dict) and not r.get("ok", True):
                        ok = False
                except Exception as e:
                    parts["retriever"] = {"ok": False, "error": str(e)}
                    ok = False
            
            return {"ok": ok, "parts": parts, "circuit_open": await self._circuit_open()}
        except Exception:
            return {"ok": False, "error": "health_check_failed"}

    async def shutdown(self) -> None:
        # Best-effort shutdown of components
        components = [self.generator, self.retriever, self.augmenter, self.reranker]
        for comp in components:
            if comp and (hasattr(comp, "shutdown") or hasattr(comp, "close")):
                try:
                    method = getattr(comp, "shutdown", getattr(comp, "close", None))
                    if method:
                        await self._maybe_call(method)
                except Exception:
                    pass


                'ig'