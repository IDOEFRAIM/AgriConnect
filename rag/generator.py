from __future__ import annotations
import asyncio
import time
import uuid
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Sequence, Type
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError
import functools

_logger = logging.getLogger("generator_adapter")
_logger.setLevel(logging.INFO)

# ==============================================================================
# 1. HIERARCHIE D'ERREURS PERSONNALISÉES (Granularité & Robustesse)
# ==============================================================================
class AdapterError(Exception):
    """Erreur de base pour l'adaptateur de génération."""
    pass

class SafetyError(AdapterError):
    """Erreur levée lorsque le prompt ou le chunk est bloqué par un filtre de sécurité."""
    pass

class CircuitOpenError(AdapterError):
    """Erreur levée lorsque le Circuit Breaker est ouvert et bloque la requête."""
    pass

# ==============================================================================
# 2. DATA CLASSES
# ==============================================================================
@dataclass
class GenerationResult:
    """Résultat standard d'une opération de génération."""
    text: str
    tokens: int = 0
    finish_reason: Optional[str] = None
    raw: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GenOptions:
    """Options configurables pour l'appel à la génération."""
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.0
    stop_sequences: Optional[List[str]] = None
    stream: bool = False # Note: L'adaptateur gère l'appel stream ou non, mais cette option reste pour le backend.
    timeout_s: Optional[float] = 30.0 # Timeout par tentative
    retries: int = 2 # Nombre de réessais (donc tentatives = retries + 1)
    retry_backoff_base_s: float = 0.5 # Base pour l'attente exponentielle entre les réessais
    circuit_breaker_duration_s: float = 10.0 # Durée d'ouverture du circuit après échec critique
    function_calling: Optional[Dict[str, Any]] = None
    model_kwargs: Dict[str, Any] = field(default_factory=dict)

# ==============================================================================
# 3. CIRCUIT BREAKER (Granularité)
# ==============================================================================
class CircuitBreaker:
    """
    Implémentation d'un Disjoncteur pour les services défaillants (Fail-Fast).
    """
    def __init__(self, duration_s: float = 10.0, logger: Optional[logging.Logger] = None):
        self._duration_s = duration_s
        self._circuit_open_until: float = 0.0
        self._lock = threading.Lock()
        self.logger = logger or _logger

    def is_open(self) -> bool:
        """Vérifie si le circuit est actuellement ouvert."""
        with self._lock:
            if time.time() < self._circuit_open_until:
                return True
            # Transition de Half-Open à Closed (temps écoulé)
            if self._circuit_open_until > 0.0:
                 self.logger.info("Circuit Breaker transitioned to CLOSED (timeout elapsed).")
                 self._circuit_open_until = 0.0 # Réinitialise pour éviter les checks inutiles
            return False

    def trip(self) -> None:
        """Ouvre le circuit après une défaillance."""
        with self._lock:
            self._circuit_open_until = time.time() + self._duration_s
            self.logger.warning("Circuit Breaker TRIPED, opened for %.1fs.", self._duration_s)

    def __enter__(self):
        """Bloque l'exécution si le circuit est ouvert."""
        if self.is_open():
            raise CircuitOpenError("Circuit Breaker is OPEN. Request blocked (fail-fast).")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Déclenche le circuit si une exception se produit."""
        if exc_type is not None and not issubclass(exc_type, (asyncio.CancelledError, SafetyError)):
            self.trip()
        return False # Ne supprime pas l'exception


# ==============================================================================
# 4. PROTOCOLE BACKEND
# ==============================================================================
class GeneratorBackend:
    """
    Contrat pour les adaptateurs de backend concrets.
    Les implémentations DOIVENT lever des exceptions standard de Python
    (e.g., ConnectionError, TimeoutError, RuntimeError) en cas de défaillance.
    """
    def generate(self, prompt: str, opts: GenOptions) -> GenerationResult:
        """Génération synchrone d'une seule réponse."""
        raise NotImplementedError

    async def stream_generate(self, prompt: str, opts: GenOptions) -> AsyncIterator[str]:
        """Génération asynchrone par morceaux."""
        raise NotImplementedError

    def shutdown(self) -> None:
        """Nettoyage des ressources du backend."""
        pass

# ==============================================================================
# 5. EXEMPLES DE BACKEND (Granularité)
# ==============================================================================
class DummyBackend(GeneratorBackend):
    """Backend factice pour les tests."""
    def generate(self, prompt: str, opts: GenOptions) -> GenerationResult:
        time.sleep(0.01)
        text = f"Dummy response for: {prompt[:200]}"
        return GenerationResult(text=text, tokens=len(text.split()), finish_reason="stop", raw={"dummy": True})

    async def stream_generate(self, prompt: str, opts: GenOptions) -> AsyncIterator[str]:
        parts = ["Dummy", " response", " streaming", " for: ", prompt[:120]]
        for p in parts:
            await asyncio.sleep(0.01)
            yield p

class HFBackend(GeneratorBackend):
    """
    Exemple de Backend basé sur HuggingFace (synchrone).
    Démontre la gestion des imports lourds et l'isolation des dépendances.
    """
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs
        self._model = None
        self._tokenizer = None
        self._lock = threading.RLock()
        self._is_initialized = False

    def _lazy_init_model(self):
        """Initialisation différée (Lazy Initialization) des dépendances lourdes."""
        if self._is_initialized:
            return

        with self._lock:
            if self._is_initialized:
                return
            try:
                # Importations ici pour éviter de les charger si le backend n'est pas utilisé
                import importlib
                transformers = importlib.import_module("transformers")
                AutoTokenizer = getattr(transformers, "AutoTokenizer")
                AutoModel = getattr(transformers, "AutoModelForCausalLM", None) or getattr(transformers, "AutoModelForSeq2SeqLM", None)
                
                # Exemple de chargement (simplifié)
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
                self._model = AutoModel.from_pretrained(self.model_name, device_map="auto", **self.config)
                self._model.eval()
                
                _logger.info("HFBackend loaded model %s", self.model_name)
                self._is_initialized = True
            except ImportError:
                 _logger.error("HFBackend: Les dépendances (e.g., transformers, torch) ne sont pas installées.")
                 raise RuntimeError("Dépendances HF manquantes.")
            except Exception:
                _logger.exception("HFBackend init failed.")
                raise RuntimeError("Échec du chargement du modèle HF.")

    def generate(self, prompt: str, opts: GenOptions) -> GenerationResult:
        self._lazy_init_model()
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Modèle HF non chargé ou échec de l'initialisation.")
        
        # Logique de génération réelle (ici simulée)
        try:
            # Code de génération réel...
            time.sleep(0.1) 
            text = self._tokenizer.decode(self._tokenizer.encode(prompt)[:opts.max_tokens], skip_special_tokens=True)
            text += f" [Réponse HF simulée, MaxTokens={opts.max_tokens}]"
            return GenerationResult(text=text, tokens=len(text.split()), finish_reason="stop", raw={"hf_config": self.config})
        except Exception as e:
            # Lève une erreur standard qui sera gérée par le GeneratorAdapter (retry/circuit)
            raise RuntimeError(f"Erreur de génération HF: {e}")

    async def stream_generate(self, prompt: str, opts: GenOptions) -> AsyncIterator[str]:
        # Implémentation réelle nécessiterait un wrapper async sur le générateur de streaming HF.
        # Ici, simulation d'un streaming naif pour satisfaire le contrat.
        res = self.generate(prompt, opts)
        text = res.text or ""
        chunk_size = 64
        for i in range(0, len(text), chunk_size):
            await asyncio.sleep(0.01)
            yield text[i:i+chunk_size]

    def shutdown(self) -> None:
        self._model = None
        self._tokenizer = None
        self._is_initialized = False

# ==============================================================================
# 6. ADAPTATEUR PRINCIPAL (Scalabilité & Orchestration)
# ==============================================================================
class GeneratorAdapter:
    """
    Adaptateur de générateur robuste et résilient :
    - Orchestre les backends (synchrone/asynchrone).
    - Gère la concurrence (Semaphore, ThreadPoolExecutor).
    - Met en œuvre la résilience (Retries/Backoff/Circuit Breaker).
    - Applique des contrôles de sécurité.
    """
    def __init__(
        self,
        backend: GeneratorBackend,
        *,
        default_opts: Optional[GenOptions] = None,
        concurrency: int = 8,
        thread_workers: int = 4, # Utilisé pour les backends synchrones comme HFBackend
        safety_filters: Optional[List[Callable[[str], bool]]] = None,
        metrics_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.backend = backend
        self.opts = default_opts or GenOptions()
        self.logger = logger or _logger

        # Concurrence (Scalabilité)
        self._sem = asyncio.Semaphore(concurrency)
        self._executor = ThreadPoolExecutor(max_workers=max(1, thread_workers))

        # Résilience (Granularité/Industrialisation)
        self.circuit_breaker = CircuitBreaker(duration_s=self.opts.circuit_breaker_duration_s, logger=self.logger)
        
        # Sécurité
        self.safety_filters = safety_filters or []
        self.metrics_hook = metrics_hook
        
        self._running = True

    # ---------- Gestion des erreurs et retries ----------
    def _retry_logic(self, func: Callable[..., GenerationResult], prompt: str, opts: GenOptions, request_id: str) -> GenerationResult:
        """
        Gère les tentatives, les réessais avec backoff exponentiel et les timeouts.
        Exécuté dans un ThreadPoolExecutor (pour le backend synchrone).
        """
        attempt = 0
        last_exc: Optional[Exception] = None
        
        while attempt <= opts.retries:
            start_attempt = time.time()
            try:
                # 1. Utilisation du Circuit Breaker pour bloquer l'appel si nécessaire
                with self.circuit_breaker:
                    # 2. Exécution de l'appel au backend avec Timeout
                    future: Future = self._executor.submit(func, prompt, opts)
                    res: GenerationResult = future.result(timeout=opts.timeout_s)
                    
                    # Succès
                    res.metadata.setdefault("request_id", request_id)
                    res.metadata.setdefault("diagnostics", {})
                    res.metadata["diagnostics"].update({
                        "latency_s": time.time() - start_attempt, 
                        "attempts": attempt + 1
                    })
                    if self.metrics_hook:
                        self.metrics_hook({"event": "generate", "latency_s": time.time() - start_attempt, "request_id": request_id, "status": "success"})
                    return res
            
            except CircuitOpenError as e:
                raise e # Remonte immédiatement, Circuit Breaker est géré plus tôt.
            except (TimeoutError, Exception) as e:
                # 3. Gestion des échecs (Timeout ou Backend Error)
                last_exc = e
                self.logger.warning("generate attempt %d failed (%.2fs): %s", attempt + 1, time.time() - start_attempt, type(e).__name__)
                
                attempt += 1
                if attempt <= opts.retries:
                    backoff_time = opts.retry_backoff_base_s * (2 ** (attempt - 1))
                    self.logger.debug("Waiting for %.2fs before retry...", backoff_time)
                    time.sleep(backoff_time)
        
        # 4. Échec après tous les réessais
        self.circuit_breaker.trip()
        raise RuntimeError(f"Génération échouée après {opts.retries + 1} tentatives. Dernière erreur: {str(last_exc)}")


    # ---------- Fonctions d'Orchestration ----------
    def generate(self, prompt: str, override_opts: Optional[GenOptions] = None, request_id: Optional[str] = None) -> GenerationResult:
        """Point d'entrée synchrone pour la génération non-streaming."""
        opts = override_opts or self.opts
        request_id = request_id or str(uuid.uuid4())
        start = time.time()

        try:
            self._check_prompt_safety(prompt)
            
            # Utilise la logique de retry qui contient l'exécution dans le pool de threads
            res = self._retry_logic(self.backend.generate, prompt, opts, request_id)
            return res

        except SafetyError as e:
            self.logger.error("Request blocked by safety filter: %s", request_id)
            return GenerationResult(text="", tokens=0, finish_reason="blocked", metadata={"request_id": request_id, "error": str(e)})
        except CircuitOpenError as e:
            self.logger.warning("Request blocked by circuit breaker: %s", request_id)
            return GenerationResult(text="", tokens=0, finish_reason="circuit_open", metadata={"request_id": request_id, "error": str(e)})
        except RuntimeError as e:
            # Catch l'échec final du _retry_logic (Circuit Breaker a été déclenché)
            self.logger.error("Final generation failure: request_id=%s, error=%s", request_id, str(e))
            return GenerationResult(text="", tokens=0, finish_reason="error", metadata={"request_id": request_id, "error": str(e)})
        except Exception as e:
            # Catch tout autre imprévu (e.g., erreur dans l'adaptateur lui-même)
            self.logger.critical("Uncaught critical error in adapter: %s", e)
            return GenerationResult(text="", tokens=0, finish_reason="critical_error", metadata={"request_id": request_id, "error": str(e)})


    async def stream_generate(self, prompt: str, override_opts: Optional[GenOptions] = None, request_id: Optional[str] = None) -> AsyncIterator[str]:
        """Point d'entrée asynchrone pour la génération streaming."""
        opts = override_opts or self.opts
        request_id = request_id or str(uuid.uuid4())

        try:
            self._check_prompt_safety(prompt)
        except SafetyError as e:
            yield f"[ERROR] prompt blocked: {str(e)}"
            return

        # Le circuit breaker est géré dans le contexte du retry ci-dessous
        async with self._sem: # Limite de concurrence asynchrone
            start = time.time()
            attempt = 0
            last_exc = None
            
            while attempt <= opts.retries:
                try:
                    # Vérifie si le circuit est ouvert avant d'appeler
                    if self.circuit_breaker.is_open():
                        raise CircuitOpenError("Circuit Breaker is OPEN.")
                        
                    # 1. Appel du backend (potentiellement async)
                    stream = self.backend.stream_generate(prompt, opts)
                    if asyncio.iscoroutine(stream):
                        stream = await stream
                    
                    # 2. Streaming des chunks
                    async for chunk in stream:
                        if not self._chunk_allowed(chunk):
                            self.logger.debug("stream chunk blocked by safety filter")
                            continue
                        yield chunk
                        
                    # Succès
                    if self.metrics_hook:
                         self.metrics_hook({"event": "stream_complete", "request_id": request_id, "latency_s": time.time() - start, "status": "success"})
                    return # Sortie du générateur

                except CircuitOpenError as e:
                    yield f"[ERROR] circuit_open request_id={request_id}"
                    return
                except Exception as e:
                    # 3. Gestion des échecs (retry/backoff)
                    last_exc = e
                    self.logger.warning("stream attempt %d failed: %s", attempt + 1, type(e).__name__)
                    
                    attempt += 1
                    if attempt <= opts.retries:
                        backoff_time = opts.retry_backoff_base_s * (2 ** (attempt - 1))
                        await asyncio.sleep(backoff_time)
                    else:
                        # 4. Échec final, déclenchement du circuit
                        self.circuit_breaker.trip()
                        yield f"[STREAM_ERROR] Generation failed after retries: {str(last_exc)}"
                        return

    # ---------- Helpers Sécurité ----------
    def _check_prompt_safety(self, prompt: str) -> None:
        """Vérifie si le prompt initial est autorisé."""
        for f in self.safety_filters:
            try:
                if not f(prompt):
                    raise SafetyError("Prompt bloqué par le filtre de sécurité.")
            except Exception as e:
                self.logger.error("Safety filter failure: %s", e)
                raise SafetyError("Échec du filtre de sécurité lors de la vérification du prompt.")

    def _chunk_allowed(self, chunk: str) -> bool:
        """Vérifie si un chunk de streaming est autorisé."""
        for f in self.safety_filters:
            try:
                if not f(chunk):
                    return False
            except Exception:
                # Si le filtre échoue, bloquez par défaut pour des raisons de sécurité.
                return False
        return True

    # ---------- Cycle de Vie ----------
    def shutdown(self) -> None:
        """Arrêt propre de l'adaptateur et des ressources."""
        self._running = False
        try:
            self.backend.shutdown()
        except Exception as e:
            self.logger.error("Backend shutdown error: %s", e)
        try:
            self._executor.shutdown(wait=False)
        except Exception as e:
            self.logger.error("Executor shutdown error: %s", e)


            'ig'