# orchestrator.py
import logging
import time
import json
import os
import signal
from datetime import datetime
from typing import Dict, Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

import config

# Importer tes services (adapter si signatures différentes)
from services.scraper import DocumentScraper, WeatherForecastService, SonagessScraper

# Logger
logger = logging.getLogger("scraper.orchestrator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Paramètres d'orchestration
MAX_WORKERS = getattr(config, "ORCHESTRATOR_MAX_WORKERS", 3)
TASK_TIMEOUT_S = getattr(config, "ORCHESTRATOR_TASK_TIMEOUT_S", 300)  # timeout par tâche
RETRY_ATTEMPTS = getattr(config, "ORCHESTRATOR_RETRY_ATTEMPTS", 3)
RETRY_BACKOFF_BASE = getattr(config, "ORCHESTRATOR_RETRY_BACKOFF_BASE", 2)  # seconds
REPORT_FILE = getattr(config, "ORCHESTRATOR_REPORT_FILE", "orchestration_report.json")


def _normalize_result(res: Any) -> Dict[str, Any]:
    """
    Normalise la sortie d'une tâche en dict:
    { status: "SUCCESS"|"ERROR", results: list, error: optional str }
    """
    if isinstance(res, dict):
        status = res.get("status", "SUCCESS" if res.get("results") is not None else "ERROR")
        results = res.get("results", [])
        error = res.get("error")
        return {"status": status, "results": results or [], "error": error}
    # si la fonction renvoie une liste ou autre
    if isinstance(res, list):
        return {"status": "SUCCESS", "results": res, "error": None}
    if res is None:
        return {"status": "ERROR", "results": [], "error": "No result returned"}
    return {"status": "SUCCESS", "results": [res], "error": None}


def _safe_call_retryable(fn: Callable, *args, attempts: int = RETRY_ATTEMPTS, backoff_base: int = RETRY_BACKOFF_BASE, **kwargs) -> Dict[str, Any]:
    """
    Appelle fn(*args, **kwargs) avec retries simples et backoff exponentiel.
    Retourne un dict normalisé.
    """
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            start = time.time()
            res = fn(*args, **kwargs)
            duration = time.time() - start
            logger.info("Task %s succeeded in %.2fs (attempt %d/%d)", getattr(fn, "__name__", str(fn)), duration, attempt, attempts)
            return _normalize_result(res)
        except Exception as e:
            last_exc = e
            wait = backoff_base ** (attempt - 1)
            logger.warning("Task %s failed on attempt %d/%d: %s. Backing off %ds", getattr(fn, "__name__", str(fn)), attempt, attempts, e, wait)
            time.sleep(wait)
    logger.error("Task %s failed after %d attempts: %s", getattr(fn, "__name__", str(fn)), attempts, last_exc)
    return {"status": "ERROR", "results": [], "error": str(last_exc)}


class ScraperOrchestrator:
    def __init__(self, headless: bool = True):
        self.headless = headless
        # Instancier les services (adapter les signatures si besoin)
        self.document_scraper = DocumentScraper(headless=headless)
        self.weather_service = WeatherForecastService(headless=headless)
        self.sonagess_scraper = SonagessScraper(
            start_url=getattr(config, "START_URL", config.START_URL),
            download=True,
            max_depth=getattr(config, "MAX_DEPTH", config.MAX_DEPTH),
            max_pages=getattr(config, "MAX_PAGES", config.MAX_PAGES),
            output_dir=getattr(config, "OUTPUT_DIR", config.OUTPUT_DIR),
        )

        self.report: Dict[str, Any] = {}
        self._stop_requested = False
        # installer handler pour arrêt propre
        signal.signal(signal.SIGINT, self._handle_stop)
        signal.signal(signal.SIGTERM, self._handle_stop)

    def _handle_stop(self, signum, frame):
        logger.warning("Stop signal reçu (%s). Arrêt propre demandé.", signum)
        self._stop_requested = True

    def _run_task_with_timeout(self, executor: ThreadPoolExecutor, fn: Callable, *args, task_name: str = "task", timeout_s: int = TASK_TIMEOUT_S, **kwargs) -> Dict[str, Any]:
        """
        Soumet la tâche au ThreadPoolExecutor et attend avec timeout.
        Utilise _safe_call_retryable pour retries internes.
        """
        future = executor.submit(_safe_call_retryable, fn, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_s)
            return result
        except TimeoutError:
            logger.error("Timeout: la tâche %s a dépassé %ds", task_name, timeout_s)
            return {"status": "ERROR", "results": [], "error": f"Timeout after {timeout_s}s"}
        except Exception as e:
            logger.exception("Exception inattendue lors de l'exécution de %s: %s", task_name, e)
            return {"status": "ERROR", "results": [], "error": str(e)}

    def run_all(self, save_report: bool = True) -> Dict[str, Any]:
        """
        Exécute les tâches en parallèle (contrôlé), collecte les résultats,
        construit un rapport consolidé et le sauvegarde.
        """
        logger.info("=== DÉMARRAGE ORCHESTRATEUR ===")
        start_time = time.time()
        tasks = {
            "document_scraper": (self.document_scraper.scrape_bulletins, ()),
            "weather_service": (self.weather_service.scrape_forecast, ()),
            "sonagess_scraper": (self.sonagess_scraper.run, ()),
        }

        results: Dict[str, Dict[str, Any]] = {}
        durations: Dict[str, float] = {}

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures_map = {}
            # soumettre toutes les tâches
            for name, (fn, fn_args) in tasks.items():
                if self._stop_requested:
                    logger.warning("Arrêt demandé avant soumission de %s", name)
                    results[name] = {"status": "ERROR", "results": [], "error": "Stopped before start"}
                    continue
                logger.info("Soumission de la tâche %s", name)
                # on encapsule l'appel pour mesurer la durée
                def _timed_call(f, *a, **kw):
                    t0 = time.time()
                    out = _safe_call_retryable(f, *a, **kw)
                    t1 = time.time()
                    return {"out": out, "duration": round(t1 - t0, 2)}
                futures_map[ex.submit(_timed_call, fn, *fn_args)] = name

            # récupérer les résultats au fur et à mesure
            for fut in as_completed(futures_map):
                name = futures_map[fut]
                try:
                    res_wrapper = fut.result(timeout=TASK_TIMEOUT_S)
                    out = res_wrapper.get("out", {"status": "ERROR", "results": [], "error": "No output"})
                    dur = res_wrapper.get("duration", 0.0)
                    results[name] = _normalize_result(out)
                    durations[name] = dur
                    logger.info("Tâche %s terminée en %.2fs (status=%s)", name, dur, results[name].get("status"))
                except TimeoutError:
                    logger.error("Timeout global pour la tâche %s", name)
                    results[name] = {"status": "ERROR", "results": [], "error": f"Timeout after {TASK_TIMEOUT_S}s"}
                    durations[name] = TASK_TIMEOUT_S
                except Exception as e:
                    logger.exception("Erreur lors de la récupération du résultat de %s: %s", name, e)
                    results[name] = {"status": "ERROR", "results": [], "error": str(e)}
                if self._stop_requested:
                    logger.warning("Arrêt demandé : on arrête la collecte des résultats restants.")
                    break

        total_time = round(time.time() - start_time, 2)

        # calcul du statut global
        statuses = [r.get("status", "ERROR") for r in results.values()]
        if all(s == "SUCCESS" for s in statuses):
            overall_status = "SUCCESS"
        elif all(s == "ERROR" for s in statuses):
            overall_status = "FAILURE"
        else:
            overall_status = "PARTIAL_SUCCESS"

        # compter documents collectés
        def _count(r):
            res = r.get("results", [])
            return len(res) if isinstance(res, list) else 0

        total_documents = sum(_count(r) for r in results.values())

        self.report = {
            "overall_status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "total_duration_s": total_time,
            "task_durations_s": durations,
            "data_pipelines": results,
            "total_documents_collected": total_documents,
        }

        # sauvegarde du rapport
        if save_report:
            try:
                with open(REPORT_FILE, "w", encoding="utf-8") as f:
                    json.dump(self.report, f, ensure_ascii=False, indent=2)
                logger.info("Rapport sauvegardé: %s", REPORT_FILE)
            except Exception as e:
                logger.warning("Impossible de sauvegarder le rapport: %s", e)

        logger.info("=== ORCHESTRATION TERMINÉE : %s (%.2fs) ===", overall_status, total_time)
        return self.report


# Exécution directe
if __name__ == "__main__":
    orch = ScraperOrchestrator(headless=True)
    final = orch.run_all(save_report=True)
    print(json.dumps(final, ensure_ascii=False, indent=2))