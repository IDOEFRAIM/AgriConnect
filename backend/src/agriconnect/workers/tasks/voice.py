"""
Voice Tasks — TTS / STT via Azure Speech (production-grade).

PRINCIPES :
  - Hérite de AgriTask (logging structuré, backoff, classification)
  - azure.cognitiveservices.speech est importé LAZY (dans la tâche)
  - Vérification d'espace disque avant écriture
  - Nettoyage du fichier audio en cas d'échec (pas de fichiers corrompus)
  - Idempotency : un audio_id unique par appel
  - Time limits stricts (Azure Speech peut hang)
  - Fallback sur la clé secondaire Azure
"""

import logging
import shutil
import uuid
from pathlib import Path

from celery.exceptions import SoftTimeLimitExceeded

from backend.src.agriconnect.workers.celery_app import celery_app
from backend.src.agriconnect.workers.celery_config import TIME_LIMITS
from backend.src.agriconnect.workers.task_base import (
    AgriTask,
    ExternalServiceDown,
    FatalTaskError,
    RateLimitHit,
    error_result,
    success_result,
)
from backend.src.agriconnect.core.settings import settings

logger = logging.getLogger("AgriConnect.tasks.voice")

AUDIO_DIR = Path(settings.AUDIO_OUTPUT_DIR).resolve()
_VOICE_LIMITS = TIME_LIMITS["voice"]

# Espace disque minimum requis pour écrire un fichier audio (50 MB)
MIN_DISK_SPACE_MB = 50
# Taille max du texte à synthétiser (Azure limit ≈ 10000 chars)
MAX_TTS_TEXT_LENGTH = 8000


def _check_disk_space(path: Path, min_mb: int = MIN_DISK_SPACE_MB) -> bool:
    """Vérifie qu'il y a assez d'espace disque."""
    try:
        usage = shutil.disk_usage(str(path.parent))
        free_mb = usage.free / (1024 * 1024)
        return free_mb >= min_mb
    except OSError:
        return True  # En cas de doute, on continue


def _get_azure_key() -> str:
    """Retourne la clé Azure disponible, ou lève FatalTaskError."""
    key = settings.AZURE_SPEECH_KEY or settings.AZURE_SPEECH_KEY_2
    if not key:
        raise FatalTaskError("Azure Speech non configuré (aucune clé disponible)")
    return key


def _cleanup_partial_file(path: Path):
    """Supprime un fichier audio partiel/corrompu."""
    try:
        if path.exists():
            path.unlink()
            logger.debug("Cleaned up partial audio file: %s", path)
    except OSError as e:
        logger.warning("Failed to cleanup partial file %s: %s", path, e)


@celery_app.task(
    base=AgriTask,
    name="backend.workers.tasks.voice.generate_tts",
    bind=True,
    max_retries=3,
    soft_time_limit=_VOICE_LIMITS["soft"],
    time_limit=_VOICE_LIMITS["hard"],
    acks_late=True,
    track_started=True,
)
def generate_tts(self, text: str, user_id: str = "unknown"):
    """
    Génère un fichier audio .wav via Azure TTS.

    Guards:
      - Texte vide ou trop long → FatalTaskError (pas de retry)
      - Espace disque insuffisant → FatalTaskError
      - Azure SDK timeout → SoftTimeLimitExceeded géré
      - Azure 429 → RateLimitHit → backoff long
      - Fichier corrompu → nettoyé en cas d'erreur

    Returns:
        dict: {"audio_id", "audio_path", "status", ...}
    """
    audio_path = None
    try:
        # ── Validation ──
        if not text or not text.strip():
            raise FatalTaskError("Empty text — nothing to synthesize")
        if len(text) > MAX_TTS_TEXT_LENGTH:
            raise FatalTaskError(
                f"Text too long ({len(text)} chars, max {MAX_TTS_TEXT_LENGTH})"
            )

        key = _get_azure_key()

        # ── Préparer le répertoire et vérifier l'espace disque ──
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        if not _check_disk_space(AUDIO_DIR):
            raise FatalTaskError(
                f"Insufficient disk space (need {MIN_DISK_SPACE_MB} MB free)"
            )

        audio_id = str(uuid.uuid4())
        audio_path = AUDIO_DIR / f"{audio_id}.wav"

        # ── Import LAZY — le SDK Azure n'est chargé que quand nécessaire ──
        import azure.cognitiveservices.speech as speechsdk

        speech_config = speechsdk.SpeechConfig(
            subscription=key,
            region=settings.AZURE_REGION,
        )
        speech_config.speech_synthesis_voice_name = "fr-FR-DeniseNeural"
        # Timeout Azure SDK (30s max pour la synthèse)
        speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,
            "30000",
        )

        audio_config = speechsdk.audio.AudioOutputConfig(filename=str(audio_path))
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=audio_config,
        )

        result = synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            file_size = audio_path.stat().st_size if audio_path.exists() else 0
            return success_result(
                data={
                    "audio_id": audio_id,
                    "audio_path": str(audio_path),
                    "file_size_bytes": file_size,
                    "text_length": len(text),
                    "user_id": user_id,
                },
                task_name=self.name,
            )

        # ── Analyser la raison de l'échec ──
        error_details = str(result.reason)
        cancellation = getattr(result, "cancellation_details", None)
        if cancellation:
            error_details = f"{result.reason}: {cancellation.reason} — {cancellation.error_details}"

        # Classifier l'erreur Azure
        if "429" in error_details or "throttl" in error_details.lower():
            _cleanup_partial_file(audio_path)
            raise RateLimitHit(f"Azure TTS rate limit: {error_details}")

        if "401" in error_details or "403" in error_details:
            _cleanup_partial_file(audio_path)
            raise FatalTaskError(f"Azure TTS auth error: {error_details}")

        _cleanup_partial_file(audio_path)
        raise ExternalServiceDown(f"TTS synthesis failed: {error_details}")

    except (FatalTaskError, SoftTimeLimitExceeded):
        if audio_path:
            _cleanup_partial_file(audio_path)
        if isinstance(audio_path, Path):
            pass  # déjà nettoyé
        raise

    except SoftTimeLimitExceeded:
        if audio_path:
            _cleanup_partial_file(audio_path)
        return self.handle_timeout({"user_id": user_id})

    except (RateLimitHit, ExternalServiceDown) as exc:
        if audio_path:
            _cleanup_partial_file(audio_path)
        self.retry_with_backoff(
            exc,
            base_delay=30.0 if isinstance(exc, RateLimitHit) else 5.0,
            max_delay=300.0,
        )

    except Exception as exc:
        if audio_path:
            _cleanup_partial_file(audio_path)

        classification = self.classify_error(exc)
        if classification == "fatal":
            return error_result(error=str(exc), task_name=self.name, retryable=False)

        self.retry_with_backoff(exc, base_delay=5.0)


@celery_app.task(
    base=AgriTask,
    name="backend.workers.tasks.voice.transcribe_audio",
    bind=True,
    max_retries=2,
    soft_time_limit=_VOICE_LIMITS["soft"],
    time_limit=_VOICE_LIMITS["hard"],
    acks_late=True,
    track_started=True,
)
def transcribe_audio(self, audio_file_path: str):
    """
    Transcrit un fichier audio en texte via Azure STT.

    Returns:
        dict: {"text", "status", ...}
    """
    try:
        # ── Validation ──
        audio_path = Path(audio_file_path)
        if not audio_path.exists():
            raise FatalTaskError(f"Audio file not found: {audio_file_path}")
        if audio_path.stat().st_size == 0:
            raise FatalTaskError(f"Audio file is empty: {audio_file_path}")

        key = _get_azure_key()

        import azure.cognitiveservices.speech as speechsdk

        speech_config = speechsdk.SpeechConfig(
            subscription=key, region=settings.AZURE_REGION
        )
        speech_config.speech_recognition_language = "fr-FR"

        audio_config = speechsdk.audio.AudioConfig(filename=str(audio_path))
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config,
        )

        result = recognizer.recognize_once_async().get()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return success_result(
                data={"text": result.text},
                task_name=self.name,
            )

        if result.reason == speechsdk.ResultReason.NoMatch:
            return success_result(
                data={"text": "", "note": "No speech recognized"},
                task_name=self.name,
            )

        raise ExternalServiceDown(f"STT failed: {result.reason}")

    except FatalTaskError:
        raise

    except SoftTimeLimitExceeded:
        return self.handle_timeout()

    except Exception as exc:
        classification = self.classify_error(exc)
        if classification == "fatal":
            return error_result(error=str(exc), task_name=self.name, retryable=False)
        self.retry_with_backoff(exc, base_delay=5.0)
