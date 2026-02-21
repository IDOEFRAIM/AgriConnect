import uuid
import logging
from pathlib import Path
from typing import Optional, Tuple
import azure.cognitiveservices.speech as speechsdk

logger = logging.getLogger(__name__)

class VoiceEngine:
    """
    Moteur de voix robuste pour AgriConnect.
    Gère le TTS (Text-to-Speech) avec système de secours (fallback).
    """

    def __init__(
        self, 
        api_key: str, 
        region: str = "westeurope", 
        fallback_key: Optional[str] = None,
        storage_dir: str = "./audio_output"
    ):
        self.api_key = api_key
        self.fallback_key = fallback_key
        self.region = region
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

    def _get_config(self, use_fallback: bool = False) -> speechsdk.SpeechConfig:
        key = self.fallback_key if use_fallback and self.fallback_key else self.api_key
        if not key:
            raise ValueError("Clé Azure Speech manquante.")
        
        config = speechsdk.SpeechConfig(subscription=key, region=self.region)
        config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
        )
        return config

    def generate_audio(self, text: str, voice: str = "fr-FR-HenriNeural") -> str:
        """
        Transforme le texte en fichier .wav.
        Retourne le CHEMIN COMPLET du fichier généré.
        """
        file_id = str(uuid.uuid4())
        output_path = self.storage_dir / f"{file_id}.wav"
        
        try:
            return self._synthesize(text, voice, str(output_path))
        except Exception as e:
            if self.fallback_key:
                logger.warning("Primary voice failed, trying fallback. Error: %s", str(e))
                return self._synthesize(text, voice, str(output_path), use_fallback=True)
            raise e

    def _synthesize(self, text: str, voice: str, output_path: str, use_fallback: bool = False) -> str:
        speech_config = self._get_config(use_fallback)
        speech_config.speech_synthesis_voice_name = voice
        audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)

        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, 
            audio_config=audio_config
        )

        logger.info("Synthesizing voice: %s (text preview: %s...)", voice, text[:30])
        result = synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            logger.info("Synthesis success. Path: %s", output_path)
            return output_path
        
        elif result.reason == speechsdk.ResultReason.Canceled:
            details = result.cancellation_details
            logger.warning("Synthesis failed. Reason: %s, Error: %s", details.reason, details.error_details)
            raise Exception(f"Azure Speech Error: {details.error_details}")

        return output_path

    def transcribe_audio(self, file_path: str, lang: str = "fr-FR") -> Tuple[str, float]:
        """
        Transforme l'audio du paysan en texte (STT).
        Retourne (texte_transcrit, score_confiance).
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Fichier audio introuvable: {file_path}")

        speech_config = self._get_config()
        speech_config.speech_recognition_language = lang
        audio_config = speechsdk.audio.AudioConfig(filename=file_path)
        
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, 
            audio_config=audio_config
        )

        result = recognizer.recognize_once_async().get()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text, getattr(result, "confidence", 0.0)
        
        logger.warning("Speech recognition failed. Reason: %s", result.reason)
        return "", 0.0
