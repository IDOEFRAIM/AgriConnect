"""
Interface Gradio pour tester l'interface vocale Agri-OS en local
Permet de tester Azure Speech (STT/TTS) sans configurer WhatsApp
"""

import os
import sys
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Éviter le conflit avec mcp
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr

# Charger les variables d'environnement
load_dotenv()

# Import conditionnel des services
try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    print("⚠️  Azure Speech SDK non installé. Installez avec: pip install azure-cognitiveservices-speech")

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    print("⚠️  gTTS non installé. Installez avec: pip install gtts")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Configuration Azure Speech
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_REGION = os.getenv("AZURE_REGION", "westeurope")
USE_AZURE_SPEECH = os.getenv("USE_AZURE_SPEECH", "false").lower() == "true"

# URL du backend
API_URL = "http://127.0.0.1:8000/api/v1/ask"


def azure_stt(audio_file_path: str) -> str:
    """
    Transcrit un fichier audio avec Azure Speech STT
    """
    if not AZURE_AVAILABLE:
        return "❌ Azure Speech SDK non installé"
    
    if not AZURE_SPEECH_KEY or not USE_AZURE_SPEECH:
        return "❌ Azure Speech non configuré (vérifiez .env)"
    
    try:
        speech_config = speechsdk.SpeechConfig(
            subscription=AZURE_SPEECH_KEY,
            region=AZURE_REGION
        )
        speech_config.speech_recognition_language = "fr-FR"
        
        audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config
        )
        
        result = speech_recognizer.recognize_once()
        
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text
        elif result.reason == speechsdk.ResultReason.NoMatch:
            return "❌ Aucune parole détectée dans l'audio"
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            return f"❌ Erreur STT: {cancellation.reason} - {cancellation.error_details}"
        
        return "❌ Erreur de transcription"
        
    except Exception as e:
        return f"❌ Erreur Azure STT: {str(e)}"


def azure_tts(text: str, language: str = "fr") -> str:
    """
    Génère un fichier audio à partir de texte avec Azure Neural TTS
    """
    if not AZURE_AVAILABLE:
        return gtts_tts(text, language)
    
    if not AZURE_SPEECH_KEY or not USE_AZURE_SPEECH:
        return gtts_tts(text, language)
    
    try:
        speech_config = speechsdk.SpeechConfig(
            subscription=AZURE_SPEECH_KEY,
            region=AZURE_REGION
        )
        
        # Voix neurale française (haute qualité)
        voice_map = {
            "fr": "fr-FR-DeniseNeural",  # Voix féminine
            "en": "en-US-JennyNeural"
        }
        speech_config.speech_synthesis_voice_name = voice_map.get(language, "fr-FR-DeniseNeural")
        
        # Créer fichier temporaire
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        audio_config = speechsdk.audio.AudioOutputConfig(filename=temp_audio.name)
        
        speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=audio_config
        )
        
        result = speech_synthesizer.speak_text_async(text).get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return temp_audio.name
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            print(f"❌ Azure TTS annulé: {cancellation.reason} - {cancellation.error_details}")
            return gtts_tts(text, language)
        
        return gtts_tts(text, language)
        
    except Exception as e:
        print(f"❌ Erreur Azure TTS: {e}, fallback vers gTTS")
        return gtts_tts(text, language)


def gtts_tts(text: str, language: str = "fr") -> str:
    """
    Génère un fichier audio avec gTTS (fallback gratuit)
    """
    if not GTTS_AVAILABLE:
        return None
    
    try:
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save(temp_audio.name)
        return temp_audio.name
    except Exception as e:
        print(f"❌ Erreur gTTS: {e}")
        return None


def query_backend_text(query: str, user_id: str, zone_id: str, crop: str) -> str:
    """
    Envoie une requête texte au backend
    """
    if not REQUESTS_AVAILABLE:
        return "❌ Requests non installé"
    
    payload = {
        "user_id": user_id or "user_gradio_voice",
        "zone_id": zone_id or "boromo",
        "crop": crop or "Maïs",
        "query": query,
        "flow_type": "MESSAGE"
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "Aucune réponse")
    except Exception as e:
        return f"❌ Erreur backend: {str(e)}"


def process_voice_message(audio, user_id, zone_id, crop, use_tts):
    """
    Traite un message vocal complet : STT → Backend → TTS
    """
    if audio is None:
        return "❌ Aucun audio fourni", None, ""
    
    # Étape 1 : STT (Speech-to-Text)
    transcription = "🎤 Transcription en cours..."
    yield transcription, None, "🔄 Étape 1/3 : Transcription audio..."
    
    audio_path = audio
    if isinstance(audio, tuple):
        audio_path = audio[1]  # Gradio retourne (sample_rate, audio_data)
    
    transcribed_text = azure_stt(audio_path)
    
    if transcribed_text.startswith("❌"):
        yield transcribed_text, None, "❌ Échec de la transcription"
        return
    
    yield f"📝 Transcription: {transcribed_text}", None, "🔄 Étape 2/3 : Traitement par l'IA..."
    
    # Étape 2 : Requête au backend
    response_text = query_backend_text(transcribed_text, user_id, zone_id, crop)
    
    if response_text.startswith("❌"):
        yield f"📝 Transcription: {transcribed_text}\n\n{response_text}", None, "❌ Échec de la requête"
        return
    
    # Afficher réponse texte
    full_text = f"📝 Transcription: {transcribed_text}\n\n🤖 Réponse: {response_text}"
    
    if not use_tts:
        yield full_text, None, "✅ Terminé (mode texte uniquement)"
        return
    
    yield full_text, None, "🔄 Étape 3/3 : Génération audio..."
    
    # Étape 3 : TTS (Text-to-Speech)
    audio_response = azure_tts(response_text, language="fr")
    
    if audio_response:
        yield full_text, audio_response, "✅ Terminé ! Écoutez la réponse audio ci-dessous 🔊"
    else:
        yield full_text, None, "⚠️  Réponse texte uniquement (TTS non disponible)"


def test_azure_config():
    """
    Teste la configuration Azure Speech
    """
    status = "🔧 Configuration Azure Speech:\n\n"
    
    if AZURE_SPEECH_KEY:
        status += f"✅ AZURE_SPEECH_KEY: {AZURE_SPEECH_KEY[:20]}...\n"
    else:
        status += "❌ AZURE_SPEECH_KEY: Non configurée\n"
    
    status += f"✅ AZURE_REGION: {AZURE_REGION}\n"
    status += f"✅ USE_AZURE_SPEECH: {USE_AZURE_SPEECH}\n\n"
    
    if AZURE_AVAILABLE:
        status += "✅ Azure Speech SDK installé\n"
    else:
        status += "❌ Azure Speech SDK non installé\n"
    
    if GTTS_AVAILABLE:
        status += "✅ gTTS disponible (fallback)\n"
    else:
        status += "⚠️  gTTS non installé\n"
    
    # Test rapide de synthèse vocale
    if AZURE_AVAILABLE and AZURE_SPEECH_KEY and USE_AZURE_SPEECH:
        status += "\n🧪 Test Azure TTS...\n"
        test_audio = azure_tts("Bonjour, je suis Agri-OS, votre assistant agricole.")
        if test_audio:
            status += "✅ Synthèse vocale Azure fonctionnelle !\n"
            return status, test_audio
        else:
            status += "❌ Échec du test TTS\n"
    
    return status, None


# Interface Gradio
with gr.Blocks(title="Agri-OS Voice Test", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎤 Agri-OS - Test Interface Vocale (Local)
    
    Testez l'interface vocale complète avec Azure Speech Services avant le déploiement WhatsApp.
    
    **Pipeline complet :**  
    🎤 Audio → 📝 STT (Azure) → 🤖 Backend AI → 🔊 TTS (Azure Neural) → Audio
    """)
    
    with gr.Tabs():
        # TAB 1 : Test vocal complet
        with gr.Tab("🎙️ Test Vocal Complet"):
            gr.Markdown("### Enregistrez votre question vocale")
            
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="🎤 Enregistrez votre question"
                    )
                    
                    user_id_voice = gr.Textbox(
                        label="User ID",
                        value="farmer_voice_001"
                    )
                    zone_id_voice = gr.Dropdown(
                        choices=["Boucle du Mouhoun - Dedougou", "Hauts-Bassins - Bobo", "Sahel - Dori"],
                        value="Hauts-Bassins - Bobo",
                        label="Zone Agro-écologique"
                    )
                    crop_voice = gr.Dropdown(
                        choices=["Maïs", "Coton", "Sésame", "Sorgho", "Riz"],
                        value="Maïs",
                        label="Culture"
                    )
                    
                    use_tts = gr.Checkbox(
                        label="Activer réponse audio (TTS)",
                        value=True
                    )
                    
                    submit_voice_btn = gr.Button("🚀 Traiter le message vocal", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    status_output = gr.Textbox(
                        label="📊 Statut du traitement",
                        lines=2
                    )
                    text_output = gr.Markdown(label="📝 Transcription & Réponse")
                    audio_output = gr.Audio(
                        label="🔊 Réponse Audio",
                        type="filepath"
                    )
            
            submit_voice_btn.click(
                fn=process_voice_message,
                inputs=[audio_input, user_id_voice, zone_id_voice, crop_voice, use_tts],
                outputs=[text_output, audio_output, status_output]
            )
        
        # TAB 2 : Configuration
        with gr.Tab("⚙️ Configuration Azure"):
            gr.Markdown("### Vérification de la configuration Azure Speech Services")
            
            test_btn = gr.Button("🧪 Tester la configuration", variant="secondary")
            config_status = gr.Textbox(
                label="Statut de la configuration",
                lines=15
            )
            test_audio_output = gr.Audio(
                label="🔊 Audio de test",
                type="filepath"
            )
            
            test_btn.click(
                fn=test_azure_config,
                inputs=[],
                outputs=[config_status, test_audio_output]
            )
        
        # TAB 3 : Mode texte (original)
        with gr.Tab("💬 Mode Texte (Original)"):
            gr.Markdown("### Interface texte classique")
            
            with gr.Row():
                with gr.Column(scale=1):
                    user_id_text = gr.Textbox(label="User ID", value="farmer_001")
                    zone_id_text = gr.Dropdown(
                        choices=["Centre", "Nord", "Sud", "Bobo-Dioulasso"],
                        value="Bobo-Dioulasso",
                        label="Zone"
                    )
                    crop_text = gr.Dropdown(
                        choices=["Maïs", "Coton", "Sésame", "Sorgho", "Riz"],
                        value="Maïs",
                        label="Culture"
                    )
                
                with gr.Column(scale=2):
                    query_text = gr.Textbox(
                        label="Votre question",
                        lines=3,
                        placeholder="Ex: Quand planter le maïs à Bobo ?"
                    )
                    submit_text_btn = gr.Button("🚀 Envoyer", variant="primary")
                    response_text = gr.Markdown(label="Réponse")
            
            submit_text_btn.click(
                fn=query_backend_text,
                inputs=[query_text, user_id_text, zone_id_text, crop_text],
                outputs=[response_text]
            )
    
    gr.Markdown("""
    ---
    ### 📝 Instructions
    
    1. **Test Vocal** : Cliquez sur le micro, posez votre question en français, puis "Traiter"
    2. **Upload Audio** : Ou uploadez un fichier audio existant (.wav, .mp3)
    3. **Vérifier Config** : Allez dans l'onglet "Configuration" pour tester Azure Speech
    
    **Exemples de questions vocales :**
    - "Quand dois-je planter le maïs à Bobo-Dioulasso ?"
    - "Quels sont les risques de maladies pour le coton ?"
    - "Quel est le prix actuel du sésame ?"
    
    💡 **Note** : Le backend doit être lancé sur `http://127.0.0.1:8000`
    """)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Lancement Agri-OS Voice Test Interface")
    print("="*60)
    print(f"\n✅ Azure Speech: {'Activé' if USE_AZURE_SPEECH else 'Désactivé'}")
    print(f"✅ Région: {AZURE_REGION}")
    print(f"✅ Backend API: {API_URL}")
    print("\n📝 Ouvrez http://127.0.0.1:7860 dans votre navigateur\n")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False  # Mettre True pour partager publiquement
    )
