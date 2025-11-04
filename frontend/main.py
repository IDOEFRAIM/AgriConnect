import os
import sys
import json
import logging
import uuid
import gradio as gr
from pathlib import Path
import requests

# Vu que main.py est dans frontend , on ajoute cette ligne pour permettre a main.py de se retrouver
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Les modules du project
from src.data_extraction import run_extraction
from src.data_processing import split_corpus_data
from src.data_vectordb import run_vectorstore

# Le setup du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)]
)
logger = logging.getLogger(__name__)



# Session management

SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(exist_ok=True)

# Ollama tunnel check
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://made-tanks-tissue-accuracy.trycloudflare.com")

def is_ollama_available(url: str) -> bool:
    try:
        response = requests.get(url, timeout=3)
        return response.status_code == 200
    except Exception:
        return False

def _save_session_history(session_id: str, chat_history):
    try:
        path = SESSIONS_DIR / f"{session_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save session {session_id}: {e}")

def _load_session_history(session_id: str):
    path = SESSIONS_DIR / f"{session_id}.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load session history {session_id}: {e}")
    return []




def chat_with_profile(user_query, chat_history):
    if not user_query or not str(user_query).strip():
        return chat_history or []

    try:
        payload = {"query": user_query, "history": chat_history or []}
        response = requests.post(os.getenv("API_URL","http://localhost:7860/chat"), json=payload)
        updated_history = response.json()
        return updated_history
    except Exception as e:
        logger.exception("Error in chat_with_profile")
        return (chat_history or []) + [{"role": "assistant", "content": f"Error: {e}"}]
    
def create_gradio_interface():
    with gr.Blocks(title="Agri Bot") as demo:
        gr.Markdown("# 🌾 Agri Bot")
        gr.Markdown("Pose une question a propos de l'agriculture et notre agent IA repo")

        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(height=500, type="messages")
            chat_input = gr.Textbox(label="Ask a question", placeholder="E.g., How can they increase production?")
            chat_btn = gr.Button("Send")

            chat_btn.click(fn=chat_with_profile, inputs=[chat_input, chatbot], outputs=[chatbot])
            chat_input.submit(fn=chat_with_profile, inputs=[chat_input, chatbot], outputs=[chatbot])

    return demo

if __name__ == "__main__":
    CORPUS_DIR = os.getenv("CORPUS_DIR","data/corpus.json")
    CHROMA_DB = os.getenv("CHROMA_DB_PATH","./chroma_db")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://made-tanks-tissue-accuracy.trycloudflare.com")

    # Si le fichier corpus n existe pas on lance run_extraction pour creer le fichier corpus.json
    if not os.path.exists(CORPUS_DIR):
        run_extraction()

    # Si la base vectorielle de chroma n exista pas on la creee
    if not os.path.exists(CHROMA_DB):
        run_vectorstore()

    # On cherche a savoir si nous avons pu nous connecter au tunel de cloudfared
    if is_ollama_available(OLLAMA_BASE_URL):
        logging.info(f"Ollama tunnel is reachable at {OLLAMA_BASE_URL}")
    else:
        logging.warning(f"Ollama tunnel unreachable at {OLLAMA_BASE_URL}. You may need to restart it.")

    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=8000, share=True)