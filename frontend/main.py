import gradio as gr
import requests
import json

# URL of the FastAPI Backend
API_URL = "http://127.0.0.1:8000/api/v1/ask"

def query_backend(user_query, user_id, zone_id, mode):
    flow_type = "REPORT" if mode == "Daily Report" else "MESSAGE"
    user_query = user_query or "Generate Report"
    
    payload = {
        "user_id": user_id or "user_gradio",
        "zone_id": zone_id or "Centre",
        "query": user_query,
        "flow_type": flow_type
    }

    try:
        # 💡 CRUCIAL : timeout=None permet d'attendre indéfiniment (les 13 min)
        # sans que la librairie 'requests' ne coupe la connexion.
        response = requests.post(API_URL, json=payload, timeout=None)
        response.raise_for_status()
        
        data = response.json()
        trace = data.get("details", {}).get("execution_path", [])
        trace_str = " -> ".join(trace) if isinstance(trace, list) else str(trace)
        
        return data.get("response", "No Content"), trace_str
        
    except requests.exceptions.Timeout:
        return "⏳ Le traitement est très long, mais le serveur travaille toujours...", "Timeout"
    except requests.exceptions.ConnectionError:
        return "❌ Error: Backend is not running", "Connection Refused"
    except Exception as e:
        return f"❌ Error: {str(e)}", "Error Trace"

# Define Gradio Interface
with gr.Blocks(title="AgConnect - Assistant") as demo:
    gr.Markdown("# 🌾 AgConnect Assistant")
    
    with gr.Row():
        with gr.Column(scale=1):
            user_id = gr.Textbox(label="User ID", value="farmer_001")
            zone_id = gr.Dropdown(choices=["Centre", "Nord", "Sud"], value="Centre", label="Zone")
            mode = gr.Radio(choices=["Conversation", "Daily Report"], value="Conversation", label="Mode")
        
        with gr.Column(scale=2):
            # 💡 On remplace Markdown par Chatbot ou un Textbox protégé pour le statut
            output_msg = gr.Markdown(label="Response")
    
    with gr.Row():
         query = gr.Textbox(label="Your Question", lines=2)
         # 💡 Utilisation de 'status_tracker' automatique de Gradio
         submit_btn = gr.Button("🚀 Send Inquiry", variant="primary")
    
    with gr.Accordion("🛠️ Debug Trace", open=False):
        trace_output = gr.Textbox(label="Execution Path")

    # Event Handler avec indicateur de chargement
    submit_btn.click(
        fn=query_backend,
        inputs=[query, user_id, zone_id, mode],
        outputs=[output_msg, trace_output],
        show_progress="full" # 💡 Affiche une barre de chargement pendant les 13 min
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)