import gradio as gr
import requests

API_URL = "http://127.0.0.1:5000/api/ask"  # ton backend Flask

def query_backend(user_query, user_id, zone_id):
    if not user_query or user_query.strip() == "":
        return "Veuillez entrer une question.", ""

    payload = {
        "query": user_query,
        "user_id": user_id or "anonymous",
        "zone_id": zone_id or "Mopti"
    }

    try:
        print('on commence a envoyer la question')
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            data = response.json()
            print('reponse:',data)
            return data.get("response", ""), "\n".join(data.get("trace", []))
        else:
            return f"Erreur API ({response.status_code})", ""
    except Exception as e:
        return f"Erreur de connexion au backend : {e}", ""


# Interface Gradio
with gr.Blocks(title="Assistant Agricole – Frontend") as demo:
    gr.Markdown("# 🌾 Assistant Agricole (Frontend Gradio)\nInterface qui appelle le backend Flask.")

    with gr.Row():
        user_id = gr.Textbox(label="User ID", value="test_user")
        zone_id = gr.Textbox(label="Zone ID", value="Mopti")

    query = gr.Textbox(label="Votre question", placeholder="Ex: Mon sol est sableux, que faire ?")

    submit_btn = gr.Button("Envoyer")

    response_box = gr.Markdown(label="Réponse")
    trace_box = gr.Markdown(label="Trace d'exécution")

    submit_btn.click(
        query_backend,
        inputs=[query, user_id, zone_id],
        outputs=[response_box, trace_box]
    )

if __name__ == "__main__":
    demo.launch(server_name="localhost", server_port=7861)