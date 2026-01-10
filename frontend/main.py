import gradio as gr
import requests
import json

# URL of the FastAPI Backend
API_URL = "http://127.0.0.1:8000/api/v1/ask"

def query_backend(user_query, user_id, zone_id, mode):
    
    # Determine Flow Type
    flow_type = "MESSAGE"
    if mode == "Daily Report":
        flow_type = "REPORT"
        # For reports, query can be empty
        user_query = user_query or "Generate Report"
    
    payload = {
        "user_id": user_id or "user_gradio",
        "zone_id": zone_id or "Centre",
        "query": user_query,
        "flow_type": flow_type
    }

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        
        data = response.json()
        
        # Format Trace
        # Extract execution path from the 'details' nested object
        trace = data.get("details", {}).get("execution_path", [])
        trace_str = " -> ".join(trace) if isinstance(trace, list) else str(trace)
        
        return data.get("response", "No Content"), trace_str
        
    except requests.exceptions.ConnectionError:
        return "❌ Error: Backend is not running at http://127.0.0.1:8000", "Connection Refused"
    except Exception as e:
        return f"❌ Error: {str(e)}", "Error Trace"

# Define Gradio Interface
with gr.Blocks(title="AgConnect - Agricultural Assistant") as demo:
    gr.Markdown(
        """
        # 🌾 AgConnect Assistant
        **Connected to FastAPI Backend**
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            user_id = gr.Textbox(label="User ID", value="farmer_001")
            zone_id = gr.Dropdown(choices=["Centre", "Nord", "Sud", "Ouest", "Est"], value="Centre", label="Zone")
            mode = gr.Radio(choices=["Conversation", "Daily Report"], value="Conversation", label="Mode")
        
        with gr.Column(scale=2):
            chatbot = gr.Markdown(label="Response")
    
    with gr.Row():
         query = gr.Textbox(label="Your Question", placeholder="Ex: Is it going to rain today? or What is the price of Maize?", lines=2)
         submit_btn = gr.Button("🚀 Send Inquiry", variant="primary")
    
    with gr.Accordion("🛠️ Debug Trace", open=False):
        trace_output = gr.Textbox(label="Execution Path")

    # Event Handlers
    submit_btn.click(
        fn=query_backend,
        inputs=[query, user_id, zone_id, mode],
        outputs=[chatbot, trace_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
