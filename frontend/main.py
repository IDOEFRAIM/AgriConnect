import gradio as gr
import requests

# Function triggered by chatbot
def chatbot(message, history):
    url = "http://localhost:5000/gen"   # adjust port if needed
    try:
        response = requests.post(url, json={"msg": message})
        if response.status_code == 200:
            data = response.json()
            
            reply = data['data'].get('final_response','')
            print(reply)
        else:
            reply = f"Error: {response.status_code}"
    except Exception as e:
        reply = f"Request failed: {e}"

    # Append user + assistant messages to history
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply}
    ]
    return history

# Build Gradio chatbot interface
with gr.Blocks() as demo:
    gr.Markdown("### Chatbot with API Request")
    chatbot_ui = gr.Chatbot()
    msg = gr.Textbox(label="Type a message")
    send_btn = gr.Button("Send")

    send_btn.click(fn=chatbot, inputs=[msg, chatbot_ui], outputs=chatbot_ui)
    msg.submit(fn=chatbot, inputs=[msg, chatbot_ui], outputs=chatbot_ui)

if __name__ == "__main__":
    demo.launch(share=True)