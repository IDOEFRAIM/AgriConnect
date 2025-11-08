import gradio as gr
import requests

API_URL = "http://localhost:5000/reply"


def reply(userQuery,chatHistory):
    # if there are no question, we return chat history
    if not userQuery or not str(userQuery).strip():
        return chatHistory or []
    
    try:
        payload = {"query":userQuery,"chatHistory":chatHistory or []}
        response = requests.post(API_URL,json=payload)
        print(response)
        updatedHistory = response.json()

        print(updatedHistory)
        cleanHistory = []
        for msg in updatedHistory.get("chatHistory", []):
            if isinstance(msg, dict) and isinstance(msg.get("content"), str) and msg["content"].strip():
                cleanHistory.append(msg)

        return cleanHistory, ""

    
    except Exception as e:
        return chatHistory + [{"role": "assistant", "content": f"❌ Erreur de connexion à l'API : {str(e)}"}],""

def createGradioInterface():
    with gr.Blocks(title='Agriconnect') as demo:
        with gr.Tab('chat'):
            chatbot = gr.Chatbot(height=500,type="messages")
            chatInput = gr.Textbox(label="Ask a question",placeholder="Have you a question about agriculture ,write it")
            chatBtn = gr.Button("Send")

            chatBtn.click(fn=reply, inputs=[chatInput, chatbot], outputs=[chatbot,chatInput])
            chatInput.submit(fn=reply, inputs=[chatInput, chatbot], outputs=[chatbot,chatInput])
    return demo

if __name__ == "__main__":
    demo = createGradioInterface()
    demo.launch(server_name="localhost",server_port=8002,share=True)