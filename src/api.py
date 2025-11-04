from flask import Flask, request, jsonify
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap, RunnableLambda
import os

app = Flask(__name__)

# Config
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "llama3")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "gemma:2b")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

""" Prompt template: c'est comme un moule,il nous permet de modeler la forme de notre prompt. 
Nous pouvons ainsi le reutiliser"""

prompt = ChatPromptTemplate.from_messages([
    ("system", "Tu es un assistant IA spécialisé en agriculture. Réponds en utilisant les documents fournis. S'il n'y a aucun document, réponds avec tes propres connaissances. Ne fais pas de suppositions inutiles. Sois concis."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query_str}")
])

model_kwargs = {
    "temperature": 0.2,         # Low randomness for reliable, fact-based answers
    "max_new_tokens": 300,      # Enough for concise but informative responses
    "min_new_tokens": 1,        # Allows short replies when appropriate
    "decoding_method": "sample",# Balanced generation method
    "top_k": 40,                # Filters to top 40 candidates for diversity
    "top_p": 0.9                # Nucleus sampling for controlled creativity
}



@app.route("/generate", methods=["POST"])
def generate_response():

    if not OLLAMA_BASE_URL or not EMBEDDING_MODEL_ID or not CHROMA_DB_PATH:
        raise ValueError("Certaines variables d'environnement sont manquantes.")

    data = request.get_json()
    query = data.get("query")
    chat_history = data.get("history", [])

    if not query or not str(query).strip():
        return jsonify({"history": chat_history})
    
    # Embeddings + vectorstore
    embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL_ID,
                                        base_url=OLLAMA_BASE_URL
                                        )
    
    vectorstore = Chroma(persist_directory=CHROMA_DB_PATH,
                          embedding_function=embedding_model)
    
    """Le retriever est un moyen de trouver des documents ou textes qui ont de fortes 
    avec la question de l'utilisateur. Il existe plusieurs moyen d'avoir un retriever. Nous decidons
    ic d'utiliser notre base de donnee vectorielle pour avoir notre retriever pour une question de simplicite
   
     Notre retriever nous renvoit juste les 5 documents qui ont le plus de probabilite d'avoir 
      une similiarity avec la question de l'utilisateur """
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    """ Nous cherchons les documents qui ont de fortes similarite avec la questions de l'utilisateur
    pour enrichir le prompt. Les llm ont ete entraines sur des une immense categories de donnee qui sont souvent obsolete ou depasse.Par exemple nchatgpt a ete sur des donnees datant de 2021 environ,sion lui pose des
      questions de 2025, il risque d'etre perdu. Nous lui fournissons donc des informations supplementaires graces aux documents retrieve par notre retriever
     """ 
    documents = retriever.invoke(query)

    # Format documents
    context_str = "\n\n".join(doc.page_content for doc in documents if doc.page_content.strip())

    # Nous implementons notre llm en lui changeant certains parametres tels que la temperature
    llm = ChatOllama(
        model=LLM_MODEL_ID,
        model_kwargs=model_kwargs,
        base_url=OLLAMA_BASE_URL  # Le lien vers notre tunel cloudfared
    )

    # Chain: Enchainement d'execution de components
    chain = (
        RunnableMap({
            "context": lambda x: x["docs"],
            "query_str": lambda x: x["question"],
        })
        | RunnableLambda(lambda x: {
            "context_str": context_str,
            "query_str": x["query_str"],
            "chat_history": chat_history
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    result = chain.invoke({"question": query, "docs": documents})
    reply = result.content if hasattr(result, "content") else str(result)

    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": reply})

    return jsonify({ "history": chat_history})