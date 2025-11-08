from typing import TypedDict,Optional,Tuple
from langgraph.graph import StateGraph
from langgraph.graph import END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
import time

""""
In this project, I use mistral mistral:7b-instruct-q4_K_M and langraph to build my graph
"""
""""
I set the state of question.This state will be passed to our graph through the process
    1) question:A string which represent the user query. We will use it to get the context and passed it to our model(llm)
    2)context: It represent the context of our question. That way , our model can provide some more accurate responses.this field is opional
    3)answer:A string which represents the output of our llm,you must reply in french even we speak in english
"""
class QaState(TypedDict):
    question:str
    context:Optional[Tuple[str,...]]
    answer:Optional[str]

""""
For instance , we can create a Question dict:
QaStateOne:QaState = {
    "question":"What is Agriconnect",
    "context":"Agriconnect level up AI application in agriculture. Allow farmers to get suitable informations",
    "answer":None

You can try to print attributes
for key, value in QaStateOne.items():
    print(f"{key}: {value}")
}
"""

# This function allows us to create a question state
def qaState(questionText):
    return QaState(question=questionText,context="",answer=None)


# We use this function to validate user input
def inputValidationNode(state):
    question = state.get("question","")

    if not question:
        return {"error":"question is not provided"}
        
    return state

# I actually use a simple function to get the context . I plan to use an llm to get the context of our query
def contextNode(state):
    question = state.get("question", "").lower()

    if "agriconnect" in question or "agri" in question:
        # For the context we use a tuple, cause context can be retrieve from multiple documents
        context = (
            "Agriconnect level up AI application in agriculture. Allow farmers to get suitable informations",
           "This guided project is about using LangGraph, a Python library to design state-based workflows for agriculture ",
           "LangGraph simplifies building complex applications by connecting modular nodes with conditional edges.",
        )
        state["context"] = context
        return state

    return state


"""
We define our model using mistral:7b-instruct-q4_K_M and qaNode which enables us to get the answer
Why did I use Mistal? First of all, it is open source. Its performances are quite good
"""

llm = ChatOllama(model="mistral:7b-instruct-q4_K_M")

def qaNode(state):
    question = state.get("question", "")
    context = state.get("context", "")
    firstInstruction = "Restate the user's question in your own words to ensure clarity and shared understanding."

    #  For generating , we ask gemini to give us some prompt template
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a helpful and knowledgeable assistant specialized in agriculture. Your role is to clearly and accurately answer user questions related to farming, crops, livestock, soil, climate, agricultural technologies, and sustainable practices.

            Your response should be natural, fluid, and conversational — not structured as numbered steps. Seamlessly integrate the following elements into your reply:
            - Clarify the user's question in your own words to ensure shared understanding.
            - Provide scientific and practical reasoning using agricultural knowledge, examples, and context when available.
            - If relevant, address common misconceptions in agriculture with evidence-based insights.
            - Reflect critically on your answer, considering ecological sustainability, economic viability, and scientific consistency.
            - At the end, list 1–3 search queries separately to help the user explore agricultural techniques, climate impact, or farming innovations. Do not include them inside the reflection.

            Always follow these principles:
            - Use the provided context if available. If not, rely on your own agricultural expertise.
            - Ensure the explanation is understandable and practical, even for non-experts.
            - Always reply in the same language the user used in their question.
            - If the question is outside the scope of agriculture, politely redirect the user to relevant agricultural topics.

            Focus on: sustainable farming, soil health, crop-livestock integration, climate adaptation, and practical advice for farmers.
            Your tone should be professional, supportive, and informative. Avoid speculation outside the agricultural domain.
            """
        ),
            MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Answer the user's question above using the required format, emphasizing clarity, agricultural relevance, and practical insight."
        )
    
    ])

    # I inject variables in the prompt template

    formatted_messages = prompt.format_messages(
        first_instruction=firstInstruction,
        messages=[
            HumanMessage(content=f"Context: {context}\nQuestion: {question}")
        ]
    )

    try:
        response = llm.invoke(formatted_messages)
        state["answer"] = response.content
        return state
    except Exception as e:
        state["answer"] = f" Error: {e}"
        return state
    
# Lets build our graph
def setGraph():
    try:
        qaWorkflow = StateGraph(QaState)
        # Adding nodes
        qaWorkflow.add_node("inputNode", inputValidationNode)
        qaWorkflow.add_node("contextNode", contextNode)
        qaWorkflow.add_node("qANode", qaNode)
        qaWorkflow.set_entry_point("inputNode")

        qaWorkflow.add_edge("inputNode", "contextNode")
        qaWorkflow.add_edge("contextNode", "qANode")
        qaWorkflow.add_edge("qANode", END)

        qaApp = qaWorkflow.compile()

        return qaApp
    
    except Exception as e:
        print(f'An error happen:{e}')
        return
    
# It will allow to format time in this format: minute-seconde
def formatDuration(seconds):
    minutes = int(seconds // 60)
    sec = seconds % 60
    return f"{minutes} min {sec:.2f} sec"

def qaReply(state):
    qaApp = setGraph()
    try:
        start = time.time()
        finalState = qaApp.invoke(state)
        end = time.time()

        timeToReply = formatDuration(end - start)
        return finalState ,timeToReply 
    except Exception as e:
        state["answer"] = f"Error: {e}"
        return state

