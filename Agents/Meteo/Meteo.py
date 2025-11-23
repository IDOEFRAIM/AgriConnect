import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pydantic import BaseModel
from typing import List, Optional
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from utils.retriever import load_chroma_collection, load_embedding_model, retrieve_similar_documents


class DocumentMatch(BaseModel):
    source_file: str
    distance: float
    excerpt: str


class MeteoAnswer(BaseModel):
    query: str
    documents: List[DocumentMatch]
    summary: str


class MeteoState(BaseModel):
    query: str
    documents: Optional[List[DocumentMatch]] = None
    response: Optional[MeteoAnswer] = None


class MeteoAgent:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2"):
        self.collection = load_chroma_collection()
        self.embed_model = load_embedding_model(embedding_model_name)
        self.graph = self.build_graph()

    def validate_query(self, state: MeteoState) -> MeteoState:
        if not state.query.strip():
            raise ValueError("Requête vide.")
        return state

    def retrieve_documents(self, state: MeteoState) -> MeteoState:
        raw_results = retrieve_similar_documents(
            state.query, self.collection, self.embed_model, n_results=5
        )
        docs = [DocumentMatch(**doc) for doc in raw_results]
        return state.copy(update={"documents": docs})

    def generate_response(self, state: MeteoState) -> MeteoState:
        docs = state.documents or []
        query = state.query

        if not docs:
            summary = f"Aucun document pertinent trouvé pour répondre à la question : « {query} »."
        else:
            # Réponse directe simple
            if "inondation" in query.lower():
                reponse_directe = "Les documents suggèrent un risque d'inondation dans certaines régions."
            else:
                reponse_directe = "Les documents fournissent des informations météorologiques pertinentes."

            # Justification extraite
            justifications = []
            for doc in docs:
                lines = doc.excerpt.strip().split("\n")
                for line in lines:
                    if any(kw in line.lower() for kw in ["pluie", "température", "vent", "inondation", "sécheresse"]):
                        justifications.append(f"- {line.strip()}")
                        break

            justification_text = "\n".join(justifications[:5]) or "Aucune justification explicite extraite."

            summary = (
                f"1. Réponse directe : {reponse_directe}\n\n"
                f"2. Justification basée sur les extraits :\n{justification_text}"
            )

        response = MeteoAnswer(query=query, documents=docs, summary=summary)
        return state.copy(update={"response": response})

    def log_state(self, state: MeteoState) -> MeteoState:
        print("Requête :", state.query)
        print("Documents récupérés :", len(state.documents or []))
        print(" Résumé :", state.response.summary if state.response else "Aucun")
        return state

    def build_graph(self):
        graph = StateGraph(MeteoState)
        graph.add_node("validate", self.validate_query)
        graph.add_node("retrieving", self.retrieve_documents)
        graph.add_node("generating", self.generate_response)
        graph.add_node("logging", self.log_state)

        graph.set_entry_point("validate")
        graph.add_edge("validate", "retrieving")
        graph.add_edge("retrieving", "generating")
        graph.add_edge("generating", "logging")
        graph.add_edge("logging", END)

        return graph.compile()

    def run(self, query: str) -> dict:
        initial_state = MeteoState(query=query)
        final_state_dict = self.graph.invoke(initial_state.model_dump(), config=RunnableConfig(run_name="agent_meteo"))
        final_state = MeteoState(**final_state_dict)
        return final_state.response.model_dump()


if __name__ == "__main__":
    agent = MeteoAgent()
    result = agent.run("Quels sont les risques d'inondation à Ouagadougou en octobre ?")
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))
