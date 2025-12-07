import logging
from typing import List, Dict
from rag.cross_encoder import CrossEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag.reranker")

class Reranker:
    """
    Reranker hybride en production.
    Combine : Vector store + CrossEncoder + Boost lexical métier.
    - Poids dynamiques par rôle
    - Normalisation des scores
    - Batching CrossEncoder
    - Fallback robuste
    """

    def __init__(self):
        self.encoder = CrossEncoder()

        # This part must be update with domain experts
        self.role_profiles = {
            "CROP": {"keywords": {"culture":0.2,"maïs":0.3,"semis":0.2,"sécheresse":0.3,"rendement":0.2},
                     "boost_factor": 0.15, "weights": (0.55, 0.35)},
            "SOIL": {"keywords": {"sol":0.3,"drainage":0.2,"irrigation":0.2,"compactage":0.2,"fertilité":0.3,"argileux":0.2},
                     "boost_factor": 0.20, "weights": (0.50, 0.40)},
            "METEO": {"keywords": {"débit":0.3,"barrage":0.2,"crue":0.3,"cote d'alerte":0.2,"pluviométrie":0.3},
                      "boost_factor": 0.15, "weights": (0.45, 0.45)},
            "MARKET": {"keywords": {"route":0.2,"pont":0.3,"accessibilité":0.2,"ouvrages d'art":0.2,"submergé":0.3,"transport":0.3},
                       "boost_factor": 0.25, "weights": (0.60, 0.30)},
            "HEALTH": {"keywords": {"maladie":0.3,"paludisme":0.3,"hygiène":0.2,"choléra":0.3,"eau potable":0.3,"épidémie":0.3},
                       "boost_factor": 0.20, "weights": (0.50, 0.40)},
            "Coordinateur": {"keywords": {"synthèse":0.2,"population":0.2,"urgence":0.3,"sinistré":0.3},
                             "boost_factor": 0.10, "weights": (0.60, 0.30)}
        }
        logger.info("⚖️ Reranker hybride initialisé.")

    def rerank(self, documents: List[Dict], agent_role: str, original_query: str) -> List[Dict]:
        """Applique la logique de tri avancée sur une liste de documents."""
        profile = self.role_profiles.get(agent_role)

        if not profile:
            logger.warning(f"⚠️ Unknown role: {agent_role}. Fallback to vector score only.")
            return sorted(documents, key=lambda x: x.get('score', 0.5), reverse=True)

        logger.info(f"⚖️ Reranking for role: {agent_role} (Boost: {profile['boost_factor']})")

        # --- Étape 1 : Batch scoring avec CrossEncoder ---
        pairs = [(original_query, doc.get('text_content', '')) for doc in documents]
        try:
            ce_scores = self.encoder.predict_batch(pairs)
        except Exception as e:
            logger.error(f"CrossEncoder inference failed: {e}. Fallback to vector-only rerank.")
            return sorted(documents, key=lambda x: x.get('score', 0.5), reverse=True)

        # --- Étape 2 : Normalisation des scores ---
        base_scores = [doc.get('score', 0.5) for doc in documents]
        min_b, max_b = min(base_scores), max(base_scores)
        min_c, max_c = min(ce_scores), max(ce_scores)

        norm_base = [(s - min_b) / (max_b - min_b + 1e-6) for s in base_scores]
        norm_ce   = [(s - min_c) / (max_c - min_c + 1e-6) for s in ce_scores]

        # --- Étape 3 : Calcul final avec boost lexical pondéré ---
        reranked_docs = []
        keywords = profile["keywords"]
        boost = profile["boost_factor"]
        w_base, w_ce = profile.get("weights", (0.60, 0.30))

        for doc, b, c, ce_raw in zip(documents, norm_base, norm_ce, ce_scores):
            processed_doc = doc.copy()
            content_text = processed_doc.get('text_content', '')
            content_lower = content_text.lower()

            # Score lexical pondéré
            matches_score = sum(weight for word, weight in keywords.items() if word.lower() in content_lower)
            multiplier = 1.0 + (matches_score * boost)

            final_score = (w_base * b + w_ce * c) * multiplier

            if 'metadata' not in processed_doc:
                processed_doc['metadata'] = {}
            processed_doc['metadata']['ce_score_raw'] = f"{ce_raw:.2f}"
            processed_doc['metadata']['ce_score_norm'] = f"{c:.2f}"
            processed_doc['metadata']['rerank_boost'] = f"+{int((multiplier-1)*100)}%"
            processed_doc['metadata']['weights'] = f"base={w_base}, ce={w_ce}"
            processed_doc['final_score'] = final_score
            reranked_docs.append(processed_doc)

        return sorted(reranked_docs, key=lambda x: x['final_score'], reverse=True)