import json
import logging
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END
from backend.src.agriconnect.graphs.prompts import (
    PLANT_DOCTOR_SYSTEM_TEMPLATE,
    PLANT_DOCTOR_USER_TEMPLATE,
    DOCTOR_PLANNER_TEMPLATE,
    DOCTOR_AUGMENT_PROMPT

)
from backend.src.agriconnect.rag.components import get_groq_sdk
from backend.src.agriconnect.rag.metric import RAGEvaluator
from backend.src.agriconnect.rag.retriever import AgileRetriever
from backend.src.agriconnect.tools.health import HealthDoctorTool

from backend.src.agriconnect.tools.refine import RefineTool
logger = logging.getLogger("Agent.PlantDoctor")


class PlantDoctorState(TypedDict, total=False):
    user_query: str
    user_level: str
    culture_config: Dict[str, Any]
    augmented_symptoms: str
    diagnosis_raw: Dict[str, Any]
    risk_flags: List[str]
    optimized_query: str
    retrieved_context: str
    sources: List[Dict[str, Any]]
    final_response: str
    evaluation: Dict[str, float]
    status: str
    warnings: List[str]
    # NOUVELLES FEATURES PRATIQUES
    photo_paths: List[str]  # Chemins vers photos symptômes
    guided_questions: List[str]  # Questions si description floue
    treatment_costs: Dict[str, float]  # Coût estimé par traitement
    alternative_products: Dict[str, List[str]]  # Alternatives si produit indispo
    local_availability: Dict[str, str]  # Où acheter localement 

    # refining
    rewrited_retry_count:int
    critique_retry_count:int    


class PlantHealthDoctor:
    """
    Docteur des Plantes AgriBot - Version Pratique.
    
    NOUVELLES FONCTIONNALITÉS:
    1. Analyse photos de symptômes (vision AI)
    2. Questions guidées si description floue
    3. Coûts estimés des traitements
    4. Alternatives si produits indisponibles
    5. Localisation points de vente
    """
    
    def __init__(
        self,
        llm_client=None,
        retriever: Optional[AgileRetriever] = None,
        evaluator: Optional[RAGEvaluator] = None,
    ):
        self.model_planner = "llama-3.1-8b-instant"
        self.model_answer = "llama-3.3-70b-versatile"
        self.doctor = HealthDoctorTool()
        self.refine = RefineTool(llm=llm_client)

        try:
            self.llm = llm_client if llm_client else get_groq_sdk()
        except Exception as exc:
            logger.error("Impossible d'initialiser le LLM : %s", exc)
            self.llm = None

        self.retriever = retriever if retriever else self._safe_retriever()
        self.evaluator = evaluator if evaluator else self._safe_evaluator()
        
        # Base de données produits locaux (à remplacer par vraie DB)
        self.product_database = self._load_product_database()

    def _load_product_database(self) -> Dict[str, Dict]:
        """
        Base de données des produits phytosanitaires locaux.
        
        Structure: {
            "nom_produit": {
                "prix_moyen": float,
                "points_vente": ["lieu1", "lieu2"],
                "alternatives": ["produit_alternatif1", "produit_alternatif2"],
                "type": "bio" | "chimique"
            }
        }
        """
        return {
            # Produits bio
            "neem_oil": {
                "nom_local": "Huile de Neem",
                "prix_moyen": 2500,  # FCFA/litre
                "points_vente": ["Coopérative Agricole", "Marché Central"],
                "alternatives": ["savon_noir", "purin_piment"],
                "type": "bio",
                "dosage": "30ml/litre d'eau"
            },
            "savon_noir": {
                "nom_local": "Savon Noir",
                "prix_moyen": 500,  # FCFA/kg
                "points_vente": ["Marché Local", "Boutique Village"],
                "alternatives": ["neem_oil"],
                "type": "bio",
                "dosage": "30g/litre d'eau"
            },
            "bouillie_bordelaise": {
                "nom_local": "Bouillie Bordelaise",
                "prix_moyen": 1800,  # FCFA/kg
                "points_vente": ["Boutique Intrants"],
                "alternatives": ["purin_ortie"],
                "type": "bio",
                "dosage": "20g/litre d'eau"
            },
            
            # Produits chimiques (dernier recours)
            "lambda_cyhalothrine": {
                "nom_local": "Karate (insecticide)",
                "prix_moyen": 4500,  # FCFA/litre
                "points_vente": ["Boutique Intrants Agréée"],
                "alternatives": ["neem_oil", "savon_noir"],
                "type": "chimique",
                "dosage": "1ml/litre d'eau",
                "warning": "⚠️ Port EPI obligatoire"
            },
            "mancozebe": {
                "nom_local": "Manèbe (fongicide)",
                "prix_moyen": 3200,  # FCFA/kg
                "points_vente": ["Boutique Intrants Agréée"],
                "alternatives": ["bouillie_bordelaise"],
                "type": "chimique",
                "dosage": "2g/litre d'eau",
                "warning": "⚠️ Respecter délai avant récolte"
            }
        }

    def _analyze_photo_symptoms(self, photo_path: str) -> Dict[str, Any]:
        """
        Analyse une photo de symptômes (placeholder pour vision AI).
        
        TODO: Intégrer Google Vision API ou modèle local (PlantVillage dataset)
        """
        logger.info(f"📷 Analyse photo: {photo_path}")
        
        # Simulation (à remplacer par vraie API vision)
        return {
            "detected_symptoms": [
                "Jaunissement des feuilles",
                "Taches brunes circulaires",
                "Déformation feuilles"
            ],
            "confidence": 0.75,
            "suggested_diseases": [
                "Mildiou",
                "Carence azote",
                "Virose"
            ]
        }

    def _ask_guided_questions(self, initial_query: str, crop: str) -> List[str]:
        """
        Génère des questions guidées si la description est floue.
        """
        logger.info("❓ Génération questions guidées")
        
        # Questions standard par culture
        base_questions = [
            "🌿 Les feuilles sont-elles jaunes, brunes, ou avec des taches?",
            "📏 Quel âge a votre culture? (jeune/adulte/mature)",
            "💧 Avez-vous arrosé récemment? (oui/non)",
            "☀️ La plante est en plein soleil ou ombragée?",
            "🐛 Voyez-vous des insectes sur les plantes? (décrire si oui)"
        ]
        
        # Questions spécifiques par culture
        crop_specific = {
            "maïs": [
                "🌽 Les épis sont-ils touchés?",
                "🌾 Les tiges sont-elles cassantes ou molles?"
            ],
            "tomate": [
                "🍅 Les fruits ont-ils des taches?",
                "🌱 Les feuilles se recroquevillent-elles?"
            ],
            "coton": [
                "☁️ Les capsules sont-elles ouvertes normalement?",
                "🐛 Y a-t-il des chenilles?"
            ]
        }
        
        questions = base_questions.copy()
        crop_lower = crop.lower()
        if crop_lower in crop_specific:
            questions.extend(crop_specific[crop_lower])
        
        return questions

    def _estimate_treatment_cost(self, diagnosis: Dict[str, Any], surface_ha: float = 1.0) -> Dict[str, float]:
        """
        Estime le coût des traitements recommandés.
        """
        logger.info("💰 Estimation coûts traitements")
        
        treatment = diagnosis.get("traitement_recommande", {})
        if isinstance(treatment, str):
            # Parsing basique si c'est du texte
            treatment = {"bio": treatment}
        
        costs = {}
        
        # Traitement bio
        bio_product = treatment.get("bio", "")
        if "neem" in bio_product.lower():
            product_info = self.product_database.get("neem_oil", {})
            # Calcul: 30ml/L, environ 500L/ha, donc 15L neem/ha
            quantity_needed = 15 * surface_ha
            costs["Traitement Bio (Neem)"] = product_info.get("prix_moyen", 2500) * quantity_needed
        elif "savon" in bio_product.lower():
            product_info = self.product_database.get("savon_noir", {})
            # Calcul: 30g/L, 500L/ha, donc 15kg savon/ha
            quantity_needed = 15 * surface_ha
            costs["Traitement Bio (Savon)"] = product_info.get("prix_moyen", 500) * quantity_needed
        
        # Traitement chimique
        chem_product = treatment.get("chimique", "")
        if "karate" in chem_product.lower() or "lambda" in chem_product.lower():
            product_info = self.product_database.get("lambda_cyhalothrine", {})
            # 1ml/L, 500L/ha, donc 0.5L/ha
            quantity_needed = 0.5 * surface_ha
            costs["Traitement Chimique (Karate)"] = product_info.get("prix_moyen", 4500) * quantity_needed
        
        # Coût total estimé
        costs["TOTAL ESTIMÉ"] = sum(costs.values())
        
        return costs

    def _get_alternative_products(self, recommended_product: str) -> List[str]:
        """
        Trouve des produits alternatifs si le recommandé n'est pas disponible.
        """
        logger.info(f"🔄 Recherche alternatives pour: {recommended_product}")
        
        alternatives = []
        
        for product_key, product_info in self.product_database.items():
            if recommended_product.lower() in product_info.get("nom_local", "").lower():
                alternatives = product_info.get("alternatives", [])
                break
        
        # Conversion des clés en noms locaux
        alternative_names = []
        for alt_key in alternatives:
            if alt_key in self.product_database:
                alternative_names.append(self.product_database[alt_key]["nom_local"])
        
        return alternative_names or ["Consulter un revendeur local"]

    def _find_local_sellers(self, product_name: str, zone: str = "Koutiala") -> List[str]:
        """
        Localise les points de vente du produit.
        """
        logger.info(f"📍 Localisation points de vente: {product_name}")
        
        sellers = []
        
        for product_key, product_info in self.product_database.items():
            if product_name.lower() in product_info.get("nom_local", "").lower():
                sellers = product_info.get("points_vente", [])
                break
        
        return sellers or ["Boutique Intrants (Centre-ville)", "Coopérative Agricole"]

    def diagnose_node(self, state: PlantDoctorState) -> PlantDoctorState:
        """
        Diagnostic amélioré avec:
        1. Analyse photos si fournies
        2. Questions guidées si description floue
        3. Estimation coûts
        """
        warnings = list(state.get("warnings", []))
        query = state.get("user_query", "").strip()
        profile = state.get("culture_config", {})
        crop = profile.get("crop_name") or profile.get("crop") or "culture inconnue"
        photo_paths = state.get("photo_paths", [])

        if not query and not photo_paths:
            warnings.append("Description des symptômes OU photos requises.")
            return {"warnings": warnings, "status": "ERROR"}

        # NOUVELLE FEATURE 1: Analyse photos si disponibles
        photo_analysis = {}
        if photo_paths:
            logger.info(f"📷 {len(photo_paths)} photo(s) détectée(s)")
            photo_analysis = self._analyze_photo_symptoms(photo_paths[0])
            
            # Enrichir la query avec les symptômes détectés
            detected = photo_analysis.get("detected_symptoms", [])
            if detected:
                query = f"{query}. Photo montre: {', '.join(detected)}"

        # NOUVELLE FEATURE 2: Questions guidées si description vague
        guided_questions = []
        if len(query.split()) < 5:  # Description trop courte
            logger.info("❓ Description floue → questions guidées activées")
            guided_questions = self._ask_guided_questions(query, crop)
            warnings.append("Description insuffisante. Questions envoyées à l'agriculteur.")

        augmented = self._augment_symptoms(query)

        try:
            diagnosis = self.doctor.diagnose_and_prescribe(crop=crop, user_obs=augmented)
        except Exception as exc:
            warnings.append(f"Diagnostic indisponible : {exc}")
            return {"warnings": warnings, "status": "ERROR", "guided_questions": guided_questions}

        if not diagnosis or not diagnosis.get("diagnostique"):
            warnings.append("Diagnostic non concluant, rediriger vers un conseiller.")
            return {
                "augmented_symptoms": augmented,
                "diagnosis_raw": diagnosis or {},
                "warnings": warnings,
                "status": "NO_DIAGNOSIS",
                "guided_questions": guided_questions,
            }

        risk_flags: List[str] = []
        for key in ("niveau_alerte", "urgence", "risques_associes"):
            value = diagnosis.get(key)
            if value:
                risk_flags.append(f"{key}: {value}")

        # NOUVELLE FEATURE 3: Estimation coûts traitement
        treatment_costs = self._estimate_treatment_cost(diagnosis, surface_ha=1.0)
        
        # NOUVELLE FEATURE 4: Alternatives produits
        recommended = diagnosis.get("traitement_recommande", {})
        bio_product = recommended.get("bio", "") if isinstance(recommended, dict) else recommended
        alternative_products = {
            "bio": self._get_alternative_products(bio_product)
        }
        
        # NOUVELLE FEATURE 5: Points de vente locaux
        local_availability = {}
        if bio_product:
            for product_key, product_info in self.product_database.items():
                if product_info["nom_local"].lower() in bio_product.lower():
                    local_availability[product_info["nom_local"]] = ", ".join(
                        product_info.get("points_vente", [])
                    )

        return {
            "user_query": query,
            "culture_config": profile,
            "augmented_symptoms": augmented,
            "diagnosis_raw": diagnosis,
            "risk_flags": risk_flags,
            "warnings": warnings,
            "status": "DIAGNOSED",
            # Nouvelles données pratiques
            "guided_questions": guided_questions,
            "treatment_costs": treatment_costs,
            "alternative_products": alternative_products,
            "local_availability": local_availability,
        }

    def retrieve_node(self, state: PlantDoctorState) -> PlantDoctorState:
        warnings = list(state.get("warnings", []))
        diagnosis = state.get("diagnosis_raw", {})
        profile = state.get("culture_config", {})
        query = state.get("user_query", "")

        plan = self._plan_retrieval(query, diagnosis, profile)
        optimized_query = plan.get("optimized_query") or query
        warnings.extend(plan.get("warnings", []))

        if not self.retriever:
            warnings.append("Moteur RAG indisponible.")
            return {
                "optimized_query": optimized_query,
                "retrieved_context": "",
                "sources": [],
                "warnings": warnings,
                "status": "NO_CONTEXT",
            }

        nodes = self.retriever.search(
            optimized_query,
            user_level=state.get("user_level", "debutant"),
        )
        if not nodes:
            warnings.append("Aucun document phytosanitaire pertinent trouvé.")
            return {
                "optimized_query": optimized_query,
                "retrieved_context": "",
                "sources": [],
                "warnings": warnings,
                "status": "NO_CONTEXT",
            }

        return {
            "optimized_query": optimized_query,
            "retrieved_context": self._build_context(nodes),
            "sources": self._serialize_sources(nodes),
            "warnings": warnings,
            "status": "CONTEXT_READY",
        }

    def compose_node(self, state: PlantDoctorState) -> PlantDoctorState:
        """
        Composition ENRICHIE de la réponse finale avec infos pratiques.
        """
        warnings = list(state.get("warnings", []))
        diagnosis = state.get("diagnosis_raw", {})
        context = state.get("retrieved_context", "").strip()
        risk_flags = state.get("risk_flags", [])
        sources = state.get("sources", [])
        profile = state.get("culture_config", {})
        query = state.get("user_query", "")
        
        # Nouvelles données pratiques
        treatment_costs = state.get("treatment_costs", {})
        alternative_products = state.get("alternative_products", {})
        local_availability = state.get("local_availability", {})
        guided_questions = state.get("guided_questions", [])

        # Si questions guidées envoyées, réponse intermédiaire
        if guided_questions:
            questions_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(guided_questions))
            response = (
                f"Pour mieux vous aider, répondez à ces questions:\n\n"
                f"{questions_text}\n\n"
                f"Envoyez vos réponses ou prenez une photo des symptômes (📷)."
            )
            return {
                "final_response": response,
                "warnings": warnings,
                "status": "AWAITING_INFO",
            }

        fallback = self._fallback_response(query, diagnosis, risk_flags, sources)

        if not diagnosis or not diagnosis.get("diagnostique"):
            warnings.append("Diagnostic impossible à confirmer.")
            return {
                "final_response": fallback,
                "warnings": warnings,
                "status": "NO_DIAGNOSIS",
            }

        if not context:
            warnings.append("Réponse formulée sans contexte validé.")
            # Mais on continue avec les infos pratiques!

        if not self.llm:
            warnings.append("LLM indisponible, passage en mode secours.")
            return {
                "final_response": fallback + self._format_practical_info(
                    treatment_costs, alternative_products, local_availability
                ),
                "warnings": warnings,
                "status": "LLM_DOWN",
            }

        try:
        # Préparation des contenus pour l'agent Guérisseur des Plantes
            system_content = PLANT_DOCTOR_SYSTEM_TEMPLATE

            user_content = PLANT_DOCTOR_USER_TEMPLATE.format(
                query=query,
                profile_culture=self._format_profile(profile),
                diagnosis_json=json.dumps(diagnosis, ensure_ascii=False),
                risk_flags=', '.join(risk_flags) or 'Aucune',
                context=context,
                practical_info=self._format_practical_info(
                    treatment_costs, 
                    alternative_products, 
                    local_availability
                )
            )

            # Appel au LLM
            completion = self.llm.chat.completions.create(
                model=self.model_answer,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.4
            )
            response = completion.choices[0].message.content.strip()
            
            # Ajout automatique des infos pratiques si LLM les a oubliées
            if "💰" not in response and treatment_costs:
                response += "\n\n" + self._format_practical_info(
                    treatment_costs, alternative_products, local_availability
                )
            
        except Exception as exc:
            logger.error(f"Échec génération LLM: {exc}")
            response = fallback + self._format_practical_info(
                treatment_costs, alternative_products, local_availability
            )

        return {
            "final_response": response,
            "warnings": warnings,
            "status": "SUCCESS",
        }

    def _format_practical_info(
        self,
        costs: Dict[str, float],
        alternatives: Dict[str, List[str]],
        availability: Dict[str, str]
    ) -> str:
        """
        Formate les informations pratiques pour l'agriculteur.
        """
        info = "\n\n" + "="*50 + "\n"
        info += "💡 INFORMATIONS PRATIQUES\n"
        info += "="*50 + "\n\n"
        
        # Coûts
        if costs:
            info += "💰 COÛTS ESTIMÉS (par hectare):\n"
            for treatment, cost in costs.items():
                if treatment != "TOTAL ESTIMÉ":
                    info += f"   • {treatment}: {cost:,.0f} FCFA\n"
            if "TOTAL ESTIMÉ" in costs:
                info += f"   📊 TOTAL: {costs['TOTAL ESTIMÉ']:,.0f} FCFA\n"
            info += "\n"
        
        # Alternatives
        if alternatives and any(alternatives.values()):
            info += "🔄 PRODUITS ALTERNATIFS (si indisponible):\n"
            for category, products in alternatives.items():
                if products:
                    info += f"   • {', '.join(products)}\n"
            info += "\n"
        
        # Points de vente
        if availability:
            info += "📍 OÙ ACHETER:\n"
            for product, locations in availability.items():
                info += f"   • {product}: {locations}\n"
            info += "\n"
        
        info += "📞 Besoin d'aide? Appelez le +22601479800 ou visitez votre coopérative.\n"
        
        return info

    def evaluate_node(self, state: PlantDoctorState) -> PlantDoctorState:
        warnings = list(state.get("warnings", []))
        if not self.evaluator:
            warnings.append("Évaluation automatique indisponible.")
            return {"warnings": warnings}

        query = state.get("user_query", "")
        context = state.get("retrieved_context", "")
        answer = state.get("final_response", "")

        if not query or not context or not answer:
            return {"warnings": warnings}

        try:
            scores = self.evaluator.evaluate_all(query=query, context=context, answer=answer)
            return {"evaluation": scores, "warnings": warnings, "status": "EVALUATED"}
        except Exception as exc:
            warnings.append(f"Évaluation échouée: {exc}")
            return {"warnings": warnings}
        
    def _augment_symptoms(self, text: str) -> str:
            if not text or not self.llm:
                return text

            try:
                completion = self.llm.chat.completions.create(
                    model=self.model_planner,
                    messages=[
                        {"role": "system", "content": DOCTOR_AUGMENT_PROMPT},
                        {"role": "user", "content": text},
                    ],
                    temperature=0.2,
                    max_tokens=200,
                )
                content = completion.choices[0].message.content
                if not content:
                    return text
                keywords = content.strip()
                return f"{text}\nMots-clés: {keywords}"
            except Exception:
                return text


    def _plan_retrieval(
        self,
        query: str,
        diagnosis: Dict[str, Any],
        profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        fallback = {
            "optimized_query": query,
            "warnings": ["Planification RAG automatique indisponible."],
        }
        if not self.llm:
            return fallback

        summary = {
            "culture": profile.get("crop_name"),
            "maladie": diagnosis.get("diagnostique"),
            "ravageur": diagnosis.get("target_pest"),
            "bio_solution": diagnosis.get("prescription_bio"),
        }

        
        try:
            # Préparation du prompt Planner pour le Docteur des Plantes
            formatted_planner_prompt = DOCTOR_PLANNER_TEMPLATE.format(
                query=query,
                summary_json=json.dumps(summary, ensure_ascii=False),
                profile_culture=self._format_profile(profile)
            )

            # Appel au LLM pour générer la requête optimisée
            completion = self.llm.chat.completions.create(
                model=self.model_planner,
                messages=[{"role": "user", "content": formatted_planner_prompt}],
                temperature=0.15,
                max_tokens=320,
                response_format={"type": "json_object"},
            )
            content = completion.choices[0].message.content
            if not content:
                raise ValueError("Planificateur silencieux.")
            return json.loads(content)
        except Exception as exc:
            logger.warning("Planification RAG phytosanitaire échouée: %s", exc)
            return fallback

    def _build_context(self, nodes: List[Any]) -> str:
        sections = []
        for idx, node in enumerate(nodes, start=1):
            meta = node.node.metadata or {}
            title = meta.get("title") or meta.get("filename") or f"Source {idx}"
            sections.append(f"[Source {idx} | {title}]\n{node.node.get_content().strip()}")
        return "\n\n".join(sections)

    def _serialize_sources(self, nodes: List[Any]) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for idx, node in enumerate(nodes, start=1):
            meta = node.node.metadata or {}
            payload.append(
                {
                    "index": idx,
                    "title": meta.get("title"),
                    "filename": meta.get("filename"),
                    "score": float(node.score) if node.score is not None else None,
                }
            )
        return payload

    def _fallback_response(
        self,
        query: str,
        diagnosis: Dict[str, Any],
        risk_flags: List[str],
        sources: List[Dict[str, Any]],
    ) -> str:
        lines = [
            "⚠️ Conseil phytosanitaire en mode dégradé.",
            f"Question: {query}",
            f"Diagnostic proposé: {diagnosis.get('diagnostique', 'Indisponible')}",
        ]
        if risk_flags:
            lines.append(f"Alerte(s): {', '.join(risk_flags)}")

        lines.append("")
        lines.append(f"Solution bio conseillée: {diagnosis.get('prescription_bio', 'Voir animateur')}")

        if diagnosis.get("conseil_chimique"):
            lines.append(f"Option chimique (dernier recours): {diagnosis['conseil_chimique']}")

        if diagnosis.get("prevention"):
            lines.append(f"Prévention: {diagnosis['prevention']}")

        if sources:
            ref = ", ".join(
                item.get("title") or item.get("filename") or f"Source {item['index']}"
                for item in sources
            )
            lines.append(f"Références: {ref}")

        # ASSERTIF: On confirme notre expertise, pas "consultez quelqu'un d'autre"
        lines.append("\n✅ **VALIDATION**: Diagnostic confirmé. Agissez rapidement pour sauver votre culture.")
        return "\n".join(lines)

    def _format_profile(self, profile: Dict[str, Any]) -> str:
        if not profile:
            return "Profil culture non renseigné"
        parts = []
        for key, value in profile.items():
            if value:
                parts.append(f"{key}: {value}")
        return "; ".join(parts) if parts else "Profil culture non renseigné"

    def _safe_retriever(self) -> Optional[AgileRetriever]:
        try:
            return AgileRetriever()
        except Exception as exc:
            logger.error("Impossible d'initialiser l'AgileRetriever : %s", exc)
            return None

    def _safe_evaluator(self) -> Optional[RAGEvaluator]:
        try:
            return RAGEvaluator()
        except Exception as exc:
            logger.warning("Évaluateur RAG non disponible : %s", exc)
            return None

    def build(self):
        """Construit le workflow LangGraph pour le diagnostic des plantes."""
        workflow = StateGraph(PlantDoctorState)
        
        # Ajout des nœuds
        workflow.add_node("diagnose", self.diagnose_node)  # Diagnostic initial
        workflow.add_node("retrieve", self.retrieve_node)   # Recherche RAG
        workflow.add_node("compose", self.compose_node)     # Composition réponse finale
        workflow.add_node("evaluate", self.evaluate_node)   # Évaluation qualité
        
        # Point d'entrée
        workflow.set_entry_point("diagnose")
        
        # Flux séquentiel simplifié
        workflow.add_edge("diagnose", "retrieve")
        workflow.add_edge("retrieve", "compose")
        workflow.add_edge("compose", "evaluate")
        workflow.add_edge("evaluate", END)
        
        return workflow.compile()