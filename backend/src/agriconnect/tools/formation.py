from typing import Any,List,Dict

import json
import logging
import re


logger = logging.getLogger("FormationCoachTool")

class FormationTool:
    def __init__(self,llm,model_planner="llama-3.3-70b-versatile",model_answer="llama-3.3-70b-versatile"    ):
        self.llm = llm
        self.model_planner = model_planner
        self.model_answer = model_answer    

    def _extract_json_block(self, text: str) -> Dict[str, Any]:
        matches = re.findall(r"\{[\s\S]*?\}", text)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        return json.loads(text)

    def _plan_retrieval(self, query: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        fallback = {
            "optimized_query": query,
            "modules": [],
            "prerequisites": [],
            "reasoning": "",
            "warnings": [],
        }

        if not self.llm:
            fallback["warnings"].append("LLM indisponible pour planifier la recherche.")
            return fallback

        profile_text = self._format_profile(profile)
        planner_prompt = (
            "Tu es l'orchestrateur pédagogique d'AgriConnect. "
            "Analyse la question suivante et prépare une recherche RAG.\n"
            f"Profil apprenant : {profile_text}\n"
            f"Question : {query}\n\n"
            'Réponds en JSON avec : {"optimized_query": "...", "modules": ["..."], '
            '"prerequisites": ["..."], "reasoning": "..."}'
        )

        try:
            completion = self.llm.chat.completions.create(
                model=self.model_planner,
                messages=[{"role": "user", "content": planner_prompt}],
                temperature=0.2,
                max_tokens=400,
                response_format={"type": "json_object"},
            )
            content = completion.choices[0].message.content
            if not content:
                raise ValueError("Réponse vide du planificateur.")
            plan = json.loads(content)
            return {
                "optimized_query": plan.get("optimized_query") or query,
                "modules": plan.get("modules", []),
                "prerequisites": plan.get("prerequisites", []),
                "reasoning": plan.get("reasoning", ""),
                "warnings": [],
            }
        except Exception as exc:
            logger.warning("Planification RAG impossible : %s", exc)
            fallback["warnings"].append("Planification RAG automatique indisponible.")
            return fallback

    def _build_context(self, nodes: List[Any]) -> str:
        sections = []
        for idx, node in enumerate(nodes, start=1):
            metadata = node.node.metadata or {}
            label = metadata.get("title") or metadata.get("filename") or f"Source {idx}"
            chunk = node.node.get_content().strip()
            sections.append(f"[Source {idx} | {label}]\n{chunk}")
        return "\n\n".join(sections)

    def _serialize_sources(self, nodes: List[Any]) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for idx, node in enumerate(nodes, start=1):
            metadata = node.node.metadata or {}
            payload.append(
                {
                    "index": idx,
                    "title": metadata.get("title"),
                    "filename": metadata.get("filename"),
                    "score": float(node.score) if node.score is not None else None,
                }
            )
        return payload

    def _format_profile(self, profile: Dict[str, Any]) -> str:
        if not profile:
            return "Non renseigné"
        parts: List[str] = []
        for key, value in profile.items():
            if value in (None, "", []):
                continue
            parts.append(f"{key}: {value}")
        return "; ".join(parts) if parts else "Non renseigné"

    def _analyze_request(self, query: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        fallback = {
            "intent": "FORMATION",
            "focus_topics": [],
            "field_actions": [],
            "safety_flags": [],
            "urgency": "NORMAL",
            "warnings": [],
        }

        if not self.llm:
            fallback["warnings"].append("LLM indisponible pour analyser la demande.")
            return fallback

        profile_text = self._format_profile(profile)
        analyzer_prompt = (
            "Tu es l'ingénieur pédagogique expert d'AgriConnect. Ton rôle est de qualifier la demande de l'utilisateur "
            "pour optimiser la recherche documentaire (RAG) et garantir la sécurité des conseils.\n\n"
            
            f"PROFIL APPRENANT : {profile_text}\n"
            f"QUESTION : {query}\n\n"
            
            "CONSIGNES DE GÉNÉRATION JSON :\n"
            "1. is_relevant : (Boolean) True si la question concerne l'agriculture, l'élevage, la météo agricole ou la formation. False pour tout sujet hors-domaine (sport, politique, cuisine non-agricole, etc.).\n"
            "2. rejection_reason : (String) Si is_relevant=False, explique poliment pourquoi tu ne peux pas répondre (en restant dans ton rôle d'expert agricole).\n"
            "3. intent : Choisir parmi [FORMATION, URGENCE, CONSEIL].\n"
            "4. focus_topics : Liste de mots-clés optimisés pour une recherche sémantique (ex: 'entretien culture niébé', 'lutte chenilles').\n"
            "5. field_actions : Liste les catégories techniques à vérifier dans les documents (ex: 'densité de semis', 'dosage engrais'). Ne donne JAMAIS de chiffres ou de méthodes à ce stade.\n"
            "6. safety_flags : Identifie les risques critiques (ex: 'toxicité pesticides', 'santé animale', 'érosion') nécessitant une attention particulière.\n"
            "7. urgency : Choisir selon l'impact sur la récolte : [NORMAL, HAUTE, CRITIQUE].\n\n"
            
            "RÉPONDS UNIQUEMENT SOUS CE FORMAT JSON :\n"
            "{\n"
            '  "is_relevant": true,\n'
            '  "rejection_reason": "",\n'
            '  "intent": "...",\n'
            '  "focus_topics": [],\n'
            '  "field_actions": [],\n'
            '  "safety_flags": [],\n'
            '  "urgency": "...",\n'
            '  "warnings": []\n'
            "}"
        )

        try:
            completion = self.llm.chat.completions.create(
                model=self.model_planner,
                messages=[{"role": "user", "content": analyzer_prompt}],
                temperature=0.1,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            content = completion.choices[0].message.content
            if not content:
                raise ValueError("Réponse vide de l'analyseur.")
            
            analysis = json.loads(content)
            return {
                "is_relevant": analysis.get("is_relevant", True), 
                "rejection_reason": analysis.get("rejection_reason", ""),
                "intent": analysis.get("intent", "FORMATION"),
                "focus_topics": analysis.get("focus_topics", []),
                "field_actions": analysis.get("field_actions", []),
                "safety_flags": analysis.get("safety_flags", []),
                "urgency": analysis.get("urgency", "NORMAL"),
                "warnings": analysis.get("warnings", []),
            }
        except Exception as e:
            logger.warning("Analyse de requête impossible : %s", e)
            fallback["warnings"].append("Analyse de requête automatique indisponible.")
            return fallback
     
    def _fallback_answer(
        self,
        query: str,
        profile_text: str,
        prerequisites: List[str],
        modules: List[str],
        sources: List[Dict[str, Any]],
    ) -> str:
        # Construction d'un texte propre pour les sources
        source_titles = []
        for s in sources:
            title = s.get("title") or s.get("filename") or f"Source {s.get('index')}"
            source_titles.append(title)
        
        sources_text = ", ".join(source_titles) if source_titles else "Fiches techniques locales"

        return (
            "Désolé, je rencontre une difficulté technique momentanée pour générer une réponse détaillée.\n\n"
            "Cependant, voici les ressources identifiées pour vous aider :\n\n"
            f"❓ **Question** : {query}\n"
            f"📚 **Sujet** : {', '.join(modules) if modules else 'Agriculture générale'}\n"
            f"📄 **Documents trouvés** : {sources_text}\n\n"
            "Conseil : Vous pouvez consulter ces documents ou reformuler votre question."
        )

