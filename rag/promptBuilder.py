# prompt_builder.py
class PromptBuilder:
    def __init__(self, template: str = None):
        """
        Args:
            template: chaîne de format avec {query} et {context}
        """
        self.template = template or (
            "Tu es un expert local en hydrologie et climat.\n"
            "Réponds à la question suivante en t'appuyant uniquement sur les documents fournis.\n\n"
            "Documents :\n{context}\n\n"
            "Question : {query}\n\n"
            "Réponse :"
        )

    def build(self, query: str, context: str) -> str:
        """
        Construit un prompt complet à partir de la question et du contexte.

        Args:
            query: question utilisateur
            context: documents récupérés (concaténés)

        Returns:
            Prompt formaté (str)
        """
        return self.template.format(query=query.strip(), context=context.strip())
    

# prompt_templates.py

class MeteoPromptBuilder(PromptBuilder):
    def __init__(self):
        super().__init__(template=(
            "Tu es un expert en climat et météo au Burkina Faso.\n"
            "Réponds à la question en t'appuyant uniquement sur les données météorologiques suivantes :\n\n"
            "{context}\n\nQuestion : {query}\nRéponse :"
        ))

class FloodPromptBuilder(PromptBuilder):
    def __init__(self):
        super().__init__(template=(
            "Tu es un spécialiste des risques d'inondation.\n"
            "Analyse les données hydrologiques ci-dessous pour répondre à la question :\n\n"
            "{context}\n\nQuestion : {query}\nRéponse :"
        ))

class IrrigationPromptBuilder(PromptBuilder):
    def __init__(self):
        super().__init__(template=(
            "Tu es un conseiller agricole spécialisé en irrigation.\n"
            "Utilise les données climatiques et hydriques suivantes pour répondre :\n\n"
            "{context}\n\nQuestion : {query}\nRéponse :"
        ))