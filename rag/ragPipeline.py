from typing import Dict, List, Optional

class RAGPipeline:
    def __init__(
        self,
        retriever,
        generator,
        reranker=None,
        prompt_builder=None,
        evaluator=None,
        logger=None,
        loader=None,
        normalizer=None,
        indexer=None,
        text_builder=None,
        metadata_builder=None
    ):
        """
        Pipeline RAG modulaire avec indexation automatique si nécessaire.
        """
        self.retriever = retriever
        self.generator = generator
        self.reranker = reranker
        self.prompt_builder = prompt_builder
        self.evaluator = evaluator
        self.logger = logger

        self.loader = loader
        self.normalizer = normalizer
        self.indexer = indexer
        self.text_builder = text_builder
        self.metadata_builder = metadata_builder

    def ensure_index_ready(self):
        """
        Vérifie si la base est prête. Si vide, déclenche l'indexation.
        """
        if not hasattr(self.retriever, "collection"):
            print("⚠️ Retriever sans collection ChromaDB.")
            return

        try:
            count = self.retriever.collection.count()
            if count > 0:
                print(f"✅ Collection prête avec {count} documents.")
                return
            print("📂 Collection vide. Lancement de l'indexation...")
        except Exception as e:
            print("⚠️ Erreur d'accès à la collection :", e)
            return

        if not all([self.loader, self.normalizer, self.indexer, self.text_builder]):
            print("❌ Indexation impossible : composants manquants.")
            return

        try:
            raw_docs = self.loader.load_all(with_filenames=True)
        except TypeError:
            raw_docs = self.loader.load_all()

        try:
            records = self.normalizer.normalize_batch(raw_docs)
        except TypeError:
            # Si normalizer est un routeur multi-type
            records = []
            for doc, fname in raw_docs:
                records.extend(self.normalizer.adapter(doc, fname))

        self.indexer.index_records(records, self.text_builder, self.metadata_builder)
        print("✅ Indexation terminée.")

    def answer(self, query: str, top_k: int = 5) -> Dict:
        """
        Exécute le pipeline RAG complet : ensure_index → retrieve → rerank → prompt → generate → evaluate → log
        """
        self.ensure_index_ready()

        # Étape 1 : Retrieve
        retrieved_docs = self.retriever.retrieve(query, top_k=top_k)

        # Étape 2 : ReRank (optionnel)
        reranked_docs = self.reranker.rerank(query, retrieved_docs, top_k=top_k) if self.reranker else retrieved_docs

        # Étape 3 : Prompt construction
        context = "\n".join([doc.get("excerpt", "") for doc in reranked_docs])
        prompt = self.prompt_builder.build(query, context) if self.prompt_builder else context

        # Étape 4 : Génération
        response = self.generator.generate(query, prompt)

        # Étape 5 : Évaluation (optionnelle)
        scores = self.evaluator.evaluate(query, response, reranked_docs) if self.evaluator else None

        # Étape 6 : Logging (optionnel)
        if self.logger:
            self.logger.log(query, response, reranked_docs, scores, prompt)

        return {
            "query": query,
            "response": response,
            "sources": reranked_docs,
            "scores": scores,
            "prompt": prompt
        }