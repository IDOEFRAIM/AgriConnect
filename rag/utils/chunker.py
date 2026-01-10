from typing import List, Dict, Any, Optional
import re
import logging

class TextChunker:
    """
    Service de découpage intelligent de texte (Chunking).
    Optimisé pour les bulletins PDF et articles web.
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger("rag.text_chunker")

    def split_text(self, text: str) -> List[str]:
        """
        Découpe un texte brut en chunks avec recouvrement.
        Tente de respecter les fins de phrases pour la sémantique.
        """
        if not text:
            return []
            
        # Nettoyage basique
        text = re.sub(r'\s+', ' ', text).strip()
        
        chunks = []
        if len(text) <= self.chunk_size:
            return [text]
            
        # Split by sentences to avoid cutting in the middle of a word/phrase
        sentences = re.split(r'(?<=[.!?]) +', text)
        current_chunk = []
        current_len = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            if current_len + sentence_len > self.chunk_size:
                # If current chunk is full, add it
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(chunk_text)
                    
                    # Overlap management: keep the last sentences
                    overlap_len = 0
                    overlap_sentences = []
                    for s in reversed(current_chunk):
                        if overlap_len + len(s) < self.chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_len += len(s)
                        else:
                            break
                    current_chunk = overlap_sentences
                    current_len = overlap_len
                
            current_chunk.append(sentence)
            current_len += sentence_len
            
        # Ajouter le dernier morceau
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Applique le chunking sur une liste de documents dictionnaires.
        Expand le document original en N documents (1 par chunk).
        """
        chunked_docs = []
        for doc in documents:
            original_text = doc.get("content") or doc.get("text_content") or ""
            chunks = self.split_text(original_text)
            
            for i, chunk in enumerate(chunks):
                new_doc = doc.copy()
                new_doc["chunk_id"] = i
                new_doc["text_content"] = chunk
                new_doc["original_id"] = doc.get("id", "unknown")
                # On retire le contenu full pour alléger
                if "content" in new_doc: del new_doc["content"]
                chunked_docs.append(new_doc)
        
        return chunked_docs