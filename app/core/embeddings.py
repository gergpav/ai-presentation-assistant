from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


class DocumentIndex:
    def __init__(self):
        self.documents = []
        self.embeddings = None
        self.is_built = False

    def add_documents(self, documents: List[dict]):
        for doc in documents:
            if doc.get("text") and doc["text"].strip():
                self.documents.append({
                    "content": doc["text"],
                    "source": doc["metadata"]["filename"]
                })
        logger.info(f"Добавлено документов: {len(self.documents)}")

    def build_index(self):
        if not self.documents:
            return

        texts = [doc["content"] for doc in self.documents]
        self.embeddings = model.encode(texts, convert_to_numpy=True)
        self.is_built = True

    def search(self, query: str, k: int = 5) -> List[Tuple[str, str, str]]:
        if not self.is_built or not self.documents:
            return []

        try:
            query_embedding = model.encode([query])

            # Простое решение - используем numpy напрямую
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            top_indices = np.argsort(similarities)[-k:][::-1]

            results = []
            for idx in top_indices:
                doc = self.documents[idx]
                results.append((doc["content"], "text", doc["source"]))  # ← здесь все правильно

            return results

        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")
            return []

    def get_stats(self):
        return {
            "documents_count": len(self.documents),
            "index_built": self.is_built
        }


document_index = DocumentIndex()