from app.repositories.qdrant_repository import QdrantRepository
from app.embeddings.sentence_transformer_embedding_service import (
    SentenceTransformerEmbeddingService,
)


class SearchService:

    def __init__(self):

        self._embedding_service = (
            SentenceTransformerEmbeddingService()
        )

        self._repository = QdrantRepository()

    def semantic_search(
        self,
        query: str,
        limit: int = 5,
    ):

        embedding = self._embedding_service.generate_embedding(
            query
        )

        return self._repository.search(
            collection_name="documents",
            query_vector=embedding,
            limit=limit,
        )