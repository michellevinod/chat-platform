from app.embeddings.sentence_transformer_embedding_service import (
    SentenceTransformerEmbeddingService,
)
from app.repositories.qdrant_repository import QdrantRepository


class SearchService:

    def __init__(self):

        self._embedding_service = (
            SentenceTransformerEmbeddingService()
        )

        self._repository = (
            QdrantRepository()
        )

    def semantic_search(
        self,
        query: str,
        limit: int = 8,
        project_name: str | None = None,
        document_name: str | None = None,
        chunk_type: str | None = None,
    ):

        embedding = (
            self._embedding_service.generate_embedding(
                query
            )
        )

        return self._repository.search(
            collection_name="documents",
            query_vector=embedding,
            limit=limit,
            project_name=project_name,
            document_name=document_name,
            chunk_type=chunk_type,
        )

    def get_distinct_documents(
        self,
        project_name: str | None = None,
    ) -> list[str]:
        return self._repository.get_distinct_documents(
            collection_name="documents",
            project_name=project_name,
        )

    def get_distinct_projects(
        self,
    ) -> list[str]:
        return self._repository.get_distinct_projects(
            collection_name="documents",
        )