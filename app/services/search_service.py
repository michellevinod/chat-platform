from app.repositories.qdrant_repository import QdrantRepository
from app.embeddings.sentence_transformer_embedding_service import (
    SentenceTransformerEmbeddingService,
)


class SearchService:
    """
    Performs semantic retrieval against Qdrant.

    Optional metadata filters allow callers to restrict retrieval
    to a project, document, specific chunk type, or page.
    """

    def __init__(self):
        self._embedding_service = (
            SentenceTransformerEmbeddingService()
        )

        self._repository = QdrantRepository()

    def semantic_search(
        self,
        query: str,
        limit: int = 5,
        project_name: str | None = None,
        document_name: str | None = None,
        project_id: str | None = None,
        document_id: str | None = None,
        chunk_type: str | None = None,
        page_number: int | None = None,
    ):
        """
        Perform semantic retrieval with optional metadata filters.

        When page_number is supplied, Qdrant restricts retrieval
        to that exact page before semantic ranking.
        """

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
            project_id=project_id,
            document_id=document_id,
            chunk_type=chunk_type,
            page_number=page_number,
        )