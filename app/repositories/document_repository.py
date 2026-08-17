from app.repositories.qdrant_repository import QdrantRepository


class DocumentRepository:
    """
    Repository for document metadata querying from Qdrant.
    """

    def __init__(self) -> None:
        self._qdrant = QdrantRepository()

    def list_documents(self, project_name: str | None = None) -> list[str]:
        return self._qdrant.get_distinct_documents(project_name=project_name)
