from app.repositories.qdrant_repository import QdrantRepository


class ProjectRepository:
    """
    Repository for project metadata querying from Qdrant.
    """

    def __init__(self) -> None:
        self._qdrant = QdrantRepository()

    def list_projects(self) -> list[str]:
        return self._qdrant.get_distinct_projects()
