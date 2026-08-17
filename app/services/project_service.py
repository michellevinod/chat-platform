from app.repositories.qdrant_repository import QdrantRepository


class ProjectService:
    """
    Service for querying and managing projects in Qdrant.
    """

    def __init__(self) -> None:
        self._repo = QdrantRepository()

    def list_projects(self) -> list[str]:
        return self._repo.get_distinct_projects()

    def list_project_documents(self, project_name: str) -> list[str]:
        return self._repo.get_distinct_documents(project_name=project_name)
