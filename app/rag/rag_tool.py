from app.services.search_service import SearchService


class RAGTool:

    def __init__(self):

        self._search_service = SearchService()

    def search(
        self,
        query: str,
        limit: int = 8,
        project_name: str | None = None,
        document_name: str | None = None,
        chunk_type: str | None = None,
    ):

        return self._search_service.semantic_search(
            query=query,
            limit=limit,
            project_name=project_name,
            document_name=document_name,
            chunk_type=chunk_type,
        )

    def get_distinct_documents(
        self,
        project_name: str | None = None,
    ) -> list[str]:
        return self._search_service.get_distinct_documents(
            project_name=project_name,
        )

    def get_distinct_projects(
        self,
    ) -> list[str]:
        return self._search_service.get_distinct_projects()