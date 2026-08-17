from app.services.search_service import SearchService


class RAGTool:
    """
    Thin interface over SearchService.

    Supports normal semantic retrieval as well as metadata-scoped
    retrieval for tables and images.
    """

    def __init__(self):
        self._search_service = SearchService()

    def search(
        self,
        query: str,
        limit: int = 5,
        project_name: str | None = None,
        document_name: str | None = None,
        project_id: str | None = None,
        document_id: str | None = None,
        chunk_type: str | None = None,
    ):
        return self._search_service.semantic_search(
            query=query,
            limit=limit,
            project_name=project_name,
            document_name=document_name,
            project_id=project_id,
            document_id=document_id,
            chunk_type=chunk_type,
        )