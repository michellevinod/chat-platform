from app.services.search_service import SearchService


class RAGTool:

    def __init__(self):

        self._search_service = SearchService()

    def search(
        self,
        query: str,
        limit: int = 5,
    ):

        return self._search_service.semantic_search(
            query=query,
            limit=limit,
        )