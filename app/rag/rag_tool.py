from app.services.search_service import SearchService
from app.rag.retrieved_chunk import RetrievedChunk


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
        page_number: int | None = None,
    ):
        """
        Search documents using semantic retrieval with optional
        metadata filters.
        """

        return self._search_service.semantic_search(
            query=query,
            limit=limit,
            project_name=project_name,
            document_name=document_name,
            project_id=project_id,
            document_id=document_id,
            chunk_type=chunk_type,
            page_number=page_number,
        )

    def search_many(
        self,
        queries: list[str] | tuple[str, ...],
        limit: int = 8,
        project_name: str | None = None,
        document_name: str | None = None,
        chunk_type: str | None = None,
        page_number: int | None = None,
    ) -> list[RetrievedChunk]:
        """Fuse a small set of retrieval formulations in one scope."""
        fused: dict[tuple[str, int, int, str], tuple[float, RetrievedChunk]] = {}
        usable_queries = [query for query in dict.fromkeys(queries) if query.strip()]

        for query in usable_queries[:4]:
            candidates = self.search(
                query=query,
                limit=max(limit, 8),
                project_name=project_name,
                document_name=document_name,
                chunk_type=chunk_type,
                page_number=page_number,
            )
            for rank, chunk in enumerate(candidates, start=1):
                key = (
                    chunk.document_name,
                    chunk.page_number,
                    chunk.chunk_number,
                    chunk.chunk_type,
                )
                rrf_score = 1.0 / (60 + rank)
                existing = fused.get(key)
                if existing is None:
                    fused[key] = (rrf_score, chunk)
                else:
                    fused[key] = (existing[0] + rrf_score, existing[1])

        ranked = sorted(
            fused.values(),
            key=lambda item: item[0],
            reverse=True,
        )
        for fused_score, chunk in ranked:
            chunk.score = fused_score

        return sorted(
            (chunk for _, chunk in ranked),
            key=lambda chunk: chunk.score,
            reverse=True,
        )[: max(limit, 1)]

    def get_page_content(self, page_number: int, project_name: str | None = None, document_name: str | None = None):
        return self._search_service.page_search(page_number, project_name, document_name)

    def get_representative_document_content(self, document_name: str, project_name: str | None = None, limit: int = 16):
        return self._search_service.representative_document_search(document_name, project_name, limit)

    def get_distinct_documents(self, project_name: str | None = None) -> list[str]:
        return self._search_service.distinct_documents(project_name)
