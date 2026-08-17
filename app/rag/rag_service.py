from app.rag.citation_service import CitationService
from app.rag.reranker import Reranker
from app.rag.retriever import Retriever
from app.rag.retrieved_chunk import RetrievedChunk
from app.schemas.chat_schema import Citation


class RAGService:
    """
    Core RAG pipeline service.
    """

    def __init__(self) -> None:
        self._retriever = Retriever()
        self._reranker = Reranker()
        self._citation_service = CitationService()

    def search_and_rank(
        self,
        query: str,
        limit: int = 8,
        project_name: str | None = None,
        document_name: str | None = None,
        chunk_type: str | None = None,
    ) -> list[RetrievedChunk]:
        candidates = self._retriever.retrieve(
            query=query,
            limit=limit,
            project_name=project_name,
            document_name=document_name,
            chunk_type=chunk_type,
        )
        return self._reranker.rerank(candidates, top_k=limit)

    def get_citations(
        self,
        chunks: list[RetrievedChunk],
    ) -> list[Citation]:
        return self._citation_service.build_citations(chunks)
