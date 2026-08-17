from app.agents.base_agent import BaseAgent
from app.rag.rag_tool import RAGTool
from app.rag.retrieved_chunk import RetrievedChunk


class DocumentAgent(BaseAgent):
    """
    Handles document-specific retrieval capabilities with filtering.

    Supports project, document, chunk-type, and page-level
    filtering for deterministic document queries.
    """

    def __init__(self):
        self._rag = RAGTool()

    def execute(
        self,
        query: str,
        limit: int = 8,
        project_name: str | None = None,
        document_name: str | None = None,
        chunk_type: str | None = None,
        page_number: int | None = None,
    ) -> list[RetrievedChunk]:
        return self._rag.search(
            query=query,
            limit=limit,
            project_name=project_name,
            document_name=document_name,
            chunk_type=chunk_type,
            page_number=page_number,
        )

    def search_tables(
        self,
        query: str,
        limit: int = 5,
        project_name: str | None = None,
        document_name: str | None = None,
        page_number: int | None = None,
    ) -> list[RetrievedChunk]:
        return self._rag.search(
            query=query,
            limit=limit,
            project_name=project_name,
            document_name=document_name,
            chunk_type="table",
            page_number=page_number,
        )

    def search_images(
        self,
        query: str,
        limit: int = 5,
        project_name: str | None = None,
        document_name: str | None = None,
        page_number: int | None = None,
    ) -> list[RetrievedChunk]:
        return self._rag.search(
            query=query,
            limit=limit,
            project_name=project_name,
            document_name=document_name,
            chunk_type="image",
            page_number=page_number,
        )

    def get_document_summary_chunks(
        self,
        document_name: str,
        limit: int = 15,
        project_name: str | None = None,
    ) -> list[RetrievedChunk]:
        return self._rag.search(
            query=(
                "summary overview purpose "
                "introduction conclusion findings"
            ),
            limit=limit,
            project_name=project_name,
            document_name=document_name,
        )

    def get_project_summary_chunks(
        self,
        project_name: str,
        limit: int = 20,
    ) -> list[RetrievedChunk]:
        return self._rag.search(
            query=(
                "project summary overview "
                "objectives results"
            ),
            limit=limit,
            project_name=project_name,
        )