from app.rag.rag_tool import RAGTool
from app.rag.retrieved_chunk import RetrievedChunk


class Retriever:
    """
    Retrieves candidate chunks for a query from Qdrant.
    """

    def __init__(self) -> None:
        self._rag = RAGTool()

    def retrieve(
        self,
        query: str,
        limit: int = 8,
        project_name: str | None = None,
        document_name: str | None = None,
        chunk_type: str | None = None,
    ) -> list[RetrievedChunk]:
        return self._rag.search(
            query=query,
            limit=limit,
            project_name=project_name,
            document_name=document_name,
            chunk_type=chunk_type,
        )
