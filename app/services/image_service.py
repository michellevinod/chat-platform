from app.rag.rag_tool import RAGTool
from app.rag.retrieved_chunk import RetrievedChunk


class ImageService:
    """
    Service for querying and retrieving image/figure references from documents.
    """

    def __init__(self) -> None:
        self._rag = RAGTool()

    def get_images(
        self,
        query: str,
        project_name: str | None = None,
        document_name: str | None = None,
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        return self._rag.search(
            query=query,
            limit=limit,
            project_name=project_name,
            document_name=document_name,
            chunk_type="image",
        )
