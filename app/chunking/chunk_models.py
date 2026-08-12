from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    project_id: str
    project_name: str

    document_id: str
    document_name: str

    page_number: int

    chunk_number: int

    source: str = "pdf"


class DocumentChunk(BaseModel):
    """
    Semantic chunk that will later be embedded and stored in Qdrant.
    """

    id: str

    text: str

    metadata: ChunkMetadata

    embedding: list[float] | None = None