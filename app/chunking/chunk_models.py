from pydantic import BaseModel, Field


from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    project_id: str = "project_001"
    project_name: str = "Default Project"

    document_id: str
    document_name: str
    document_type: str = "pdf"

    page_number: int
    chunk_number: int

    heading: str | None = None
    section: str | None = None

    chunk_type: str = "text"  # "text", "table", "image"
    table_id: str | None = None
    image_id: str | None = None
    image_path: str | None = None

    source: str = "upload"


class DocumentChunk(BaseModel):
    """
    Semantic chunk that will later be embedded and stored in Qdrant.
    """

    id: str

    text: str

    metadata: ChunkMetadata

    embedding: list[float] | None = None