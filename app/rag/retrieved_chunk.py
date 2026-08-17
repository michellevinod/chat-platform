from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """
    Represents a retrieved chunk returned from semantic search.
    """

    text: str
    score: float

    project_id: str = "project_001"
    project_name: str = "Default Project"

    document_id: str = ""
    document_name: str
    document_type: str = "pdf"

    page_number: int
    chunk_number: int

    heading: str | None = None
    section: str | None = None

    chunk_type: str = "text"
    table_id: str | None = None
    image_id: str | None = None
    image_path: str | None = None

    source: str = "upload"