from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    text: str
    score: float

    project_id: str
    project_name: str

    document_id: str
    document_name: str
    document_type: str

    page_number: int
    chunk_number: int

    heading: str | None = None
    section: str | None = None

    chunk_type: str = "text"

    # Table metadata
    table_id: str | None = None
    table_number: str | None = None
    table_caption: str | None = None
    table_headers: list[str] = Field(default_factory=list)
    table_rows: list[list[str]] = Field(default_factory=list)

    # Image metadata
    image_id: str | None = None
    image_path: str | None = None
    image_number: str | None = None
    image_caption: str | None = None

    source: str = "upload"