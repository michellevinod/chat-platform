from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    """
    Represents a retrieved chunk returned from semantic search.
    """

    text: str

    score: float

    project_name: str

    document_name: str

    page_number: int

    chunk_number: int