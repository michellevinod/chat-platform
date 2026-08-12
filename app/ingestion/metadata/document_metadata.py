from pydantic import BaseModel


class DocumentMetadata(BaseModel):

    project_id: str

    project_name: str

    document_id: str

    document_name: str

    document_type: str

    total_pages: int

    uploaded_by: str | None = None

    source: str = "upload"

    language: str | None = None