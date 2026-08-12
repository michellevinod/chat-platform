from pydantic import BaseModel, Field

from app.ingestion.extractors.raw_models import RawBlock


class Section(BaseModel):
    """
    Logical section inside a document.
    """

    title: str

    page_number: int

    blocks: list[RawBlock] = Field(default_factory=list)