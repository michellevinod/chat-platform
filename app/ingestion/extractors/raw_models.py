from pathlib import Path

from pydantic import BaseModel, Field

from app.models.enums import BlockType

from app.ingestion.metadata.document_metadata import DocumentMetadata


class RawBlock(BaseModel):
    """
    Base class for every extracted block.
    """

    page_number: int

    block_number: int

    block_type: BlockType

    bbox: tuple[float, float, float, float]

    heading: str | None = None

    section: str | None = None

    reading_order: int | None = None

    language: str | None = None

    metadata: dict = Field(default_factory=dict)


class RawTextBlock(RawBlock):
    """
    Raw text block extracted from a document.
    """

    text: str

    tokens: int | None = None


class RawTableBlock(RawBlock):
    """
    Raw table extracted from a document.
    """

    markdown: str

    rows: list[list[str]]

    caption: str | None = None

    headers: list[str] = Field(default_factory=list)


class RawImageBlock(RawBlock):
    """
    Raw image extracted from a document.
    """

    image_name: str

    image_path: Path

    caption: str | None = None

    alt_text: str | None = None


class RawPage(BaseModel):
    """
    Represents one extracted page.
    """

    page_number: int

    blocks: list[RawBlock] = Field(default_factory=list)


class RawDocument(BaseModel):
    """
    Complete raw document extracted from a file.
    """

    file_name: str

    total_pages: int

    metadata: dict = Field(default_factory=dict)

    pages: list[RawPage] = Field(default_factory=list)