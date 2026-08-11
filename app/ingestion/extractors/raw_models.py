from pathlib import Path

from pydantic import BaseModel, Field

from app.models.enums import BlockType


class RawBlock(BaseModel):
    """
    Base class for every extracted block.
    """

    page_number: int

    block_number: int

    block_type: BlockType

    bbox: tuple[float, float, float, float]


class RawTextBlock(RawBlock):
    """
    Raw text block extracted from a document.
    """

    text: str


class RawTableBlock(RawBlock):
    """
    Raw table extracted from a document.
    """

    markdown: str

    rows: list[list[str]]


class RawImageBlock(RawBlock):
    """
    Raw image extracted from a document.
    """

    image_name: str

    image_path: Path


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