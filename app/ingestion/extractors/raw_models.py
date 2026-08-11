from pathlib import Path

from pydantic import BaseModel, Field


class RawTextBlock(BaseModel):
    """
    Raw text extracted from a document.
    """

    text: str

    bbox: tuple[float, float, float, float] | None = None


class RawTable(BaseModel):
    """
    Raw table extracted from a document.
    """

    markdown: str

    rows: list[list[str]]

    bbox: tuple[float, float, float, float] | None = None


class RawImage(BaseModel):
    """
    Raw image extracted from a document.
    """

    image_name: str

    image_path: Path

    bbox: tuple[float, float, float, float] | None = None


class RawPage(BaseModel):
    """
    Represents one extracted page.
    """

    page_number: int

    text_blocks: list[RawTextBlock] = Field(default_factory=list)

    tables: list[RawTable] = Field(default_factory=list)

    images: list[RawImage] = Field(default_factory=list)


class RawDocument(BaseModel):
    """
    Complete raw document extracted from a file.
    """

    file_name: str

    total_pages: int

    pages: list[RawPage] = Field(default_factory=list)