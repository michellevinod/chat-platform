from typing import Any

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """
    Metadata describing the uploaded document.
    """

    project_id: str
    project_name: str

    document_id: str
    document_name: str

    document_type: str
    total_pages: int = 0


class TableContent(BaseModel):
    """
    Represents a table extracted from a page.
    """

    table_id: str

    markdown: str
    data: list[dict[str, Any]]

    caption: str | None = None


class ImageContent(BaseModel):
    """
    Represents an image extracted from a page.
    """

    image_id: str

    image_path: str

    caption: str | None = None


class PageContent(BaseModel):
    """
    Represents all extracted information from one page.
    """

    page_number: int

    text: str = ""

    tables: list[TableContent] = Field(default_factory=list)

    images: list[ImageContent] = Field(default_factory=list)


class DocumentStatistics(BaseModel):
    """
    Statistics collected during parsing.
    """

    total_pages: int

    total_tables: int = 0

    total_images: int = 0

    total_characters: int = 0


class ParsedDocument(BaseModel):
    """
    Standard output returned by every parser.
    """

    metadata: DocumentMetadata

    pages: list[PageContent]

    statistics: DocumentStatistics