from enum import Enum


class DocumentType(str, Enum):
    """
    Supported document types.
    """

    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    TXT = "txt"


class BlockType(str, Enum):
    """
    Supported content block types.
    """

    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"