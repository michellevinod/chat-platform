from abc import ABC, abstractmethod
from pathlib import Path

from app.models.parsed_document import ParsedDocument


class BaseParser(ABC):
    """
    Abstract base class for all document parsers.

    Every supported document type (PDF, DOCX, PPTX, XLSX, etc.)
    must inherit from this class and implement the parse method.
    """

    @abstractmethod
    def parse(
        self,
        file_path: Path,
    ) -> ParsedDocument:
        """
        Parses a document and returns a standardized ParsedDocument.

        Args:
            file_path:
                Path to the uploaded document.

        Returns:
            ParsedDocument
        """
        raise NotImplementedError