from abc import ABC, abstractmethod
from pathlib import Path

from app.ingestion.extractors.raw_models import RawDocument


class BaseExtractor(ABC):
    """
    Base interface for all document extractors.
    """

    @abstractmethod
    def extract(
        self,
        file_path: Path,
    ) -> RawDocument:
        """
        Extracts raw document content.

        Args:
            file_path:
                Path to the uploaded file.

        Returns:
            RawDocument
        """
        raise NotImplementedError