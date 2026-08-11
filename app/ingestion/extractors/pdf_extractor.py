from pathlib import Path

import fitz

from app.ingestion.extractors.base_extractor import BaseExtractor
from app.ingestion.extractors.raw_models import (
    RawDocument,
    RawPage,
)


class PDFExtractor(BaseExtractor):
    """
    Extracts raw information from PDF documents.
    """

    def extract(
        self,
        file_path: Path,
    ) -> RawDocument:

        document = fitz.open(file_path)

        pages = []

        for page_number in range(len(document)):

            pages.append(
                RawPage(
                    page_number=page_number + 1,
                )
            )

        raw_document = RawDocument(
            file_name=file_path.name,
            total_pages=len(document),
            pages=pages,
        )

        document.close()

        return raw_document