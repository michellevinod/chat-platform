from pathlib import Path

from docx import Document

from app.ingestion.extractors.base_extractor import BaseExtractor
from app.ingestion.extractors.raw_models import (
    RawDocument,
    RawPage,
    RawTextBlock,
)
from app.models.enums import BlockType


class DOCXExtractor(BaseExtractor):
    """
    Extracts text from Microsoft Word (.docx) documents.
    """

    def extract(
        self,
        file_path: Path,
    ) -> RawDocument:

        document = Document(file_path)

        blocks: list[RawTextBlock] = []

        block_number = 0

        for paragraph in document.paragraphs:

            text = paragraph.text.strip()

            if not text:
                continue

            blocks.append(
                RawTextBlock(
                    page_number=1,
                    block_number=block_number,
                    block_type=BlockType.TEXT,
                    bbox=(0.0, 0.0, 0.0, 0.0),
                    text=text,
                )
            )

            block_number += 1

        page = RawPage(
            page_number=1,
            blocks=blocks,
        )

        return RawDocument(
            file_name=file_path.name,
            total_pages=1,
            pages=[page],
        )