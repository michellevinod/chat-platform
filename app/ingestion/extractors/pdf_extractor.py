from pathlib import Path

import pymupdf

from app.ingestion.extractors.base_extractor import BaseExtractor
from app.ingestion.extractors.raw_models import (
    RawDocument,
    RawPage,
    RawTextBlock,
)
from app.models.enums import BlockType


class PDFExtractor(BaseExtractor):
    """
    Extracts raw information from PDF documents.
    """

    def extract(
        self,
        file_path: Path,
    ) -> RawDocument:
        """
        Extracts raw document information from a PDF.
        """

        document = pymupdf.open(file_path)

        pages = self._extract_pages(document)

        raw_document = RawDocument(
            file_name=file_path.name,
            total_pages=len(document),
            pages=pages,
        )

        document.close()

        return raw_document

    def _extract_pages(
        self,
        document: pymupdf.Document,
    ) -> list[RawPage]:
        """
        Extract all pages from the PDF.
        """

        pages: list[RawPage] = []

        for page_number, page in enumerate(document, start=1):

            blocks = self._extract_text_blocks(
                page=page,
                page_number=page_number,
            )

            pages.append(
                RawPage(
                    page_number=page_number,
                    blocks=blocks,
                )
            )

        return pages

    def _extract_text_blocks(
        self,
        page: pymupdf.Page,
        page_number: int,
    ) -> list[RawTextBlock]:
        """
        Extract reading-order text blocks from a page.
        """

        extracted_blocks = page.get_text("dict")["blocks"]

        text_blocks: list[RawTextBlock] = []

        block_number = 0

        for block in extracted_blocks:

            # Ignore non-text blocks.
            if block["type"] != 0:
                continue

            text = ""

            for line in block["lines"]:
                for span in line["spans"]:
                    text += span["text"]

                text += "\n"

            text = text.strip()

            if not text:
                continue

            text_blocks.append(
                RawTextBlock(
                    page_number=page_number,
                    block_number=block_number,
                    block_type=BlockType.TEXT,
                    bbox=tuple(block["bbox"]),
                    text=text,
                )
            )

            block_number += 1

        return text_blocks