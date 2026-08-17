from pathlib import Path
import uuid
import pymupdf

from app.ingestion.extractors.base_extractor import BaseExtractor
from app.ingestion.extractors.raw_models import (
    RawBlock,
    RawDocument,
    RawImageBlock,
    RawPage,
    RawTableBlock,
    RawTextBlock,
)
from app.models.enums import BlockType

IMAGE_DIR = Path("storage/images")
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


class PDFExtractor(BaseExtractor):
    """
    Extracts raw text, tables, and images from PDF documents.
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

            blocks: list[RawBlock] = []
            block_number = 0

            # 1. Extract text blocks
            text_blocks = self._extract_text_blocks(
                page=page,
                page_number=page_number,
                start_block_number=block_number,
            )
            blocks.extend(text_blocks)
            block_number += len(text_blocks)

            # 2. Extract tables
            try:
                table_finder = page.find_tables()
                if table_finder and table_finder.tables:
                    for tab_idx, table in enumerate(table_finder.tables):
                        md_table = table.to_markdown()
                        if md_table and md_table.strip():
                            df_headers = getattr(table, "header", None)
                            headers = [str(h) for h in df_headers.names] if df_headers and hasattr(df_headers, "names") else []
                            rows = table.extract() or []
                            blocks.append(
                                RawTableBlock(
                                    page_number=page_number,
                                    block_number=block_number,
                                    block_type=BlockType.TABLE,
                                    bbox=tuple(getattr(table, "bbox", (0.0, 0.0, 0.0, 0.0))),
                                    markdown=md_table.strip(),
                                    rows=[[str(c) if c is not None else "" for c in r] for r in rows],
                                    headers=headers,
                                    caption=f"Table {tab_idx + 1} on Page {page_number}",
                                )
                            )
                            block_number += 1
            except Exception:
                pass

            # 3. Extract images
            try:
                image_list = page.get_images(full=True)
                for img_idx, img_info in enumerate(image_list):
                    xref = img_info[0]
                    base_image = document.extract_image(xref)
                    image_bytes = base_image.get("image")
                    image_ext = base_image.get("ext", "png")
                    if image_bytes:
                        image_filename = f"img_{uuid.uuid4().hex[:8]}_p{page_number}_{img_idx + 1}.{image_ext}"
                        image_path = IMAGE_DIR / image_filename
                        with open(image_path, "wb") as f:
                            f.write(image_bytes)

                        blocks.append(
                            RawImageBlock(
                                page_number=page_number,
                                block_number=block_number,
                                block_type=BlockType.IMAGE,
                                bbox=(0.0, 0.0, 0.0, 0.0),
                                image_name=image_filename,
                                image_path=image_path,
                                caption=f"Figure on Page {page_number}",
                                alt_text=f"Extracted image {image_filename}",
                            )
                        )
                        block_number += 1
            except Exception:
                pass

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
        start_block_number: int = 0,
    ) -> list[RawTextBlock]:
        """
        Extract reading-order text blocks from a page.
        """

        extracted_blocks = page.get_text("dict")["blocks"]

        text_blocks: list[RawTextBlock] = []

        block_number = start_block_number

        for block in extracted_blocks:

            # Ignore non-text blocks.
            if block.get("type", 0) != 0:
                continue

            text = ""

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text += span.get("text", "")

                text += "\n"

            text = text.strip()

            if not text:
                continue

            text_blocks.append(
                RawTextBlock(
                    page_number=page_number,
                    block_number=block_number,
                    block_type=BlockType.TEXT,
                    bbox=tuple(block.get("bbox", (0.0, 0.0, 0.0, 0.0))),
                    text=text,
                )
            )

            block_number += 1

        return text_blocks