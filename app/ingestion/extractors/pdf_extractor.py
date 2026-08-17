from __future__ import annotations

import re
import uuid
from pathlib import Path

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
    Extracts text, tables, images, and figure captions from PDF documents.

    Image extraction is format-independent and uses PyMuPDF.

    Figure captions are detected from nearby text blocks and attached
    to the corresponding image whenever possible.
    """

    # Common caption patterns:
    #
    # Fig. 10—Something
    # Fig. 10 - Something
    # Fig. 10: Something
    # Figure 10—Something
    # Figure 10: Something
    FIGURE_PATTERN = re.compile(
        r"^\s*"
        r"(?:fig(?:ure)?\.?)"
        r"\s*"
        r"(\d+)"
        r"\s*"
        r"(?:[-–—:.)]\s*|\s+)"
        r"(.+?)"
        r"\s*$",
        flags=re.IGNORECASE | re.DOTALL,
    )

    def extract(
        self,
        file_path: Path,
    ) -> RawDocument:
        """
        Extract raw document information from a PDF.
        """

        document = pymupdf.open(file_path)

        try:
            pages = self._extract_pages(document)

            return RawDocument(
                file_name=file_path.name,
                total_pages=len(document),
                pages=pages,
            )

        finally:
            document.close()

    # ================================================================
    # PAGE EXTRACTION
    # ================================================================

    def _extract_pages(
        self,
        document: pymupdf.Document,
    ) -> list[RawPage]:
        """
        Extract all pages from the PDF.
        """

        pages: list[RawPage] = []

        for page_number, page in enumerate(
            document,
            start=1,
        ):
            blocks: list[RawBlock] = []
            block_number = 0

            # --------------------------------------------------------
            # 1. Extract text blocks
            # --------------------------------------------------------

            text_blocks = self._extract_text_blocks(
                page=page,
                page_number=page_number,
                start_block_number=block_number,
            )

            blocks.extend(text_blocks)
            block_number += len(text_blocks)

            # --------------------------------------------------------
            # 2. Extract tables
            # --------------------------------------------------------

            try:
                table_finder = page.find_tables()

                if (
                    table_finder
                    and table_finder.tables
                ):
                    for tab_idx, table in enumerate(
                        table_finder.tables
                    ):
                        md_table = table.to_markdown()

                        if not md_table or not md_table.strip():
                            continue

                        df_headers = getattr(
                            table,
                            "header",
                            None,
                        )

                        headers = (
                            [
                                str(h)
                                for h in df_headers.names
                            ]
                            if (
                                df_headers
                                and hasattr(
                                    df_headers,
                                    "names",
                                )
                            )
                            else []
                        )

                        rows = table.extract() or []

                        blocks.append(
                            RawTableBlock(
                                page_number=page_number,
                                block_number=block_number,
                                block_type=BlockType.TABLE,
                                bbox=tuple(
                                    getattr(
                                        table,
                                        "bbox",
                                        (
                                            0.0,
                                            0.0,
                                            0.0,
                                            0.0,
                                        ),
                                    )
                                ),
                                markdown=md_table.strip(),
                                rows=[
                                    [
                                        (
                                            str(cell)
                                            if cell is not None
                                            else ""
                                        )
                                        for cell in row
                                    ]
                                    for row in rows
                                ],
                                headers=headers,
                                caption=(
                                    f"Table "
                                    f"{tab_idx + 1} "
                                    f"on Page "
                                    f"{page_number}"
                                ),
                            )
                        )

                        block_number += 1

            except Exception:
                # Table extraction should never prevent
                # text/image extraction.
                pass

            # --------------------------------------------------------
            # 3. Extract images + associate captions
            # --------------------------------------------------------

            try:
                image_blocks = self._extract_image_blocks(
                    document=document,
                    page=page,
                    page_number=page_number,
                    start_block_number=block_number,
                    text_blocks=text_blocks,
                )

                blocks.extend(image_blocks)

            except Exception:
                # Image extraction should never prevent
                # the rest of the PDF from being ingested.
                pass

            pages.append(
                RawPage(
                    page_number=page_number,
                    blocks=blocks,
                )
            )

        return pages

    # ================================================================
    # IMAGE EXTRACTION
    # ================================================================

    def _extract_image_blocks(
        self,
        document: pymupdf.Document,
        page: pymupdf.Page,
        page_number: int,
        start_block_number: int,
        text_blocks: list[RawTextBlock],
    ) -> list[RawImageBlock]:
        """
        Extract images from a page and associate nearby figure
        captions with them.

        The association is based on image position and nearby
        caption text rather than assuming that the first image
        on a page is a particular figure.
        """

        image_blocks: list[RawImageBlock] = []

        # ------------------------------------------------------------
        # Get actual image information including bounding boxes.
        # ------------------------------------------------------------

        image_infos = page.get_image_info(
            xrefs=True
        )

        # Some PDFs can expose duplicate image references.
        # Keep each displayed image occurrence separately.
        for img_idx, image_info in enumerate(
            image_infos
        ):
            xref = image_info.get(
                "xref"
            )

            if not xref:
                continue

            try:
                base_image = document.extract_image(
                    xref
                )
            except Exception:
                continue

            image_bytes = base_image.get(
                "image"
            )

            image_ext = base_image.get(
                "ext",
                "png",
            )

            if not image_bytes:
                continue

            image_filename = (
                f"img_"
                f"{uuid.uuid4().hex[:8]}"
                f"_p{page_number}"
                f"_{img_idx + 1}"
                f".{image_ext}"
            )

            image_path = (
                IMAGE_DIR
                / image_filename
            )

            with open(
                image_path,
                "wb",
            ) as file:
                file.write(
                    image_bytes
                )

            bbox = tuple(
                image_info.get(
                    "bbox",
                    (
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ),
                )
            )

            caption = self._find_image_caption(
                image_bbox=bbox,
                text_blocks=text_blocks,
            )

            if not caption:
                caption = (
                    f"Figure on "
                    f"Page {page_number}"
                )

            image_blocks.append(
                RawImageBlock(
                    page_number=page_number,
                    block_number=(
                        start_block_number
                        + img_idx
                    ),
                    block_type=BlockType.IMAGE,
                    bbox=bbox,
                    image_name=image_filename,
                    image_path=image_path,
                    caption=caption,
                    alt_text=(
                        f"Extracted image "
                        f"{image_filename}"
                    ),
                )
            )

        return image_blocks

    # ================================================================
    # CAPTION ASSOCIATION
    # ================================================================

    def _find_image_caption(
        self,
        image_bbox: tuple,
        text_blocks: list[RawTextBlock],
    ) -> str | None:
        """
        Find the most likely figure caption associated with an image.

        Preference:

        1. Caption immediately below the image.
        2. Caption immediately above the image.
        3. Nearest caption on the page.

        Only text that looks like a figure caption is considered.
        """

        if not text_blocks:
            return None

        image_x0, image_y0, image_x1, image_y1 = (
            image_bbox
        )

        candidates = []

        for text_block in text_blocks:

            text = (
                getattr(
                    text_block,
                    "text",
                    "",
                )
                or ""
            ).strip()

            if not text:
                continue

            match = self.FIGURE_PATTERN.match(
                text
            )

            if not match:
                continue

            figure_number = match.group(1)
            figure_title = match.group(2).strip()

            caption = (
                f"Figure {figure_number}: "
                f"{figure_title}"
            )

            block_bbox = getattr(
                text_block,
                "bbox",
                (
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ),
            )

            block_x0, block_y0, block_x1, block_y1 = (
                block_bbox
            )

            # Horizontal overlap helps prevent selecting a
            # caption belonging to another figure beside it.
            horizontal_overlap = max(
                0.0,
                min(
                    image_x1,
                    block_x1,
                )
                - max(
                    image_x0,
                    block_x0,
                ),
            )

            image_width = max(
                1.0,
                image_x1 - image_x0,
            )

            overlap_ratio = (
                horizontal_overlap
                / image_width
            )

            vertical_gap = min(
                abs(
                    block_y1
                    - image_y0
                ),
                abs(
                    block_y0
                    - image_y1
                ),
            )

            # Prefer captions that are reasonably aligned
            # horizontally with the image.
            if overlap_ratio <= 0:
                continue

            candidates.append(
                {
                    "caption": caption,
                    "figure_number": figure_number,
                    "vertical_gap": vertical_gap,
                    "overlap_ratio": overlap_ratio,
                    "block_y0": block_y0,
                    "block_y1": block_y1,
                }
            )

        if not candidates:
            return None

        # Prefer horizontal alignment first, then distance.
        candidates.sort(
            key=lambda candidate: (
                -candidate["overlap_ratio"],
                candidate["vertical_gap"],
            )
        )

        best = candidates[0]

        # Avoid associating a caption that is extremely far
        # away from the image.
        #
        # The threshold is deliberately generous because PDF
        # layouts vary significantly.
        if best["vertical_gap"] > 250:
            return None

        return best["caption"]

    # ================================================================
    # TEXT EXTRACTION
    # ================================================================

    def _extract_text_blocks(
        self,
        page: pymupdf.Page,
        page_number: int,
        start_block_number: int = 0,
    ) -> list[RawTextBlock]:
        """
        Extract reading-order text blocks from a page.
        """

        extracted_blocks = page.get_text(
            "dict"
        )["blocks"]

        text_blocks: list[RawTextBlock] = []

        block_number = start_block_number

        for block in extracted_blocks:

            # Ignore non-text blocks.
            if block.get(
                "type",
                0,
            ) != 0:
                continue

            text = ""

            for line in block.get(
                "lines",
                [],
            ):
                for span in line.get(
                    "spans",
                    [],
                ):
                    text += span.get(
                        "text",
                        "",
                    )

                text += "\n"

            text = text.strip()

            if not text:
                continue

            text_blocks.append(
                RawTextBlock(
                    page_number=page_number,
                    block_number=block_number,
                    block_type=BlockType.TEXT,
                    bbox=tuple(
                        block.get(
                            "bbox",
                            (
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                            ),
                        )
                    ),
                    text=text,
                )
            )

            block_number += 1

        return text_blocks