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
    Extracts text, tables, images, and captions from PDF documents.

    Image extraction is format-independent and uses PyMuPDF.

    Figure captions are detected from nearby text blocks and attached
    to the corresponding image whenever possible.

    Table captions are detected from nearby text blocks and attached
    to the corresponding table whenever possible.
    """

    # ---------------------------------------------------------------
    # FIGURE CAPTION PATTERN
    # ---------------------------------------------------------------

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

    # ---------------------------------------------------------------
    # TABLE CAPTION PATTERN
    #
    # Supports examples such as:
    #
    # Table 1. Harvey 1 well data record
    # Table 1 - Harvey 1 well data record
    # Table 1 — Harvey 1 well data record
    # Table 1: Harvey 1 well data record
    # TABLE 1. ...
    # Table 1 Harvey 1 well data record
    # ---------------------------------------------------------------

    TABLE_PATTERN = re.compile(
        r"^\s*"
        r"table"
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

                        if (
                            not md_table
                            or not md_table.strip()
                        ):
                            continue

                        df_headers = getattr(
                            table,
                            "header",
                            None,
                        )

                        headers = (
                            [
                                str(h)
                                if h is not None
                                else ""
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

                        table_bbox = tuple(
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
                        )

                        # ------------------------------------------------
                        # Find the REAL table caption from nearby text.
                        # ------------------------------------------------

                        table_caption_info = (
                            self._find_table_caption(
                                table_bbox=table_bbox,
                                text_blocks=text_blocks,
                            )
                        )

                        if table_caption_info:
                            caption = (
                                table_caption_info[
                                    "caption"
                                ]
                            )

                            table_number = (
                                table_caption_info[
                                    "table_number"
                                ]
                            )
                        else:
                            # Safe fallback when the PDF has no detectable
                            # table caption.
                            table_number = str(
                                tab_idx + 1
                            )

                            caption = (
                                f"Table "
                                f"{table_number} "
                                f"on Page "
                                f"{page_number}"
                            )

                        blocks.append(
                            RawTableBlock(
                                page_number=page_number,
                                block_number=block_number,
                                block_type=BlockType.TABLE,
                                bbox=table_bbox,
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
                                caption=caption,
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
    # TABLE CAPTION ASSOCIATION
    # ================================================================

    def _find_table_caption(
        self,
        table_bbox: tuple,
        text_blocks: list[RawTextBlock],
    ) -> dict | None:
        """
        Find the most likely table caption associated with a table.

        Preference:

        1. Caption immediately above the table.
        2. Caption immediately below the table.
        3. Nearest horizontally aligned table caption.

        Only text that matches the TABLE_PATTERN is considered.
        """

        if not text_blocks:
            return None

        table_x0, table_y0, table_x1, table_y1 = (
            table_bbox
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

            # --------------------------------------------------------
            # A PDF text block can contain multiple lines.
            # Check each line independently so a page heading or
            # paragraph containing the word "table" does not get
            # mistaken for a caption.
            # --------------------------------------------------------

            lines = [
                line.strip()
                for line in text.splitlines()
                if line.strip()
            ]

            for line_index, line in enumerate(lines):
                match = self.TABLE_PATTERN.match(line)

                if not match:
                    continue

                table_number = match.group(1)
                table_title = match.group(2).strip()

                # ----------------------------------------------------
                # If the caption wraps onto the next line, include
                # nearby continuation text when it is short enough.
                # ----------------------------------------------------

                caption_parts = [
                    table_title
                ]

                if line_index + 1 < len(lines):
                    next_line = lines[
                        line_index + 1
                    ]

                    # Avoid swallowing another caption or heading.
                    if (
                        not self.TABLE_PATTERN.match(
                            next_line
                        )
                        and not self.FIGURE_PATTERN.match(
                            next_line
                        )
                        and len(next_line) <= 200
                    ):
                        # Only append if the first line looks like a
                        # caption rather than a very long paragraph.
                        if len(table_title) < 180:
                            caption_parts.append(
                                next_line
                            )

                caption_title = " ".join(
                    caption_parts
                ).strip()

                caption = (
                    f"Table {table_number}. "
                    f"{caption_title}"
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

                (
                    block_x0,
                    block_y0,
                    block_x1,
                    block_y1,
                ) = block_bbox

                # ----------------------------------------------------
                # Horizontal overlap.
                # ----------------------------------------------------

                horizontal_overlap = max(
                    0.0,
                    min(
                        table_x1,
                        block_x1,
                    )
                    - max(
                        table_x0,
                        block_x0,
                    ),
                )

                table_width = max(
                    1.0,
                    table_x1 - table_x0,
                )

                overlap_ratio = (
                    horizontal_overlap
                    / table_width
                )

                # ----------------------------------------------------
                # Vertical relationship.
                #
                # Positive values mean the caption is separated from
                # the table vertically.
                # ----------------------------------------------------

                if block_y1 <= table_y0:
                    vertical_gap = (
                        table_y0 - block_y1
                    )

                    position = "above"

                elif block_y0 >= table_y1:
                    vertical_gap = (
                        block_y0 - table_y1
                    )

                    position = "below"

                else:
                    # Caption overlaps table vertically. This can
                    # happen in unusual PDF layouts.
                    vertical_gap = 0.0
                    position = "overlap"

                # ----------------------------------------------------
                # Captions should be reasonably close to the table.
                # ----------------------------------------------------

                if vertical_gap > 300:
                    continue

                # Prefer horizontally aligned captions.
                candidates.append(
                    {
                        "caption": caption,
                        "table_number": table_number,
                        "vertical_gap": vertical_gap,
                        "overlap_ratio": overlap_ratio,
                        "position": position,
                    }
                )

        if not candidates:
            return None

        # ------------------------------------------------------------
        # Ranking:
        #
        # 1. Caption above the table
        # 2. Horizontal alignment
        # 3. Short vertical distance
        #
        # This handles the common PDF layout where:
        #
        # Table 1. Caption
        #
        # [TABLE]
        # ------------------------------------------------------------

        candidates.sort(
            key=lambda candidate: (
                0
                if candidate["position"]
                == "above"
                else 1,
                -candidate["overlap_ratio"],
                candidate["vertical_gap"],
            )
        )

        return candidates[0]

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
    # FIGURE CAPTION ASSOCIATION
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
        Extract reading-order text blocks from the page.
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