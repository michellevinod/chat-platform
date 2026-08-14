from pathlib import Path

from openpyxl import load_workbook

from app.ingestion.extractors.base_extractor import BaseExtractor
from app.ingestion.extractors.raw_models import (
    RawDocument,
    RawPage,
    RawTextBlock,
)
from app.models.enums import BlockType


class XLSXExtractor(BaseExtractor):
    """
    Extracts text from Microsoft Excel (.xlsx) workbooks.

    Each worksheet is treated as one page.
    """

    def extract(
        self,
        file_path: Path,
    ) -> RawDocument:

        workbook = load_workbook(
            filename=file_path,
            data_only=True,
        )

        pages: list[RawPage] = []

        page_number = 1

        for sheet in workbook.worksheets:

            blocks: list[RawTextBlock] = []

            block_number = 0

            # Sheet title
            blocks.append(
                RawTextBlock(
                    page_number=page_number,
                    block_number=block_number,
                    block_type=BlockType.TEXT,
                    bbox=(0.0, 0.0, 0.0, 0.0),
                    text=f"Worksheet: {sheet.title}",
                )
            )

            block_number += 1

            for row in sheet.iter_rows(values_only=True):

                values = []

                for cell in row:
                    if cell is None:
                        continue

                    values.append(str(cell))

                if not values:
                    continue

                blocks.append(
                    RawTextBlock(
                        page_number=page_number,
                        block_number=block_number,
                        block_type=BlockType.TEXT,
                        bbox=(0.0, 0.0, 0.0, 0.0),
                        text=" | ".join(values),
                    )
                )

                block_number += 1

            pages.append(
                RawPage(
                    page_number=page_number,
                    blocks=blocks,
                )
            )

            page_number += 1

        return RawDocument(
            file_name=file_path.name,
            total_pages=len(pages),
            pages=pages,
        )