from pathlib import Path

from pptx import Presentation

from app.ingestion.extractors.base_extractor import BaseExtractor
from app.ingestion.extractors.raw_models import (
    RawDocument,
    RawPage,
    RawTextBlock,
)
from app.models.enums import BlockType


class PPTXExtractor(BaseExtractor):
    """
    Extracts text from Microsoft PowerPoint (.pptx) files.
    Each slide is treated as one page.
    """

    def extract(
        self,
        file_path: Path,
    ) -> RawDocument:

        presentation = Presentation(file_path)

        pages: list[RawPage] = []

        for slide_number, slide in enumerate(
            presentation.slides,
            start=1,
        ):

            blocks: list[RawTextBlock] = []

            block_number = 0

            for shape in slide.shapes:

                if not hasattr(shape, "text"):
                    continue

                text = shape.text.strip()

                if not text:
                    continue

                blocks.append(
                    RawTextBlock(
                        page_number=slide_number,
                        block_number=block_number,
                        block_type=BlockType.TEXT,
                        bbox=(0.0, 0.0, 0.0, 0.0),
                        text=text,
                    )
                )

                block_number += 1

            pages.append(
                RawPage(
                    page_number=slide_number,
                    blocks=blocks,
                )
            )

        return RawDocument(
            file_name=file_path.name,
            total_pages=len(pages),
            pages=pages,
        )