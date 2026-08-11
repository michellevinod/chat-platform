import re

from app.ingestion.extractors.raw_models import (
    RawDocument,
    RawTextBlock,
)
from app.models.enums import BlockType

from .base_normalizer import BaseNormalizer


class PDFNormalizer(BaseNormalizer):

    def normalize(
        self,
        document: RawDocument,
    ) -> RawDocument:

        for page in document.pages:

            normalized_blocks = []

            for block in page.blocks:

                if block.block_type != BlockType.TEXT:
                    normalized_blocks.append(block)
                    continue

                text = self._clean_text(block.text)

                if not text:
                    continue

                normalized_blocks.append(
                    RawTextBlock(
                        page_number=block.page_number,
                        block_number=block.block_number,
                        block_type=BlockType.TEXT,
                        bbox=block.bbox,
                        text=text,
                    )
                )

            page.blocks = normalized_blocks

        return document

    def _clean_text(
        self,
        text: str,
    ) -> str:

        text = text.replace("\n", " ")

        text = re.sub(r"\s+", " ", text)

        return text.strip()