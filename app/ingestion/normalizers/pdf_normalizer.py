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
            current_block: RawTextBlock | None = None

            for block in page.blocks:

                if block.block_type != BlockType.TEXT:
                    if current_block is not None:
                        normalized_blocks.append(current_block)
                        current_block = None

                    normalized_blocks.append(block)
                    continue

                cleaned_block = self._normalize_text_block(block)

                if cleaned_block is None:
                    continue

                if current_block is None:
                    current_block = cleaned_block
                    continue

                if self._should_merge(current_block, cleaned_block):
                    current_block = self._merge_text_blocks(
                        current_block,
                        cleaned_block,
                    )
                else:
                    normalized_blocks.append(current_block)
                    current_block = cleaned_block

            if current_block is not None:
                normalized_blocks.append(current_block)

            page.blocks = normalized_blocks

        return document

    def _normalize_text_block(
        self,
        block: RawTextBlock,
    ) -> RawTextBlock | None:
        text = self._clean_text(block.text)

        if not text:
            return None

        return RawTextBlock(
            page_number=block.page_number,
            block_number=block.block_number,
            block_type=BlockType.TEXT,
            bbox=block.bbox,
            text=text,
            heading=block.heading,
            section=block.section,
            reading_order=block.reading_order,
            language=block.language,
            metadata=dict(block.metadata or {}),
        )

    def _should_merge(
        self,
        previous: RawTextBlock,
        current: RawTextBlock,
    ) -> bool:
        if previous.heading != current.heading and (
            previous.heading or current.heading
        ):
            return False

        if previous.section != current.section and (
            previous.section or current.section
        ):
            return False

        previous_bbox = previous.bbox or (0.0, 0.0, 0.0, 0.0)
        current_bbox = current.bbox or (0.0, 0.0, 0.0, 0.0)

        overlap = max(
            0.0,
            min(previous_bbox[2], current_bbox[2])
            - max(previous_bbox[0], current_bbox[0]),
        )

        previous_width = max(1.0, previous_bbox[2] - previous_bbox[0])
        current_width = max(1.0, current_bbox[2] - current_bbox[0])
        overlap_ratio = overlap / max(previous_width, current_width, 1.0)

        if current_bbox[1] >= previous_bbox[3]:
            vertical_gap = current_bbox[1] - previous_bbox[3]
        else:
            vertical_gap = previous_bbox[1] - current_bbox[3]

        if overlap_ratio >= 0.55:
            return True

        if vertical_gap <= 12:
            return True

        if len(previous.text) < 80 and len(current.text) < 80 and vertical_gap <= 20:
            return True

        return False

    @staticmethod
    def _merge_text_blocks(
        previous: RawTextBlock,
        current: RawTextBlock,
    ) -> RawTextBlock:
        merged_text = "\n".join(
            [
                part
                for part in (
                    previous.text,
                    current.text,
                )
                if part and part.strip()
            ]
        )

        merged_bbox = (
            min(previous.bbox[0], current.bbox[0]),
            min(previous.bbox[1], current.bbox[1]),
            max(previous.bbox[2], current.bbox[2]),
            max(previous.bbox[3], current.bbox[3]),
        )

        return RawTextBlock(
            page_number=previous.page_number,
            block_number=previous.block_number,
            block_type=BlockType.TEXT,
            bbox=merged_bbox,
            text=merged_text,
            heading=previous.heading or current.heading,
            section=previous.section or current.section,
            reading_order=previous.reading_order,
            language=previous.language or current.language,
            metadata={
                **(previous.metadata or {}),
                **(current.metadata or {}),
            },
        )

    def _clean_text(
        self,
        text: str,
    ) -> str:

        text = text.replace("\n", " ")

        text = re.sub(r"\s+", " ", text)

        return text.strip()
