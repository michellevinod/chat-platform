from app.ingestion.extractors.raw_models import (
    RawDocument,
    RawTextBlock,
)
from app.models.enums import BlockType


class SemanticBlockBuilder:
    """
    Merges related text blocks before chunking.
    """

    def build(
        self,
        document: RawDocument,
    ) -> RawDocument:

        for page in document.pages:

            merged_blocks = []

            current_block = None

            for block in page.blocks:

                if block.block_type != BlockType.TEXT:
                    if current_block:
                        merged_blocks.append(current_block)
                        current_block = None

                    merged_blocks.append(block)
                    continue

                if current_block is None:
                    current_block = block
                    continue

                if self._should_merge(current_block, block):

                    current_block.text += "\n" + block.text

                else:

                    merged_blocks.append(current_block)

                    current_block = block

            if current_block:
                merged_blocks.append(current_block)

            page.blocks = merged_blocks

        return document

    def _should_merge(
        self,
        previous: RawTextBlock,
        current: RawTextBlock,
    ) -> bool:

        if previous.heading == current.heading:
            return True

        if len(previous.text) < 80:
            return True

        return False