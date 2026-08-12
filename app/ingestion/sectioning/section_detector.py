from app.ingestion.extractors.raw_models import (
    RawDocument,
    RawTextBlock,
)

from app.ingestion.sectioning.section import Section
from app.models.enums import BlockType


class SectionDetector:
    """
    Detects logical sections from extracted blocks.
    """

    def detect(
        self,
        document: RawDocument,
    ) -> list[Section]:

        sections: list[Section] = []

        current_section = Section(
            title="Document Start",
            page_number=1,
        )

        for page in document.pages:

            for block in page.blocks:

                if block.block_type != BlockType.TEXT:

                    current_section.blocks.append(block)
                    continue

                if self._is_heading(block.text):

                    if current_section.blocks:

                        sections.append(current_section)

                    current_section = Section(
                        title=block.text.strip(),
                        page_number=page.page_number,
                    )

                current_section.blocks.append(block)

        if current_section.blocks:
            sections.append(current_section)

        return sections

    def _is_heading(
        self,
        text: str,
    ) -> bool:

        text = text.strip()

        headings = {
            "Management Summary",
            "Operations Summary",
            "Current Operations",
            "Planned Operations",
            "Comments",
            "Safety Summary",
            "Mud Information",
            "Bit Information",
            "Well Information",
            "Daily Drilling Report",
        }

        if text in headings:
            return True

        if text.startswith("##"):
            return True

        return False