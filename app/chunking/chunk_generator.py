import uuid

from app.chunking.chunk_models import (
    ChunkMetadata,
    DocumentChunk,
)
from app.ingestion.extractors.raw_models import RawDocument
from app.models.enums import BlockType


class ChunkGenerator:
    """
    Generates semantic chunks from a parsed document.
    """

    def generate(
        self,
        document: RawDocument,
    ) -> list[DocumentChunk]:

        chunks: list[DocumentChunk] = []

        chunk_number = 0

        metadata = document.metadata

        for page in document.pages:

            for block in page.blocks:

                if block.block_type != BlockType.TEXT:
                    continue

                text = block.text.strip()

                if not text:
                    continue

                chunks.append(
                    DocumentChunk(
                        id=str(uuid.uuid4()),
                        text=text,
                        metadata=ChunkMetadata(
                            project_id=metadata["project_id"],
                            project_name=metadata["project_name"],
                            document_id=metadata["document_id"],
                            document_name=metadata["document_name"],
                            page_number=page.page_number,
                            chunk_number=chunk_number,
                        ),
                    )
                )

                chunk_number += 1

        return chunks