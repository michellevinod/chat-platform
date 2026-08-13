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

        metadata = document.metadata or {}

        project_id = metadata.get("project_id", "project_001")
        project_name = metadata.get("project_name", "Default Project")
        document_id = metadata.get("document_id", str(uuid.uuid4()))
        document_name = metadata.get(
            "document_name",
            document.file_name,
        )

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
                            project_id=project_id,
                            project_name=project_name,
                            document_id=document_id,
                            document_name=document_name,
                            page_number=page.page_number,
                            chunk_number=chunk_number,
                        ),
                    )
                )

                chunk_number += 1

        return chunks