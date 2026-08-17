import uuid
from pathlib import Path

from app.chunking.chunk_models import (
    ChunkMetadata,
    DocumentChunk,
)
from app.ingestion.extractors.raw_models import (
    RawDocument,
    RawImageBlock,
    RawTableBlock,
    RawTextBlock,
)
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
        doc_ext = Path(document.file_name).suffix.lower().lstrip(".")
        document_type = metadata.get("document_type", doc_ext or "pdf")
        source = metadata.get("source", doc_ext or "upload")

        for page in document.pages:

            for block in page.blocks:

                if block.block_type == BlockType.TEXT:
                    text = getattr(block, "text", "").strip()
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
                                document_type=document_type,
                                page_number=page.page_number,
                                chunk_number=chunk_number,
                                heading=block.heading,
                                section=block.section,
                                chunk_type="text",
                                source=source,
                            ),
                        )
                    )
                    chunk_number += 1

                elif block.block_type == BlockType.TABLE:
                    table_md = getattr(block, "markdown", "").strip()
                    if not table_md:
                        continue

                    table_id = f"tab_p{page.page_number}_b{block.block_number}"
                    chunks.append(
                        DocumentChunk(
                            id=str(uuid.uuid4()),
                            text=table_md,
                            metadata=ChunkMetadata(
                                project_id=project_id,
                                project_name=project_name,
                                document_id=document_id,
                                document_name=document_name,
                                document_type=document_type,
                                page_number=page.page_number,
                                chunk_number=chunk_number,
                                heading=getattr(block, "caption", None) or block.heading,
                                section=block.section,
                                chunk_type="table",
                                table_id=table_id,
                                source=source,
                            ),
                        )
                    )
                    chunk_number += 1

                elif block.block_type == BlockType.IMAGE:
                    img_name = getattr(block, "image_name", "")
                    img_path = str(getattr(block, "image_path", ""))
                    caption = getattr(block, "caption", "") or img_name
                    img_text = f"Image / Figure: {caption} on page {page.page_number}."
                    chunks.append(
                        DocumentChunk(
                            id=str(uuid.uuid4()),
                            text=img_text,
                            metadata=ChunkMetadata(
                                project_id=project_id,
                                project_name=project_name,
                                document_id=document_id,
                                document_name=document_name,
                                document_type=document_type,
                                page_number=page.page_number,
                                chunk_number=chunk_number,
                                heading=caption,
                                section=block.section,
                                chunk_type="image",
                                image_id=img_name,
                                image_path=img_path,
                                source=source,
                            ),
                        )
                    )
                    chunk_number += 1

        return chunks