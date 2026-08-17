from __future__ import annotations

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
    Generates semantic chunks from a normalized document.

    The generator is format-agnostic. PDF, DOCX, PPTX, and XLSX
    extractors produce normalized RawDocument objects.

    Project and document metadata are read dynamically from
    RawDocument.metadata. No project, document, or file type is
    hardcoded here.
    """

    def generate(
        self,
        document: RawDocument,
    ) -> list[DocumentChunk]:
        """
        Convert normalized document blocks into DocumentChunk objects.

        The document metadata is expected to contain:

            project_id
            project_name
            document_id
            document_name
            document_type
            source
        """

        chunks: list[DocumentChunk] = []

        metadata = document.metadata or {}

        project_id = metadata.get("project_id")
        project_name = metadata.get("project_name")
        document_id = metadata.get("document_id")
        document_name = metadata.get(
            "document_name",
            document.file_name,
        )
        document_type = metadata.get("document_type")
        source = metadata.get("source", "upload")

        # Required metadata must come from the ingestion pipeline.
        if not project_id:
            raise ValueError(
                "Missing required document metadata: project_id"
            )

        if not project_name:
            raise ValueError(
                "Missing required document metadata: project_name"
            )

        if not document_id:
            raise ValueError(
                "Missing required document metadata: document_id"
            )

        if not document_type:
            raise ValueError(
                "Missing required document metadata: document_type"
            )

        chunk_number = 0

        for page in document.pages:
            for block in page.blocks:

                # =========================================================
                # TEXT
                # =========================================================

                if block.block_type == BlockType.TEXT:
                    text = getattr(
                        block,
                        "text",
                        "",
                    ).strip()

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
                                heading=getattr(
                                    block,
                                    "heading",
                                    None,
                                ),
                                section=getattr(
                                    block,
                                    "section",
                                    None,
                                ),
                                chunk_type="text",
                                source=source,
                            ),
                        )
                    )

                    chunk_number += 1

                # =========================================================
                # TABLE
                # =========================================================

                elif block.block_type == BlockType.TABLE:
                    headers = getattr(
                        block,
                        "headers",
                        [],
                    )

                    rows = getattr(
                        block,
                        "rows",
                        [],
                    )

                    markdown = getattr(
                        block,
                        "markdown",
                        "",
                    ).strip()

                    # Normalize headers.
                    normalized_headers = [
                        str(header).strip()
                        if header is not None
                        else ""
                        for header in headers
                    ]

                    # Normalize rows while preserving
                    # the original row/column structure.
                    normalized_rows: list[list[str]] = []

                    for row in rows:
                        normalized_row = [
                            str(cell).strip()
                            if cell is not None
                            else ""
                            for cell in row
                        ]

                        normalized_rows.append(
                            normalized_row
                        )

                    # Ignore completely empty tables.
                    if (
                        not normalized_headers
                        and not normalized_rows
                        and not markdown
                    ):
                        continue

                    table_id = (
                        f"table_"
                        f"{document_id}_"
                        f"{page.page_number}_"
                        f"{block.block_number}"
                    )

                    # Use structured data as the canonical
                    # representation for semantic retrieval.
                    table_text = (
                        self._build_table_text(
                            headers=normalized_headers,
                            rows=normalized_rows,
                            fallback_markdown=markdown,
                        )
                    )

                    chunks.append(
                        DocumentChunk(
                            id=str(uuid.uuid4()),
                            text=table_text,
                            metadata=ChunkMetadata(
                                project_id=project_id,
                                project_name=project_name,
                                document_id=document_id,
                                document_name=document_name,
                                document_type=document_type,
                                page_number=page.page_number,
                                chunk_number=chunk_number,
                                heading=(
                                    getattr(
                                        block,
                                        "caption",
                                        None,
                                    )
                                    or getattr(
                                        block,
                                        "heading",
                                        None,
                                    )
                                ),
                                section=getattr(
                                    block,
                                    "section",
                                    None,
                                ),
                                chunk_type="table",
                                table_id=table_id,
                                table_headers=normalized_headers,
                                table_rows=normalized_rows,
                                source=source,
                            ),
                        )
                    )

                    chunk_number += 1

                # =========================================================
                # IMAGE
                # =========================================================

                elif block.block_type == BlockType.IMAGE:
                    image_name = getattr(
                        block,
                        "image_name",
                        "",
                    )

                    image_path = getattr(
                        block,
                        "image_path",
                        None,
                    )

                    image_id = getattr(
                        block,
                        "image_id",
                        None,
                    )

                    caption = (
                        getattr(
                            block,
                            "caption",
                            None,
                        )
                        or image_name
                        or "Document image"
                    )

                    image_text = (
                        f"Image / Figure: {caption} "
                        f"on page {page.page_number}."
                    )

                    # An image needs at least a path or identifier.
                    if not image_path and not image_id:
                        continue

                    chunks.append(
                        DocumentChunk(
                            id=str(uuid.uuid4()),
                            text=image_text,
                            metadata=ChunkMetadata(
                                project_id=project_id,
                                project_name=project_name,
                                document_id=document_id,
                                document_name=document_name,
                                document_type=document_type,
                                page_number=page.page_number,
                                chunk_number=chunk_number,
                                heading=caption,
                                section=getattr(
                                    block,
                                    "section",
                                    None,
                                ),
                                chunk_type="image",
                                image_id=image_id,
                                image_path=(
                                    str(image_path)
                                    if image_path
                                    else None
                                ),
                                source=source,
                            ),
                        )
                    )

                    chunk_number += 1

        return chunks

    @staticmethod
    def _build_table_text(
        headers: list[str],
        rows: list[list[str]],
        fallback_markdown: str = "",
    ) -> str:
        """
        Build a searchable textual representation of a table.

        This text is used for embedding/vector search only.

        The original structured table is separately preserved in:

            metadata.table_headers
            metadata.table_rows

        Therefore the table does not need to be reconstructed
        from this text representation when returned to the user.
        """

        parts: list[str] = []

        if headers:
            parts.append(
                "Table columns: "
                + " | ".join(headers)
            )

        for row in rows:
            if not row:
                continue

            parts.append(
                " | ".join(row)
            )

        if parts:
            return "\n".join(parts)

        return fallback_markdown