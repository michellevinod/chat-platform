from __future__ import annotations

import uuid

from app.chunking.chunk_models import (
    ChunkMetadata,
    DocumentChunk,
)
from app.ingestion.extractors.raw_models import RawDocument
from app.models.enums import BlockType


class ChunkGenerator:
    """
    Generates semantic chunks from a normalized document.

    The generator is format-agnostic. PDF, DOCX, PPTX, and XLSX
    extractors produce RawDocument objects which are converted into
    DocumentChunk objects here.

    Project and document metadata are read dynamically from
    RawDocument.metadata.
    """

    def generate(
        self,
        document: RawDocument,
    ) -> list[DocumentChunk]:
        """
        Convert document blocks into DocumentChunk objects.

        Required document metadata:

            project_id
            project_name
            document_id
            document_name
            document_type
            source

        Structured-content metadata such as table numbers,
        captions, image numbers, and image captions are preserved
        whenever supplied by the extractor.
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
        source = metadata.get(
            "source",
            "upload",
        )

        # ---------------------------------------------------------------
        # Validate required metadata
        # ---------------------------------------------------------------

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

        # ---------------------------------------------------------------
        # Process every page and block
        # ---------------------------------------------------------------

        for page in document.pages:
            for block in page.blocks:

                block_metadata = (
                    getattr(
                        block,
                        "metadata",
                        None,
                    )
                    or {}
                )

                # =======================================================
                # TEXT
                # =======================================================

                if block.block_type in {BlockType.TEXT, BlockType.OCR}:

                    text = (
                        getattr(
                            block,
                            "text",
                            "",
                        )
                        or ""
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
                                chunk_type=(
                                    "ocr"
                                    if block.block_type == BlockType.OCR
                                    else block_metadata.get("chunk_type", "text")
                                ),
                                source=(
                                    "ocr"
                                    if block.block_type == BlockType.OCR
                                    else source
                                ),
                            ),
                        )
                    )

                    chunk_number += 1

                # =======================================================
                # TABLE
                # =======================================================

                elif block.block_type == BlockType.TABLE:

                    headers = getattr(
                        block,
                        "headers",
                        [],
                    ) or []

                    rows = getattr(
                        block,
                        "rows",
                        [],
                    ) or []

                    markdown = (
                        getattr(
                            block,
                            "markdown",
                            "",
                        )
                        or ""
                    ).strip()

                    # ---------------------------------------------------
                    # Normalize headers
                    # ---------------------------------------------------

                    normalized_headers = [
                        str(header).strip()
                        if header is not None
                        else ""
                        for header in headers
                    ]

                    # ---------------------------------------------------
                    # Normalize rows while preserving structure
                    # ---------------------------------------------------

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

                    # ---------------------------------------------------
                    # Stable table identifier
                    # ---------------------------------------------------

                    table_id = (
                        f"table_"
                        f"{document_id}_"
                        f"{page.page_number}_"
                        f"{block.block_number}"
                    )

                    # ---------------------------------------------------
                    # Table metadata
                    # ---------------------------------------------------

                    table_number = (
                        block_metadata.get(
                            "table_number"
                        )
                        or getattr(
                            block,
                            "table_number",
                            None,
                        )
                    )

                    table_caption = (
                        getattr(
                            block,
                            "caption",
                            None,
                        )
                        or block_metadata.get(
                            "table_caption"
                        )
                        or None
                    )

                    if table_caption:
                        table_caption = str(
                            table_caption
                        ).strip()

                    if table_number is not None:
                        table_number = str(
                            table_number
                        ).strip()

                    # ---------------------------------------------------
                    # Build searchable table text
                    #
                    # The structured table itself is preserved separately
                    # in ChunkMetadata.
                    # ---------------------------------------------------

                    table_text = self._build_table_text(
                        headers=normalized_headers,
                        rows=normalized_rows,
                        fallback_markdown=markdown,
                    )

                    if table_caption:
                        table_text = (
                            f"Table caption: "
                            f"{table_caption}\n"
                            f"{table_text}"
                        )

                    if table_number:
                        table_text = (
                            f"Table number: "
                            f"{table_number}\n"
                            f"{table_text}"
                        )

                    # ---------------------------------------------------
                    # Create table chunk
                    # ---------------------------------------------------

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
                                    table_caption
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

                                table_number=table_number,

                                table_caption=table_caption,

                                table_headers=normalized_headers,

                                table_rows=normalized_rows,

                                source=source,
                            ),
                        )
                    )

                    chunk_number += 1

                # =======================================================
                # IMAGE
                # =======================================================

                elif block.block_type == BlockType.IMAGE:

                    image_name = (
                        getattr(
                            block,
                            "image_name",
                            "",
                        )
                        or ""
                    ).strip()

                    image_path = getattr(
                        block,
                        "image_path",
                        None,
                    )

                    # ---------------------------------------------------
                    # Preserve a stable internal image identifier.
                    # ---------------------------------------------------

                    image_id = (
                        image_name
                        or None
                    )

                    # ---------------------------------------------------
                    # Image/figure metadata
                    #
                    # Extractors may provide figure_number or
                    # image_number depending on the source format.
                    # ---------------------------------------------------

                    image_number = (
                        block_metadata.get(
                            "figure_number"
                        )
                        or block_metadata.get(
                            "image_number"
                        )
                        or getattr(
                            block,
                            "figure_number",
                            None,
                        )
                        or getattr(
                            block,
                            "image_number",
                            None,
                        )
                    )

                    image_caption = (
                        getattr(
                            block,
                            "caption",
                            None,
                        )
                        or block_metadata.get(
                            "figure_caption"
                        )
                        or block_metadata.get(
                            "image_caption"
                        )
                        or None
                    )

                    if image_number is not None:
                        image_number = str(
                            image_number
                        ).strip()

                    if image_caption:
                        image_caption = str(
                            image_caption
                        ).strip()

                    # ---------------------------------------------------
                    # Build searchable image text.
                    #
                    # Do not use the internal image filename as semantic
                    # content when no real caption exists.
                    # ---------------------------------------------------

                    image_text_parts: list[str] = [
                        "Image"
                    ]

                    if image_number:
                        image_text_parts.append(
                            str(image_number)
                        )

                    if image_caption:
                        image_text_parts.append(
                            f": {image_caption}"
                        )

                    image_text_parts.append(
                        f"on page {page.page_number}"
                    )

                    image_text = " ".join(
                        image_text_parts
                    )

                    # An image needs at least a path or identifier.

                    if not image_path and not image_id:
                        continue

                    # ---------------------------------------------------
                    # Create image chunk
                    # ---------------------------------------------------

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

                                heading=image_caption,

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

                                image_number=image_number,

                                image_caption=image_caption,

                                source=source,
                            ),
                        )
                    )

                    chunk_number += 1

        return chunks

    # ================================================================
    # TABLE SEARCH TEXT
    # ================================================================

    @staticmethod
    def _build_table_text(
        headers: list[str],
        rows: list[list[str]],
        fallback_markdown: str = "",
    ) -> str:
        """
        Build a searchable textual representation of a table.

        This representation is used for embedding/vector search only.

        The original structured table remains separately available in:

            metadata.table_headers
            metadata.table_rows

        Therefore, the table does not need to be reconstructed from
        the embedding text when returned to the user.
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
