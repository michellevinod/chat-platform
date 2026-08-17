from __future__ import annotations

import uuid
from pathlib import Path

from app.chunking.chunk_generator import ChunkGenerator
from app.embeddings.embedding_pipeline import EmbeddingPipeline
from app.ingestion.extractors.docx_extractor import DOCXExtractor
from app.ingestion.extractors.pdf_extractor import PDFExtractor
from app.ingestion.extractors.ppt_extractor import PPTXExtractor
from app.ingestion.extractors.xlsx_extractor import XLSXExtractor
from app.ingestion.normalizers.pdf_normalizer import PDFNormalizer
from app.repositories.qdrant_repository import QdrantRepository


class IngestionPipeline:
    """
    End-to-end document ingestion pipeline:

    Extract
        -> Normalize
        -> Attach metadata
        -> Chunk
        -> Embed
        -> Store in Qdrant

    The pipeline is format-agnostic and supports:
    PDF, DOCX, PPTX, and XLSX.
    """

    def __init__(
        self,
        collection_name: str = "documents",
    ) -> None:
        self.collection_name = collection_name

        self.normalizer = PDFNormalizer()
        self.chunk_generator = ChunkGenerator()
        self.embedding_pipeline = EmbeddingPipeline()
        self.repository = QdrantRepository()

    def process_file(
        self,
        file_path: Path,
        project_name: str,
        project_id: str,
        document_name: str | None = None,
        document_id: str | None = None,
    ) -> int:
        """
        Process and ingest one uploaded document.

        All project/document metadata is supplied by the caller.
        Nothing project-specific or document-specific is hardcoded.
        """

        extension = file_path.suffix.lower()

        # -------------------------------------------------------------
        # Select extractor dynamically from the uploaded file type.
        # -------------------------------------------------------------

        if extension == ".pdf":
            extractor = PDFExtractor()

        elif extension == ".docx":
            extractor = DOCXExtractor()

        elif extension == ".pptx":
            extractor = PPTXExtractor()

        elif extension == ".xlsx":
            extractor = XLSXExtractor()

        else:
            raise ValueError(
                f"Unsupported document type: {extension}"
            )

        # -------------------------------------------------------------
        # Extract
        # -------------------------------------------------------------

        document = extractor.extract(file_path)

        # -------------------------------------------------------------
        # Normalize
        # -------------------------------------------------------------

        document = self.normalizer.normalize(document)

        # -------------------------------------------------------------
        # Resolve metadata dynamically
        # -------------------------------------------------------------

        final_document_name = (
            document_name
            or document.file_name
        )

        final_document_id = (
            document_id
            or str(uuid.uuid4())
        )

        document_type = extension.lstrip(".")
        source = document_type

        document.metadata.update(
            {
                "project_id": project_id,
                "project_name": project_name,
                "document_id": final_document_id,
                "document_name": final_document_name,
                "document_type": document_type,
                "source": source,
            }
        )

        # -------------------------------------------------------------
        # Generate chunks
        #
        # ChunkGenerator reads the metadata from RawDocument.metadata.
        # -------------------------------------------------------------

        chunks = self.chunk_generator.generate(
            document
        )

        if not chunks:
            return 0

        # -------------------------------------------------------------
        # Generate embeddings
        # -------------------------------------------------------------

        chunks = self.embedding_pipeline.generate(
            chunks
        )

        if not chunks:
            return 0

        # -------------------------------------------------------------
        # Create Qdrant collection if required
        # -------------------------------------------------------------

        embedding = chunks[0].embedding

        if not embedding:
            raise ValueError(
                "Embedding generation returned no vector."
            )

        self.repository.create_collection(
            collection_name=self.collection_name,
            vector_size=len(embedding),
        )

        # -------------------------------------------------------------
        # Store chunks + metadata in Qdrant
        # -------------------------------------------------------------

        self.repository.upsert_chunks(
            collection_name=self.collection_name,
            chunks=chunks,
        )

        return len(chunks)