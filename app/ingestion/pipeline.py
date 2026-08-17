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
    Extract -> Normalize -> Chunk -> Embed -> Qdrant Upsert.
    """

    def __init__(self, collection_name: str = "documents") -> None:
        self.collection_name = collection_name
        self.normalizer = PDFNormalizer()
        self.chunk_generator = ChunkGenerator()
        self.embedding_pipeline = EmbeddingPipeline()
        self.repository = QdrantRepository()

    def process_file(
        self,
        file_path: Path,
        project_name: str = "Default Project",
        project_id: str = "project_001",
        document_name: str | None = None,
    ) -> int:
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            extractor = PDFExtractor()
        elif ext == ".docx":
            extractor = DOCXExtractor()
        elif ext == ".pptx":
            extractor = PPTXExtractor()
        elif ext == ".xlsx":
            extractor = XLSXExtractor()
        else:
            raise ValueError(f"Unsupported document type: {ext}")

        doc = extractor.extract(file_path)
        doc = self.normalizer.normalize(doc)

        final_doc_name = document_name or file_path.name
        doc.metadata = {
            "project_id": project_id,
            "project_name": project_name,
            "document_name": final_doc_name,
            "document_type": ext.lstrip("."),
            "source": ext.lstrip("."),
        }

        chunks = self.chunk_generator.generate(doc)
        if not chunks:
            return 0

        chunks = self.embedding_pipeline.generate(chunks)

        self.repository.create_collection(
            collection_name=self.collection_name,
            vector_size=len(chunks[0].embedding),
        )

        self.repository.upsert_chunks(
            collection_name=self.collection_name,
            chunks=chunks,
        )

        return len(chunks)
