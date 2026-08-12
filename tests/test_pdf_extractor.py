from pathlib import Path

from app.chunking.chunk_generator import ChunkGenerator
from app.embeddings.embedding_pipeline import EmbeddingPipeline
from app.ingestion.builders.semantic_block_builder import SemanticBlockBuilder
from app.ingestion.extractors.pdf_extractor import PDFExtractor
from app.ingestion.metadata.metadata_generator import MetadataGenerator
from app.ingestion.normalizers.pdf_normalizer import PDFNormalizer
from app.repositories.qdrant_repository import QdrantRepository


def main():

    # ==========================================================
    # Extract
    # ==========================================================

    extractor = PDFExtractor()

    document = extractor.extract(
        Path("storage/uploads/PDF_BenchmarkTester_63Pages.pdf")
    )

    # ==========================================================
    # Normalize
    # ==========================================================

    normalizer = PDFNormalizer()

    document = normalizer.normalize(document)

    # ==========================================================
    # Semantic Block Builder
    # ==========================================================

    builder = SemanticBlockBuilder()

    document = builder.build(document)

    # ==========================================================
    # Metadata
    # ==========================================================

    metadata_generator = MetadataGenerator()

    document = metadata_generator.generate(
        document=document,
        project_name="Forge Project",
        project_id="project_001",
        document_id="doc_001",
    )

    # ==========================================================
    # Chunk Generation
    # ==========================================================

    chunk_generator = ChunkGenerator()

    chunks = chunk_generator.generate(document)

    print()
    print("=" * 80)
    print(f"Chunks Generated : {len(chunks)}")
    print("=" * 80)

    # Show first 5 chunks

    for chunk in chunks[:5]:

        print()
        print(chunk.id)
        print(chunk.metadata)
        print(chunk.text)
        print("-" * 80)

    # ==========================================================
    # Embedding Generation
    # ==========================================================

    embedding_pipeline = EmbeddingPipeline()

    chunks = embedding_pipeline.generate(chunks)

    print()
    print("=" * 80)
    print("First Chunk Embedding")
    print("=" * 80)

    print(chunks[0].text)
    print()
    print(f"Embedding Dimension : {len(chunks[0].embedding)}")

    # ==========================================================
    # Upload to Qdrant
    # ==========================================================

    repository = QdrantRepository()

    repository.create_collection(
        collection_name="documents",
        vector_size=len(chunks[0].embedding),
    )

    repository.upsert_chunks(
        collection_name="documents",
        chunks=chunks,
    )

    print()
    print("=" * 80)
    print("UPLOAD SUCCESSFUL")
    print("=" * 80)


if __name__ == "__main__":
    main()