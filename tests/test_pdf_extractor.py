from pathlib import Path

from app.ingestion.extractors.pdf_extractor import PDFExtractor
from app.ingestion.normalizers.pdf_normalizer import PDFNormalizer


"""
def main():

    extractor = PDFExtractor()

    document = extractor.extract(
        Path("storage/uploads/PDF_BenchmarkTester_63Pages.pdf")
    )

    normalizer = PDFNormalizer()

    document = normalizer.normalize(document)


    from app.ingestion.metadata.metadata_generator import MetadataGenerator

    metadata_generator = MetadataGenerator()

    document = metadata_generator.generate(
        document=document,
        project_name="Forge Project",
        project_id="project_001",
        document_id="doc_001",
    )

    print(document.metadata)



    print("=" * 80)
    print(f"Document : {document.file_name}")
    print(f"Pages    : {document.total_pages}")
    print("=" * 80)

    first_page = document.pages[0]

    print(f"Page Number : {first_page.page_number}")
    print(f"Blocks Found: {len(first_page.blocks)}")

    print()

    for block in first_page.blocks:

        print("=" * 50)
        print(f"Block Number : {block.block_number}")
        print(f"Block Type   : {block.block_type}")
        print(f"BBox         : {block.bbox}")
        print("-" * 50)
        print(block.text)
        print()


if __name__ == "__main__":
    main()
"""



from pathlib import Path

from app.chunking.chunk_generator import ChunkGenerator
from app.ingestion.extractors.pdf_extractor import PDFExtractor
from app.ingestion.metadata.metadata_generator import MetadataGenerator
from app.ingestion.normalizers.pdf_normalizer import PDFNormalizer


def main():

    extractor = PDFExtractor()

    document = extractor.extract(
        Path("storage/uploads/PDF_BenchmarkTester_63Pages.pdf")
    )

    normalizer = PDFNormalizer()
    document = normalizer.normalize(document)

    metadata_generator = MetadataGenerator()

    document = metadata_generator.generate(
        document=document,
        project_name="Forge Project",
        project_id="project_001",
        document_id="doc_001",
    )

    generator = ChunkGenerator()

    chunks = generator.generate(document)

    print("=" * 80)
    print(f"Chunks Generated: {len(chunks)}")
    print("=" * 80)

    for chunk in chunks[:5]:
        print(chunk.id)
        print(chunk.metadata)
        print(chunk.text)
        print()


    from app.embeddings.embedding_pipeline import EmbeddingPipeline

    embedding_pipeline = EmbeddingPipeline()

    chunks = embedding_pipeline.generate(chunks)

    print("=" * 80)
    print("First Chunk")
    print("=" * 80)

    print(chunks[0].text[:100])
    print()
    print(len(chunks[0].embedding))


if __name__ == "__main__":
    main()