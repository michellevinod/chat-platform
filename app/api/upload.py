from pathlib import Path
import shutil
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.chunking.chunk_generator import ChunkGenerator
from app.embeddings.embedding_pipeline import EmbeddingPipeline
from app.ingestion.extractors.pdf_extractor import PDFExtractor
from app.ingestion.extractors.docx_extractor import DOCXExtractor
from app.ingestion.extractors.ppt_extractor import PPTXExtractor
from app.ingestion.extractors.xlsx_extractor import XLSXExtractor
from app.ingestion.normalizers.pdf_normalizer import PDFNormalizer
from app.repositories.qdrant_repository import QdrantRepository

router = APIRouter(
    prefix="/upload",
    tags=["Upload"],
)

UPLOAD_DIR = Path("storage/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("")
async def upload_document(
    file: UploadFile = File(...)
):
    extension = Path(file.filename).suffix.lower()

    if extension not in [
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
    ]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported document type.",
        )

    unique_name = f"{uuid.uuid4()}{extension}"
    saved_path = UPLOAD_DIR / unique_name

    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract document
    if extension == ".pdf":
        extractor = PDFExtractor()

    elif extension == ".docx":
        extractor = DOCXExtractor()

    elif extension == ".pptx":
        extractor = PPTXExtractor()

    elif extension == ".xlsx":
        extractor = XLSXExtractor()

    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported document type.",
        )

    document = extractor.extract(saved_path)

    # Normalize
    document = PDFNormalizer().normalize(document)

    # Generate chunks
    chunks = ChunkGenerator().generate(document)

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No content could be extracted from the uploaded document.",
        )

    # Generate embeddings
    chunks = EmbeddingPipeline().generate(chunks)

    repository = QdrantRepository()

    collection_name = "documents"

    # Create collection if it doesn't exist
    repository.create_collection(
        collection_name=collection_name,
        vector_size=len(chunks[0].embedding),
    )

    # Store chunks
    repository.upsert_chunks(
        collection_name=collection_name,
        chunks=chunks,
    )

    return {
        "success": True,
        "document": file.filename,
        "chunks_uploaded": len(chunks),
    }