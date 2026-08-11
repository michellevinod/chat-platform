from pathlib import Path

from app.ingestion.extractors.raw_models import RawDocument


class MetadataGenerator:
    """
    Generates metadata for an extracted document.
    """

    def generate(
        self,
        document: RawDocument,
        project_name: str,
        project_id: str,
        document_id: str,
    ) -> RawDocument:

        document.metadata = {
            "project_id": project_id,
            "project_name": project_name,
            "document_id": document_id,
            "document_name": document.file_name,
            "document_type": Path(document.file_name).suffix.replace(".", "").lower(),
            "total_pages": document.total_pages,
        }

        return document