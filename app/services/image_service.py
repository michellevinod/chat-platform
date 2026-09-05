import re

from app.rag.rag_tool import RAGTool
from app.rag.retrieved_chunk import RetrievedChunk
from app.repositories.qdrant_repository import QdrantRepository


class ImageService:
    """
    Retrieves document images.

    Retrieval strategy:

    1. Exact image filename -> exact Qdrant metadata lookup.
    2. Explicit page -> exact page metadata lookup.
    3. Figure/image description -> semantic image search.

    LLMs (Large Language Models) are not required for these
    retrieval operations.
    """

    IMAGE_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".gif",
        ".bmp",
    }

    def __init__(self) -> None:
        self._rag = RAGTool()
        self._repository = QdrantRepository()

    def get_images(
        self,
        query: str,
        project_name: str | None = None,
        document_name: str | None = None,
        limit: int = 5,
        page_number: int | None = None,
    ) -> list[RetrievedChunk]:

        image_id = self._extract_image_filename(
            query
        )
        image_number = self._extract_figure_number(query)

        # ---------------------------------------------------------
        # 1. EXACT IMAGE FILENAME
        # ---------------------------------------------------------

        if image_id:
            return self._repository.find_images(
                collection_name="documents",
                project_name=project_name,
                document_name=document_name,
                image_id=image_id,
                limit=limit,
            )

        # Figure numbers are explicit image metadata, not page numbers.
        if image_number is not None:
            images = self._repository.find_images(
                collection_name="documents",
                project_name=project_name,
                document_name=document_name,
                image_number=image_number,
                limit=limit,
            )

            if images:
                return images

            # Older records may have labelled values such as "Figure 6".
            # Inspect only scoped image records and normalize locally.
            legacy_images = self._repository.find_images(
                collection_name="documents",
                project_name=project_name,
                document_name=document_name,
                limit=1000,
            )
            matching_images = [
                image for image in legacy_images
                if self._normalize_identifier(getattr(image, "image_number", None)) == image_number
            ]
            if matching_images:
                return matching_images[:limit]

            ordered_images = sorted(
                legacy_images,
                key=lambda image: (
                    getattr(image, "page_number", 0),
                    getattr(image, "chunk_number", 0),
                ),
            )
            ordinal = int(image_number) - 1
            if 0 <= ordinal < len(ordered_images):
                return [ordered_images[ordinal]]
            return []

        # ---------------------------------------------------------
        # 2. EXACT PAGE
        # ---------------------------------------------------------

        # Prefer the explicitly supplied page number from
        # ChatAgent. Fall back to extracting it here so this
        # service remains safe when called directly.
        if page_number is None:
            page_number = self._extract_page_number(
                query
            )

        if page_number is not None:
            return self._repository.find_images(
                collection_name="documents",
                project_name=project_name,
                document_name=document_name,
                page_number=page_number,
                limit=limit,
            )

        # ---------------------------------------------------------
        # 3. SEMANTIC IMAGE SEARCH
        # ---------------------------------------------------------

        return self._rag.search(
            query=query,
            limit=limit,
            project_name=project_name,
            document_name=document_name,
            chunk_type="image",
        )

    @classmethod
    def _extract_image_filename(
        cls,
        query: str,
    ) -> str | None:
        """
        Extract an image filename from the user query.
        """

        extension_pattern = (
            r"([a-zA-Z0-9_.-]+"
            r"\.(?:png|jpg|jpeg|webp|gif|bmp))"
        )

        match = re.search(
            extension_pattern,
            query,
            flags=re.IGNORECASE,
        )

        if match:
            return match.group(1)

        generated_id_pattern = (
            r"\b(img[_-][a-zA-Z0-9_-]+)\b"
        )

        match = re.search(
            generated_id_pattern,
            query,
            flags=re.IGNORECASE,
        )

        if match:
            return match.group(1)

        return None

    @staticmethod
    def _extract_figure_number(
        query: str,
    ) -> str | None:
        match = re.search(
            r"\b(?:figure|fig\.?|image)\s*(?:number|no\.?)?\s*[:#-]?\s*(\d+)\b",
            query,
            flags=re.IGNORECASE,
        )
        return match.group(1) if match else None

    @staticmethod
    def _normalize_identifier(value: object) -> str:
        match = re.search(r"\d+", str(value or ""))
        return match.group(0) if match else ""

    @staticmethod
    def _extract_page_number(
        query: str,
    ) -> int | None:
        """
        Extract an explicit page number.

        Examples:

            page 44
            page 18
            pg 12
            p. 37
        """

        match = re.search(
            r"\b(?:page|pages|pg|p\.)\s*(\d+)\b",
            query.lower(),
        )

        if not match:
            return None

        return int(
            match.group(1)
        )
