import re

from app.rag.rag_tool import RAGTool
from app.rag.retrieved_chunk import RetrievedChunk
from app.repositories.qdrant_repository import QdrantRepository


class ImageService:
    """
    Retrieves document images.

    Retrieval strategy:

    1. Exact image filename -> exact Qdrant metadata lookup.
    2. Explicit page + image -> exact page metadata lookup.
    3. Figure/image description -> semantic image search.

    LLMs are not required for any of these operations.
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
    ) -> list[RetrievedChunk]:

        image_id = self._extract_image_filename(
            query
        )

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

        # ---------------------------------------------------------
        # 2. EXACT PAGE + IMAGE
        # ---------------------------------------------------------

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

        Examples:

            img_f28db4f9_p37_2.png
            figure_1.jpg
            diagram.webp
        """

        # Match normal filenames containing an image extension.
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

        # Match generated image IDs if the extension
        # has been omitted.
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