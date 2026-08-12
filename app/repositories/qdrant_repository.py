from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from app.chunking.chunk_models import DocumentChunk


class QdrantRepository:
    """
    Handles all communication with Qdrant.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
    ) -> None:

        self._client = QdrantClient(
            host=host,
            port=port,
        )

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
    ) -> None:

        collections = (
            self._client.get_collections()
            .collections
        )

        existing = [
            collection.name
            for collection in collections
        ]

        if collection_name in existing:
            return

        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    def upsert_chunks(
        self,
        collection_name: str,
        chunks: list[DocumentChunk],
    ) -> None:

        points = []

        for chunk in chunks:

            points.append(
                PointStruct(
                    id=chunk.id,
                    vector=chunk.embedding,
                    payload={
                        "text": chunk.text,

                        "project_id": (
                            chunk.metadata.project_id
                        ),
                        "project_name": (
                            chunk.metadata.project_name
                        ),

                        "document_id": (
                            chunk.metadata.document_id
                        ),
                        "document_name": (
                            chunk.metadata.document_name
                        ),

                        "page_number": (
                            chunk.metadata.page_number
                        ),
                        "chunk_number": (
                            chunk.metadata.chunk_number
                        ),
                        "chunk_key": (
                            f"{chunk.metadata.project_id}:"
                            f"{chunk.metadata.document_id}:"
                            f"{chunk.metadata.page_number}:"
                            f"{chunk.metadata.chunk_number}"
                        ),
                        "source": (
                            chunk.metadata.source
                        ),
                    },
                )
            )

        self._client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )