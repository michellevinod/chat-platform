import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.chunking.chunk_models import DocumentChunk
from app.rag.retrieved_chunk import RetrievedChunk


class QdrantRepository:
    """
    Handles all communication with Qdrant vector database.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
    ) -> None:
        qdrant_url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY") or None

        if qdrant_url:
            self._client = QdrantClient(
                url=qdrant_url,
                api_key=api_key,
            )
        else:
            self._client = QdrantClient(
                host=host,
                port=port,
                api_key=api_key,
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
            meta = chunk.metadata
            payload = {
                "text": chunk.text,
                "project_id": meta.project_id,
                "project_name": meta.project_name,
                "document_id": meta.document_id,
                "document_name": meta.document_name,
                "document_type": getattr(meta, "document_type", "pdf"),
                "page_number": meta.page_number,
                "chunk_number": meta.chunk_number,
                "heading": getattr(meta, "heading", None),
                "section": getattr(meta, "section", None),
                "chunk_type": getattr(meta, "chunk_type", "text"),
                "table_id": getattr(meta, "table_id", None),
                "image_id": getattr(meta, "image_id", None),
                "image_path": getattr(meta, "image_path", None),
                "source": getattr(meta, "source", "upload"),
                "chunk_key": f"{meta.project_id}:{meta.document_id}:{meta.page_number}:{meta.chunk_number}",
            }

            points.append(
                PointStruct(
                    id=chunk.id,
                    vector=chunk.embedding,
                    payload=payload,
                )
            )

        self._client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 8,
        project_name: str | None = None,
        document_name: str | None = None,
        project_id: str | None = None,
        document_id: str | None = None,
        chunk_type: str | None = None,
    ) -> list[RetrievedChunk]:

        must_conditions = []

        if project_name:
            must_conditions.append(
                FieldCondition(
                    key="project_name",
                    match=MatchValue(value=project_name),
                )
            )
        elif project_id:
            must_conditions.append(
                FieldCondition(
                    key="project_id",
                    match=MatchValue(value=project_id),
                )
            )

        if document_name:
            must_conditions.append(
                FieldCondition(
                    key="document_name",
                    match=MatchValue(value=document_name),
                )
            )
        elif document_id:
            must_conditions.append(
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            )

        if chunk_type:
            must_conditions.append(
                FieldCondition(
                    key="chunk_type",
                    match=MatchValue(value=chunk_type),
                )
            )

        query_filter = Filter(must=must_conditions) if must_conditions else None

        response = self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        results = []

        for point in response.points:
            payload = point.payload or {}

            results.append(
                RetrievedChunk(
                    text=payload.get("text", ""),
                    score=point.score,
                    project_id=payload.get("project_id", "project_001"),
                    project_name=payload.get("project_name", "Default Project"),
                    document_id=payload.get("document_id", ""),
                    document_name=payload.get("document_name", "Unknown Document"),
                    document_type=payload.get("document_type", "pdf"),
                    page_number=payload.get("page_number", 1),
                    chunk_number=payload.get("chunk_number", 0),
                    heading=payload.get("heading"),
                    section=payload.get("section"),
                    chunk_type=payload.get("chunk_type", "text"),
                    table_id=payload.get("table_id"),
                    image_id=payload.get("image_id"),
                    image_path=payload.get("image_path"),
                    source=payload.get("source", "upload"),
                )
            )

        return results

    def get_distinct_documents(
        self,
        collection_name: str = "documents",
        project_name: str | None = None,
    ) -> list[str]:
        try:
            must_conditions = []
            if project_name:
                must_conditions.append(
                    FieldCondition(
                        key="project_name",
                        match=MatchValue(value=project_name),
                    )
                )
            scroll_filter = Filter(must=must_conditions) if must_conditions else None
            points, _ = self._client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=1000,
                with_payload=True,
            )
            doc_names = sorted(list({p.payload.get("document_name") for p in points if p.payload and p.payload.get("document_name")}))
            return doc_names
        except Exception:
            return []

    def get_distinct_projects(
        self,
        collection_name: str = "documents",
    ) -> list[str]:
        try:
            points, _ = self._client.scroll(
                collection_name=collection_name,
                limit=1000,
                with_payload=True,
            )
            proj_names = sorted(list({p.payload.get("project_name") for p in points if p.payload and p.payload.get("project_name")}))
            return proj_names
        except Exception:
            return []