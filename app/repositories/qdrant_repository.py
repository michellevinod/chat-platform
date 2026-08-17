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
    Handles all communication with Qdrant.

    Qdrant stores document chunks together with their metadata.
    Project, document, and file-type information is always supplied
    by the ingestion pipeline and is never hardcoded here.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
    ) -> None:

        qdrant_url = os.getenv(
            "QDRANT_URL"
        )

        api_key = (
            os.getenv("QDRANT_API_KEY")
            or None
        )

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
        """
        Create a Qdrant collection if it does not already exist.
        """

        collections = (
            self._client.get_collections()
            .collections
        )

        existing = {
            collection.name
            for collection in collections
        }

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
        """
        Store document chunks and their complete metadata in Qdrant.

        All document/project metadata comes from the chunk itself.
        Nothing document-specific is hardcoded here.
        """

        points: list[PointStruct] = []

        for chunk in chunks:

            meta = chunk.metadata

            payload = {
                "text": chunk.text,

                "project_id": meta.project_id,
                "project_name": meta.project_name,

                "document_id": meta.document_id,
                "document_name": meta.document_name,
                "document_type": meta.document_type,

                "page_number": meta.page_number,
                "chunk_number": meta.chunk_number,

                "heading": meta.heading,
                "section": meta.section,

                "chunk_type": meta.chunk_type,

                "table_id": meta.table_id,
                "table_headers": meta.table_headers,
                "table_rows": meta.table_rows,

                "image_id": meta.image_id,
                "image_path": meta.image_path,

                "source": meta.source,

                # Internal retrieval key.
                # Never expose this value to the user.
                "chunk_key": (
                    f"{meta.project_id}:"
                    f"{meta.document_id}:"
                    f"{meta.page_number}:"
                    f"{meta.chunk_number}"
                ),
            }

            points.append(
                PointStruct(
                    id=chunk.id,
                    vector=chunk.embedding,
                    payload=payload,
                )
            )

        if not points:
            return

        self._client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )

    def delete_document(
        self,
        collection_name: str,
        project_name: str,
        document_name: str,
    ) -> None:
        """
        Delete all chunks belonging to one document within one project.

        This is intentionally scoped by both project_name and
        document_name so that documents with the same name in
        different projects are not affected.
        """

        document_filter = Filter(
            must=[
                FieldCondition(
                    key="project_name",
                    match=MatchValue(
                        value=project_name,
                    ),
                ),
                FieldCondition(
                    key="document_name",
                    match=MatchValue(
                        value=document_name,
                    ),
                ),
            ]
        )

        self._client.delete(
            collection_name=collection_name,
            points_selector=document_filter,
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
        """
        Search Qdrant using vector similarity and optional metadata filters.

        Filtering is dynamic and can be applied by:
        - project name
        - project ID
        - document name
        - document ID
        - chunk type
        """

        must_conditions: list[FieldCondition] = []

        if project_name:

            must_conditions.append(
                FieldCondition(
                    key="project_name",
                    match=MatchValue(
                        value=project_name,
                    ),
                )
            )

        elif project_id:

            must_conditions.append(
                FieldCondition(
                    key="project_id",
                    match=MatchValue(
                        value=project_id,
                    ),
                )
            )

        if document_name:

            must_conditions.append(
                FieldCondition(
                    key="document_name",
                    match=MatchValue(
                        value=document_name,
                    ),
                )
            )

        elif document_id:

            must_conditions.append(
                FieldCondition(
                    key="document_id",
                    match=MatchValue(
                        value=document_id,
                    ),
                )
            )

        if chunk_type:

            must_conditions.append(
                FieldCondition(
                    key="chunk_type",
                    match=MatchValue(
                        value=chunk_type,
                    ),
                )
            )

        query_filter = (
            Filter(
                must=must_conditions
            )
            if must_conditions
            else None
        )

        response = self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        results: list[RetrievedChunk] = []

        for point in response.points:

            payload = point.payload or {}

            # These are required document metadata.
            # Do not invent fallback values if they are missing.
            project_id_value = payload.get(
                "project_id"
            )

            project_name_value = payload.get(
                "project_name"
            )

            document_id_value = payload.get(
                "document_id"
            )

            document_name_value = payload.get(
                "document_name"
            )

            document_type_value = payload.get(
                "document_type"
            )

            if not all(
                [
                    project_id_value,
                    project_name_value,
                    document_id_value,
                    document_name_value,
                    document_type_value,
                ]
            ):
                continue

            results.append(
                RetrievedChunk(
                    text=payload.get(
                        "text",
                        "",
                    ),
                    score=point.score,

                    project_id=project_id_value,
                    project_name=project_name_value,

                    document_id=document_id_value,
                    document_name=document_name_value,
                    document_type=document_type_value,

                    page_number=payload.get(
                        "page_number",
                        0,
                    ),
                    chunk_number=payload.get(
                        "chunk_number",
                        0,
                    ),

                    heading=payload.get(
                        "heading"
                    ),
                    section=payload.get(
                        "section"
                    ),

                    chunk_type=payload.get(
                        "chunk_type",
                        "text",
                    ),

                    table_id=payload.get(
                        "table_id"
                    ),
                    table_headers=payload.get(
                        "table_headers",
                        [],
                    ),
                    table_rows=payload.get(
                        "table_rows",
                        [],
                    ),

                    image_id=payload.get(
                        "image_id"
                    ),
                    image_path=payload.get(
                        "image_path"
                    ),

                    source=payload.get(
                        "source",
                        "upload",
                    ),
                )
            )

        return results

    def get_distinct_documents(
        self,
        collection_name: str | None = None,
        project_name: str | None = None,
    ) -> list[str]:
        """
        Return distinct document names, optionally scoped to a project.
        """

        collection = (
            collection_name
            or os.getenv(
                "QDRANT_COLLECTION",
                "documents",
            )
        )

        try:

            must_conditions: list[
                FieldCondition
            ] = []

            if project_name:

                must_conditions.append(
                    FieldCondition(
                        key="project_name",
                        match=MatchValue(
                            value=project_name,
                        ),
                    )
                )

            scroll_filter = (
                Filter(
                    must=must_conditions
                )
                if must_conditions
                else None
            )

            points, _ = self._client.scroll(
                collection_name=collection,
                scroll_filter=scroll_filter,
                limit=1000,
                with_payload=True,
            )

            documents = {
                point.payload.get(
                    "document_name"
                )
                for point in points
                if point.payload
                and point.payload.get(
                    "document_name"
                )
            }

            return sorted(
                documents
            )

        except Exception:
            return []

    def get_distinct_projects(
        self,
        collection_name: str | None = None,
    ) -> list[str]:
        """
        Return distinct project names stored in Qdrant.
        """

        collection = (
            collection_name
            or os.getenv(
                "QDRANT_COLLECTION",
                "documents",
            )
        )

        try:

            points, _ = self._client.scroll(
                collection_name=collection,
                limit=1000,
                with_payload=True,
            )

            projects = {
                point.payload.get(
                    "project_name"
                )
                for point in points
                if point.payload
                and point.payload.get(
                    "project_name"
                )
            }

            return sorted(
                projects
            )

        except Exception:
            return []