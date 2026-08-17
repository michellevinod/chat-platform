import os
import re

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

    Project, document, file type, image, and table metadata are
    supplied dynamically by the ingestion pipeline.
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

    # ================================================================
    # COLLECTION
    # ================================================================

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
    ) -> None:

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

    # ================================================================
    # INGESTION
    # ================================================================

    def upsert_chunks(
        self,
        collection_name: str,
        chunks: list[DocumentChunk],
    ) -> None:

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

                # Table metadata
                "table_id": meta.table_id,
                "table_headers": meta.table_headers,
                "table_rows": meta.table_rows,
                "caption": getattr(meta, "caption", None),
                "table_number": getattr(meta, "table_number", None),

                # Image metadata
                "image_id": meta.image_id,
                "image_path": meta.image_path,

                "source": meta.source,

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

    # ================================================================
    # DOCUMENT DELETE
    # ================================================================

    def delete_document(
        self,
        collection_name: str,
        project_name: str,
        document_name: str,
    ) -> None:

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

    # ================================================================
    # SEMANTIC SEARCH
    # ================================================================

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
        page_number: int | None = None,
        image_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Perform vector similarity search with optional metadata filters.

        Metadata filters are applied BEFORE semantic ranking.

        This allows queries such as:

            image on page 44
            image_id = img_xxx.png
            tables from a specific document
        """

        must_conditions: list[FieldCondition] = []

        # ------------------------------------------------------------
        # Project
        # ------------------------------------------------------------

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

        # ------------------------------------------------------------
        # Document
        # ------------------------------------------------------------

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

        # ------------------------------------------------------------
        # Chunk type
        # ------------------------------------------------------------

        if chunk_type:

            must_conditions.append(
                FieldCondition(
                    key="chunk_type",
                    match=MatchValue(
                        value=chunk_type,
                    ),
                )
            )

        # ------------------------------------------------------------
        # Exact page
        # ------------------------------------------------------------

        if page_number is not None:

            must_conditions.append(
                FieldCondition(
                    key="page_number",
                    match=MatchValue(
                        value=page_number,
                    ),
                )
            )

        # ------------------------------------------------------------
        # Exact image
        # ------------------------------------------------------------

        if image_id:

            must_conditions.append(
                FieldCondition(
                    key="image_id",
                    match=MatchValue(
                        value=image_id,
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

        return self._convert_results(
            response.points
        )

    # ================================================================
    # EXACT METADATA SEARCH
    # ================================================================

    def find_images(
        self,
        collection_name: str,
        project_name: str | None = None,
        document_name: str | None = None,
        page_number: int | None = None,
        image_id: str | None = None,
        limit: int = 10,
    ) -> list[RetrievedChunk]:
        """
        Retrieve image chunks using exact metadata.

        No embedding/vector similarity is used.

        This is the correct path for:
            - exact image filename
            - exact page image
        """

        must_conditions: list[FieldCondition] = [
            FieldCondition(
                key="chunk_type",
                match=MatchValue(
                    value="image",
                ),
            )
        ]

        if project_name:

            must_conditions.append(
                FieldCondition(
                    key="project_name",
                    match=MatchValue(
                        value=project_name,
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

        if page_number is not None:

            must_conditions.append(
                FieldCondition(
                    key="page_number",
                    match=MatchValue(
                        value=page_number,
                    ),
                )
            )

        if image_id:

            must_conditions.append(
                FieldCondition(
                    key="image_id",
                    match=MatchValue(
                        value=image_id,
                    ),
                )
            )

        query_filter = Filter(
            must=must_conditions
        )

        points, _ = self._client.scroll(
            collection_name=collection_name,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        return self._convert_results(
            points
        )


    def find_tables_by_query(
        self,
        collection_name: str,
        query: str,
        project_name: str | None = None,
        document_name: str | None = None,
        table_number: str | None = None,
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Retrieve table chunks using deterministic metadata + lexical matching.

        This is intentionally separate from vector similarity search because
        table requests such as:

            Show me Table 1. Harvey 1 well data record
            Show me Table 4. Summary result of MICP tests for Harvey 1
            Show me Table on page 24

        are better handled by exact table/page constraints followed by
        lightweight content matching.

        Existing indexed documents are supported even when their payload does
        not contain the optional caption/table_number fields.
        """

        lowered_query = query.lower()

        # ------------------------------------------------------------
        # Extract explicit page number.
        # ------------------------------------------------------------

        page_match = re.search(
            r"\b(?:page|pages|pg|p\.)\s*(\d+)\b",
            lowered_query,
        )
        page_number = (
            int(page_match.group(1))
            if page_match
            else None
        )

        # ------------------------------------------------------------
        # Extract explicit table number.
        # ------------------------------------------------------------

        table_match = re.search(
            r"\btable\s*(\d+)\b",
            lowered_query,
        )
        requested_table_number = (
            table_match.group(1)
            if table_match
            else None
        )

        # ------------------------------------------------------------
        # Restrict retrieval to table chunks belonging to the requested
        # project/document/page.
        # ------------------------------------------------------------

        must_conditions: list[FieldCondition] = [
            FieldCondition(
                key="chunk_type",
                match=MatchValue(
                    value="table",
                ),
            )
        ]

        if project_name:
            must_conditions.append(
                FieldCondition(
                    key="project_name",
                    match=MatchValue(
                        value=project_name,
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

        if page_number is not None:
            must_conditions.append(
                FieldCondition(
                    key="page_number",
                    match=MatchValue(
                        value=page_number,
                    ),
                )
            )

        query_filter = Filter(
            must=must_conditions
        )

        # We deliberately use scroll here instead of vector search.
        points, _ = self._client.scroll(
            collection_name=collection_name,
            scroll_filter=query_filter,
            limit=1000,
            with_payload=True,
        )

        if not points:
            return []

        # ------------------------------------------------------------
        # Normalize query terms.
        # ------------------------------------------------------------

        stop_words = {
            "show",
            "me",
            "the",
            "a",
            "an",
            "table",
            "tables",
            "on",
            "page",
            "pages",
            "pg",
            "from",
            "of",
            "for",
            "please",
            "give",
            "display",
            "get",
            "find",
            "this",
            "that",
        }

        query_terms = {
            token
            for token in re.findall(
                r"[a-z0-9]+",
                lowered_query,
            )
            if len(token) >= 3
            and token not in stop_words
        }

        # Terms that are particularly useful for identifying the requested
        # table. Give these a larger weight than generic matching.
        important_terms = {
            "harvey",
            "micp",
            "summary",
            "result",
            "results",
            "well",
            "data",
            "record",
        }

        scored_points: list[tuple[int, object]] = []

        for point in points:
            payload = point.payload or {}

            text_value = str(
                payload.get("text") or ""
            )

            heading_value = str(
                payload.get("heading") or ""
            )

            section_value = str(
                payload.get("section") or ""
            )

            caption_value = str(
                payload.get("caption") or ""
            )

            table_id_value = str(
                payload.get("table_id") or ""
            )

            table_number_value = str(
                payload.get("table_number") or ""
            )

            searchable_text = " ".join(
                [
                    text_value,
                    heading_value,
                    section_value,
                    caption_value,
                    table_id_value,
                    table_number_value,
                ]
            ).lower()

            score = 0

            # --------------------------------------------------------
            # Explicit table number.
            #
            # Prefer a real caption/table_number if present. For older
            # records without those fields, do NOT reject the result;
            # content matching still determines relevance.
            # --------------------------------------------------------

            if requested_table_number:
                explicit_number_patterns = [
                    rf"\btable\s*{re.escape(requested_table_number)}\b",
                    rf"\btable[_\-\s]*{re.escape(requested_table_number)}\b",
                ]

                if any(
                    re.search(
                        pattern,
                        searchable_text,
                    )
                    for pattern in explicit_number_patterns
                ):
                    score += 100

                if (
                    table_number_value
                    == requested_table_number
                ):
                    score += 150

            # --------------------------------------------------------
            # Exact content matching.
            # --------------------------------------------------------

            for term in query_terms:
                if term in searchable_text:
                    if term in important_terms:
                        score += 8
                    else:
                        score += 2

            # Exact phrase matching is a strong signal.
            normalized_query = re.sub(
                r"\s+",
                " ",
                lowered_query,
            ).strip()

            normalized_text = re.sub(
                r"\s+",
                " ",
                searchable_text,
            )

            # Useful phrases from common table queries.
            for phrase in (
                "harvey 1",
                "well data record",
                "summary result",
                "micp tests",
                "summary result of micp tests",
            ):
                if phrase in normalized_query:
                    if phrase in normalized_text:
                        score += 30

            # --------------------------------------------------------
            # Page query is already an exact Qdrant filter, so every
            # returned point is relevant to the requested page.
            # --------------------------------------------------------

            if page_number is not None:
                score += 100

            if score > 0:
                scored_points.append(
                    (
                        score,
                        point,
                    )
                )

        # ------------------------------------------------------------
        # If the query is simply "table on page X", return all tables
        # from that exact page rather than requiring lexical matches.
        # ------------------------------------------------------------

        if page_number is not None and not query_terms:
            return self._convert_results(
                points[:limit]
            )

        scored_points.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        selected_points = [
            point
            for _, point in scored_points[:limit]
        ]

        return self._convert_results(
            selected_points
        )

    # ================================================================
    # RESULT CONVERSION
    # ================================================================

    @staticmethod
    def _convert_results(
        points,
    ) -> list[RetrievedChunk]:

        results: list[RetrievedChunk] = []

        for point in points:

            payload = point.payload or {}

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

            score = getattr(
                point,
                "score",
                0.0,
            )

            results.append(
                RetrievedChunk(
                    text=payload.get(
                        "text",
                        "",
                    ),
                    score=score,

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

    # ================================================================
    # DOCUMENT LISTING
    # ================================================================

    def get_distinct_documents(
        self,
        collection_name: str | None = None,
        project_name: str | None = None,
    ) -> list[str]:

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

    # ================================================================
    # PROJECT LISTING
    # ================================================================

    def get_distinct_projects(
        self,
        collection_name: str | None = None,
    ) -> list[str]:

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

    # ================================================================
    # TABLE SEARCH
    # ================================================================

    def find_tables(
        self,
        collection_name: str,
        project_name: str | None = None,
        document_name: str | None = None,
        page_number: int | None = None,
        limit: int = 10,
    ) -> list[RetrievedChunk]:
        """
        Retrieve table chunks using exact metadata filters.

        No vector similarity is used.
        """

        must_conditions: list[FieldCondition] = [
            FieldCondition(
                key="chunk_type",
                match=MatchValue(
                    value="table",
                ),
            )
        ]

        if project_name:

            must_conditions.append(
                FieldCondition(
                    key="project_name",
                    match=MatchValue(
                        value=project_name,
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

        if page_number is not None:

            must_conditions.append(
                FieldCondition(
                    key="page_number",
                    match=MatchValue(
                        value=page_number,
                    ),
                )
            )

        query_filter = Filter(
            must=must_conditions
        )

        points, _ = self._client.scroll(
            collection_name=collection_name,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        return self._convert_results(
            points
        )