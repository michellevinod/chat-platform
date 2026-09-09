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

        qdrant_url = os.getenv("QDRANT_URL")

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
                "table_number": meta.table_number,
                "table_caption": meta.table_caption,
                "table_headers": meta.table_headers,
                "table_rows": meta.table_rows,

                # Image metadata
                "image_id": meta.image_id,
                "image_path": meta.image_path,
                "image_number": meta.image_number,
                "image_caption": meta.image_caption,

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
    # PROJECT RENAME
    # ================================================================

    def rename_project(
        self,
        old_name: str,
        new_name: str,
        collection_name: str | None = None,
    ) -> None:

        collection = (
            collection_name
            or os.getenv(
                "QDRANT_COLLECTION",
                "documents",
            )
        )

        project_filter = Filter(
            must=[
                FieldCondition(
                    key="project_name",
                    match=MatchValue(
                        value=old_name,
                    ),
                )
            ]
        )

        self._client.set_payload(
            collection_name=collection,
            payload={
                "project_name": new_name,
            },
            points=project_filter,
            wait=True,
        )

    # ================================================================
    # PROJECT DELETE
    # ================================================================

    def delete_project(
        self,
        project_name: str,
        collection_name: str | None = None,
    ) -> None:

        collection = (
            collection_name
            or os.getenv(
                "QDRANT_COLLECTION",
                "documents",
            )
        )

        project_filter = Filter(
            must=[
                FieldCondition(
                    key="project_name",
                    match=MatchValue(
                        value=project_name,
                    ),
                )
            ]
        )

        self._client.delete(
            collection_name=collection,
            points_selector=project_filter,
            wait=True,
        )

    # ================================================================
    # SEMANTIC SEARCH
    # ================================================================

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        query_text: str | None = None,
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
            limit=max(limit * 4, 32),
            with_payload=True,
        )

        semantic = self._convert_results(response.points)
        candidates = {self._result_key(chunk): chunk for chunk in semantic}

        if query_text:
            lexical_points: list = []
            offset = None
            while True:
                points, next_offset = self._client.scroll(
                    collection_name=collection_name,
                    scroll_filter=query_filter,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                lexical_points.extend(points)
                if next_offset is None:
                    break
                offset = next_offset

            terms = self._query_terms(query_text)
            for chunk in self._convert_results(lexical_points):
                searchable = " ".join(
                    str(getattr(chunk, field, "") or "")
                    for field in (
                        "text", "heading", "section", "table_caption",
                        "image_caption", "table_headers",
                    )
                ).lower()
                tokens = set(re.findall(r"[a-z0-9]+", searchable))
                overlap = terms.intersection(tokens)
                if not overlap:
                    continue
                lexical_score = min(
                    0.98,
                    0.55 + 0.35 * len(overlap) / max(len(terms), 1),
                )
                key = self._result_key(chunk)
                if key not in candidates or lexical_score > candidates[key].score:
                    chunk.score = lexical_score
                    candidates[key] = chunk

        ranked = sorted(candidates.values(), key=lambda item: item.score, reverse=True)
        if chunk_type in {"image", "table"}:
            return ranked[:max(limit, 1)]
        return self.expand_context(
            collection_name=collection_name,
            seeds=ranked[:max(limit, 12)],
            limit=max(limit * 4, 24),
            project_name=project_name,
            document_name=document_name,
            chunk_type=chunk_type,
        )

    @staticmethod
    def _query_terms(query: str) -> set[str]:
        stop_words = {
            "what", "which", "where", "when", "why", "how", "does", "did",
            "is", "are", "the", "a", "an", "this", "that", "document", "say",
            "about", "show", "me", "tell", "please", "on", "in", "of", "for",
            "to", "and", "with", "all", "list", "give", "her", "his", "make",
            "use", "need", "want",
        }
        return {
            token for token in re.findall(r"[a-z0-9]+", query.lower())
            if len(token) >= 3 and token not in stop_words and not token.isdigit()
        }

    @staticmethod
    def _result_key(chunk: RetrievedChunk) -> tuple[str, int, int, str]:
        return (
            chunk.document_name,
            chunk.page_number,
            chunk.chunk_number,
            chunk.chunk_type,
        )

    def expand_context(
        self,
        collection_name: str,
        seeds: list[RetrievedChunk],
        limit: int = 24,
        project_name: str | None = None,
        document_name: str | None = None,
        chunk_type: str | None = None,
        window: int = 2,
    ) -> list[RetrievedChunk]:
        """Add bounded reading-order context around relevant chunks."""
        if not seeds:
            return []

        must = []
        if project_name:
            must.append(FieldCondition(key="project_name", match=MatchValue(value=project_name)))
        if document_name:
            must.append(FieldCondition(key="document_name", match=MatchValue(value=document_name)))
        if chunk_type:
            must.append(FieldCondition(key="chunk_type", match=MatchValue(value=chunk_type)))
        scoped_filter = Filter(must=must) if must else None

        points: list = []
        offset = None
        while True:
            batch, next_offset = self._client.scroll(
                collection_name=collection_name,
                scroll_filter=scoped_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points.extend(batch)
            if next_offset is None:
                break
            offset = next_offset

        all_chunks = self._convert_results(points)
        seed_keys = {self._result_key(seed) for seed in seeds}
        by_document = {}
        for chunk in all_chunks:
            by_document.setdefault(chunk.document_name, []).append(chunk)

        selected = {}
        for seed in seeds:
            siblings = sorted(
                by_document.get(seed.document_name, []),
                key=lambda item: (item.page_number, item.chunk_number),
            )
            try:
                index = next(
                    index for index, item in enumerate(siblings)
                    if self._result_key(item) == self._result_key(seed)
                )
            except StopIteration:
                continue
            for item in siblings[max(0, index - window):index + window + 1]:
                key = self._result_key(item)
                if key not in selected or key in seed_keys:
                    if key == self._result_key(seed):
                        item.score = seed.score
                    selected[key] = item

        ordered: list[RetrievedChunk] = []
        seen: set[tuple[str, int, int, str]] = set()
        for seed in seeds:
            siblings = sorted(
                by_document.get(seed.document_name, []),
                key=lambda item: (item.page_number, item.chunk_number),
            )
            try:
                index = next(
                    index for index, item in enumerate(siblings)
                    if self._result_key(item) == self._result_key(seed)
                )
            except StopIteration:
                continue
            nearby = (
                siblings[index:index + window + 1]
                + siblings[max(0, index - window):index]
            )
            for item in nearby:
                key = self._result_key(item)
                if key in selected and key not in seen:
                    ordered.append(selected[key])
                    seen.add(key)
                if len(ordered) >= limit:
                    return ordered
        return ordered

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
        image_number: str | None = None,
        limit: int = 10,
    ) -> list[RetrievedChunk]:
        """
        Retrieve image chunks using exact metadata.

        No embedding/vector similarity is used.
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

        if image_number is not None:
            must_conditions.append(
                FieldCondition(
                    key="image_number",
                    match=MatchValue(
                        value=str(image_number),
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

    def find_page_chunks(
        self,
        collection_name: str,
        page_number: int,
        project_name: str | None = None,
        document_name: str | None = None,
        limit: int = 100,
    ) -> list[RetrievedChunk]:
        """
        Return all indexed evidence from one page in document order.
        """

        must = [
            FieldCondition(
                key="page_number",
                match=MatchValue(
                    value=page_number,
                ),
            )
        ]

        if project_name:
            must.append(
                FieldCondition(
                    key="project_name",
                    match=MatchValue(
                        value=project_name,
                    ),
                )
            )

        if document_name:
            must.append(
                FieldCondition(
                    key="document_name",
                    match=MatchValue(
                        value=document_name,
                    ),
                )
            )

        points, _ = self._client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=must
            ),
            limit=limit,
            with_payload=True,
        )

        return sorted(
            self._convert_results(points),
            key=lambda chunk: chunk.chunk_number,
        )

    # ================================================================
    # REPRESENTATIVE DOCUMENT RETRIEVAL
    # ================================================================

    def get_representative_document_chunks(
        self,
        collection_name: str,
        document_name: str,
        project_name: str | None = None,
        limit: int = 16,
        scan_limit: int = 1000,
    ) -> list[RetrievedChunk]:
        """
        Retrieve representative text/OCR evidence from an entire document.

        Qdrant scrolling is paginated so documents larger than one Qdrant
        scroll batch are handled correctly.

        Only text and OCR chunks are used for document-level summaries.

        The final evidence is spread across the document so that summaries
        are not biased toward the beginning of the document.
        """

        if not document_name or limit <= 0:
            return []

        must = [
            FieldCondition(
                key="document_name",
                match=MatchValue(
                    value=document_name,
                ),
            )
        ]

        if project_name:
            must.append(
                FieldCondition(
                    key="project_name",
                    match=MatchValue(
                        value=project_name,
                    ),
                )
            )

        query_filter = Filter(
            must=must
        )

        all_chunks: list[RetrievedChunk] = []

        offset = None

        while True:

            points, next_offset = self._client.scroll(
                collection_name=collection_name,
                scroll_filter=query_filter,
                limit=scan_limit,
                offset=offset,
                with_payload=True,
            )

            if not points:
                break

            converted = self._convert_results(
                points
            )

            for chunk in converted:

                chunk_type = (
                    getattr(
                        chunk,
                        "chunk_type",
                        None,
                    )
                    or "text"
                ).lower()

                text = (
                    getattr(
                        chunk,
                        "text",
                        None,
                    )
                    or ""
                ).strip()

                if (
                    chunk_type in {"text", "ocr"}
                    and text
                ):
                    all_chunks.append(chunk)

            if next_offset is None:
                break

            offset = next_offset

        if not all_chunks:
            return []

        # Keep document reading order.
        all_chunks.sort(
            key=lambda chunk: (
                getattr(
                    chunk,
                    "page_number",
                    0,
                )
                or 0,
                getattr(
                    chunk,
                    "chunk_number",
                    0,
                )
                or 0,
            )
        )

        # Small document: return everything.
        if len(all_chunks) <= limit:
            return all_chunks

        # One requested representative chunk.
        if limit == 1:
            return [all_chunks[0]]

        # Spread selected chunks evenly throughout the document.
        indices = {
            round(
                index
                * (len(all_chunks) - 1)
                / (limit - 1)
            )
            for index in range(limit)
        }

        return [
            chunk
            for index, chunk in enumerate(all_chunks)
            if index in indices
        ]

    # ================================================================
    # TABLE SEARCH
    # ================================================================

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

        Explicit table/page constraints are handled before lightweight
        content matching.
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
            else self._normalize_identifier(
                table_number
            )
        )

        # ------------------------------------------------------------
        # Restrict retrieval to table chunks.
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

        scored_points: list[
            tuple[int, object]
        ] = []

        for point in points:

            payload = point.payload or {}

            text_value = str(
                payload.get("text")
                or ""
            )

            heading_value = str(
                payload.get("heading")
                or ""
            )

            section_value = str(
                payload.get("section")
                or ""
            )

            caption_value = str(
                payload.get("table_caption")
                or ""
            )

            table_id_value = str(
                payload.get("table_id")
                or ""
            )

            table_number_value = str(
                payload.get("table_number")
                or ""
            )

            # If structured numbering exists, an explicit request
            # must not return another numbered table.
            normalized_table_number = (
                self._normalize_identifier(
                    table_number_value
                )
            )

            if (
                requested_table_number
                and table_number_value
                and normalized_table_number
                != requested_table_number
            ):
                continue

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
                    for pattern
                    in explicit_number_patterns
                ):
                    score += 100

                if (
                    normalized_table_number
                    == requested_table_number
                ):
                    score += 150

            # --------------------------------------------------------
            # Exact content matching.
            # --------------------------------------------------------

            for term in query_terms:

                if term in searchable_text:
                    score += 2

            # Exact phrase matching.
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

            meaningful_terms = [
                term
                for term in query_terms
                if not term.isdigit()
            ]

            if len(meaningful_terms) >= 2:

                phrase = " ".join(
                    meaningful_terms
                )

                if phrase in normalized_text:
                    score += 20

            # Page is already an exact Qdrant filter.
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
        # "table on page X"
        # ------------------------------------------------------------

        if (
            page_number is not None
            and not query_terms
        ):
            return self._convert_results(
                points[:limit]
            )

        scored_points.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        selected_points = [
            point
            for _, point
            in scored_points[:limit]
        ]

        return self._convert_results(
            selected_points
        )

    @staticmethod
    def _normalize_identifier(
        value: object,
    ) -> str:
        """
        Normalize plain and labelled numeric metadata.
        """

        match = re.search(
            r"\d+",
            str(value or ""),
        )

        return (
            match.group(0)
            if match
            else ""
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

            # A result must contain the core metadata required by
            # RetrievedChunk.
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

                    table_number=payload.get(
                        "table_number"
                    ),

                    table_caption=payload.get(
                        "table_caption"
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

                    image_number=payload.get(
                        "image_number"
                    ),

                    image_caption=payload.get(
                        "image_caption"
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