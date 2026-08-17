from __future__ import annotations

import re

from app.rag.rag_tool import RAGTool
from app.rag.retrieved_chunk import RetrievedChunk
from app.repositories.qdrant_repository import QdrantRepository


class TableService:
    """
    Retrieves tables from uploaded documents.

    Retrieval strategy:

    1. Explicit page request -> exact table lookup on that page.
    2. Explicit table number/caption -> deterministic metadata/text lookup.
    3. General table description -> semantic table search.

    This prevents explicit requests such as:
        "Show me Table 1..."
    from returning unrelated semantically similar tables.
    """

    def __init__(self) -> None:
        self._rag = RAGTool()
        self._repository = QdrantRepository()

    def get_tables(
        self,
        query: str,
        project_name: str | None = None,
        document_name: str | None = None,
        limit: int = 5,
        page_number: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve tables using deterministic lookup whenever
        the user explicitly identifies a table or page.
        """

        # ---------------------------------------------------------
        # 1. EXPLICIT PAGE
        # ---------------------------------------------------------

        if page_number is None:
            page_number = self._extract_page_number(
                query
            )

        if page_number is not None:
            tables = self._repository.find_tables(
                collection_name="documents",
                project_name=project_name,
                document_name=document_name,
                page_number=page_number,
                limit=limit,
            )

            return tables

        # ---------------------------------------------------------
        # 2. EXPLICIT TABLE NUMBER / CAPTION
        # ---------------------------------------------------------

        table_number = self._extract_table_number(
            query
        )

        if table_number is not None:
            tables = self._repository.find_tables_by_query(
                collection_name="documents",
                project_name=project_name,
                document_name=document_name,
                query=query,
                table_number=table_number,
                limit=limit,
            )

            if tables:
                return tables

        # ---------------------------------------------------------
        # 3. GENERAL TABLE DESCRIPTION
        # ---------------------------------------------------------

        return self._rag.search(
            query=query,
            limit=limit,
            project_name=project_name,
            document_name=document_name,
            chunk_type="table",
        )

    # =============================================================
    # QUERY PARSING
    # =============================================================

    @staticmethod
    def _extract_page_number(
        query: str,
    ) -> int | None:
        """
        Extract explicit page references.

        Examples:
            page 24
            page 33
            pg 12
            p. 18
        """

        match = re.search(
            r"\b(?:page|pages|pg|p\.)\s*(\d+)\b",
            query.lower(),
        )

        if not match:
            return None

        return int(match.group(1))

    @staticmethod
    def _extract_table_number(
        query: str,
    ) -> str | None:
        """
        Extract an explicit table number.

        Examples:
            Table 1
            table 4
            TABLE 12
        """

        match = re.search(
            r"\btable\s+(\d+)\b",
            query,
            flags=re.IGNORECASE,
        )

        if not match:
            return None

        return match.group(1)

    # =============================================================
    # MARKDOWN RENDERING
    # =============================================================

    @staticmethod
    def render_markdown_table(
        chunk: RetrievedChunk,
    ) -> str:
        """
        Convert structured table metadata into Markdown.

        Falls back to the stored text when structured metadata
        is unavailable.
        """

        headers = getattr(
            chunk,
            "table_headers",
            None,
        ) or []

        rows = getattr(
            chunk,
            "table_rows",
            None,
        ) or []

        headers = [
            TableService._clean_cell(
                header
            )
            for header in headers
        ]

        normalized_rows = [
            [
                TableService._clean_cell(
                    cell
                )
                for cell in row
            ]
            for row in rows
            if row
        ]

        normalized_rows = [
            row
            for row in normalized_rows
            if any(
                cell.strip()
                for cell in row
            )
        ]

        if not headers and not normalized_rows:
            return (
                getattr(
                    chunk,
                    "text",
                    "",
                )
                or ""
            ).strip()

        column_count = max(
            [len(headers)]
            + [
                len(row)
                for row in normalized_rows
            ]
        )

        if column_count == 0:
            return ""

        if not headers:
            headers = [
                f"Column {index + 1}"
                for index in range(
                    column_count
                )
            ]

        headers = TableService._fit_row(
            headers,
            column_count,
        )

        lines = [
            "| "
            + " | ".join(headers)
            + " |",
            "| "
            + " | ".join(
                "---"
                for _ in range(column_count)
            )
            + " |",
        ]

        for row in normalized_rows:
            fitted_row = TableService._fit_row(
                row,
                column_count,
            )

            lines.append(
                "| "
                + " | ".join(fitted_row)
                + " |"
            )

        return "\n".join(lines)

    @staticmethod
    def _fit_row(
        row: list[str],
        column_count: int,
    ) -> list[str]:
        if len(row) >= column_count:
            return row[:column_count]

        return row + [
            ""
            for _ in range(
                column_count - len(row)
            )
        ]

    @staticmethod
    def _clean_cell(
        value,
    ) -> str:
        if value is None:
            return ""

        text = " ".join(
            str(value).split()
        )

        return text.replace(
            "|",
            "\\|",
        )