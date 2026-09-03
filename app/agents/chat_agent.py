from __future__ import annotations

import re
from typing import Any

from app.agents.base_agent import BaseAgent
from app.agents.query_classifier import QueryClassifier, QueryIntent
from app.agents.query_enhancer import QueryEnhancer
from app.rag.rag_tool import RAGTool
from app.services.image_service import ImageService
from app.services.table_service import TableService


class ChatAgent(BaseAgent):
    """
    Main document-chat agent.

    Routes user queries to the appropriate retrieval mechanism:
    - images -> ImageService
    - tables -> TableService
    - normal questions -> RAGTool
    - summaries/synthesis -> broad RAG retrieval
    """

    def __init__(
        self,
        rag_tool: RAGTool | None = None,
        image_service: ImageService | None = None,
        table_service: TableService | None = None,
    ) -> None:
        super().__init__()

        self._classifier = QueryClassifier()
        self._enhancer = QueryEnhancer()

        self._rag = rag_tool or RAGTool()
        self._image_service = image_service or ImageService()
        self._table_service = table_service or TableService()

    # =================================================================
    # MAIN EXECUTION
    # =================================================================

    def execute(
        self,
        query: str,
        project_name: str | None = None,
        document_name: str | None = None,
        project_id: str | None = None,
        document_id: str | None = None,
    ) -> dict[str, Any]:

        query = (query or "").strip()

        if not query:
            return {
                "intent": QueryIntent.RAG_FACTUAL,
                "query": "",
                "results": [],
            }

        intent = self._classifier.classify(query)

        # -------------------------------------------------------------
        # GREETING
        # -------------------------------------------------------------

        if intent == QueryIntent.GREETING:
            return {
                "intent": intent,
                "query": query,
                "results": [],
                "direct_response": (
                    "Hello! 👋 Upload one or more documents "
                    "and ask me anything related to them."
                ),
            }

        # -------------------------------------------------------------
        # OUT OF SCOPE
        # -------------------------------------------------------------

        if intent == QueryIntent.OUT_OF_SCOPE:
            return {
                "intent": intent,
                "query": query,
                "results": [],
                "direct_response": (
                    "I can answer questions using the uploaded documents. "
                    "Please ask something related to their content."
                ),
            }

        enhanced_query = self._enhancer.enhance(query)

        page_number = self._extract_page_number(query)
        table_number = self._extract_table_number(query)

        # -------------------------------------------------------------
        # IMAGE
        # -------------------------------------------------------------

        if intent == QueryIntent.SEARCH_IMAGE:

            results = self._search_images(
                query=query,
                project_name=project_name,
                document_name=document_name,
                page_number=page_number,
            )

            return {
                "intent": intent,
                "query": query,
                "results": results,
                "page_number": page_number,
            }

        # -------------------------------------------------------------
        # TABLE
        # -------------------------------------------------------------

        if intent == QueryIntent.SEARCH_TABLE:

            results = self._search_tables(
                query=query,
                project_name=project_name,
                document_name=document_name,
                page_number=page_number,
                table_number=table_number,
            )

            return {
                "intent": intent,
                "query": query,
                "results": results,
                "page_number": page_number,
                "table_number": table_number,
            }

        # -------------------------------------------------------------
        # SUMMARY / PROJECT SUMMARY / AMBIGUOUS SUMMARY
        # -------------------------------------------------------------

        if intent in {
            QueryIntent.DOCUMENT_SUMMARY,
            QueryIntent.PROJECT_SUMMARY,
            QueryIntent.AMBIGUOUS_DOCUMENT,
        }:

            results = self._retrieve_summary_evidence(
                query=query,
                project_name=project_name,
                document_name=document_name,
            )

            return {
                "intent": intent,
                "query": query,
                "results": results,
            }

        # -------------------------------------------------------------
        # NORMAL RAG
        # -------------------------------------------------------------

        results = self._search_rag(
            query=enhanced_query,
            project_name=project_name,
            document_name=document_name,
            page_number=page_number,
        )

        return {
            "intent": intent,
            "query": query,
            "results": results,
            "page_number": page_number,
        }

    # =================================================================
    # RAG SEARCH
    # =================================================================

    def _search_rag(
        self,
        query: str,
        project_name: str | None,
        document_name: str | None,
        page_number: int | None = None,
        limit: int = 8,
    ) -> list[Any]:

        return self._rag.search(
            query=query,
            limit=limit,
            project_name=project_name,
            document_name=document_name,
            page_number=page_number,
        )

    # =================================================================
    # IMAGE SEARCH
    # =================================================================

    def _search_images(
        self,
        query: str,
        project_name: str | None,
        document_name: str | None,
        page_number: int | None,
    ) -> list[Any]:

        """
        Use the actual ImageService API.

        ImageService exposes:
            get_images(...)
        """

        return self._image_service.get_images(
            query=query,
            project_name=project_name,
            document_name=document_name,
            page_number=page_number,
        )

    # =================================================================
    # TABLE SEARCH
    # =================================================================

    def _search_tables(
        self,
        query: str,
        project_name: str | None,
        document_name: str | None,
        page_number: int | None,
        table_number: int | None,
    ) -> list[Any]:

        """
        Use the actual TableService API.

        TableService exposes:
            get_tables(...)
        """

        return self._table_service.get_tables(
            query=query,
            project_name=project_name,
            document_name=document_name,
            page_number=page_number,
            table_number=table_number,
        )

    # =================================================================
    # SUMMARY RETRIEVAL
    # =================================================================

    def _retrieve_summary_evidence(
        self,
        query: str,
        project_name: str | None,
        document_name: str | None,
    ) -> list[Any]:

        """
        Retrieve broader evidence for summaries.

        Gemini will later synthesize ONLY from these retrieved chunks.
        """

        summary_query = (
            f"{query} "
            "main topics sections key points overview"
        )

        return self._search_rag(
            query=summary_query,
            project_name=project_name,
            document_name=document_name,
            page_number=None,
            limit=12,
        )

    # =================================================================
    # PAGE NUMBER
    # =================================================================

    @staticmethod
    def _extract_page_number(
        query: str,
    ) -> int | None:

        patterns = [
            r"\bpage\s*(?:number|no\.?)?\s*[:#-]?\s*(\d+)\b",
            r"\bp\.?\s*(\d+)\b",
        ]

        for pattern in patterns:
            match = re.search(
                pattern,
                query,
                flags=re.IGNORECASE,
            )

            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    return None

        return None

    # =================================================================
    # TABLE NUMBER
    # =================================================================

    @staticmethod
    def _extract_table_number(
        query: str,
    ) -> int | None:

        pattern = (
            r"\btable\s*(?:number|no\.?)?"
            r"\s*[:#-]?\s*(\d+)\b"
        )

        match = re.search(
            pattern,
            query,
            flags=re.IGNORECASE,
        )

        if not match:
            return None

        try:
            return int(match.group(1))
        except ValueError:
            return None