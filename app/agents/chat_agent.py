from __future__ import annotations

import re

from app.agents.base_agent import BaseAgent
from app.agents.query_classifier import (
    QueryClassifier,
    QueryIntent,
)
from app.agents.query_enhancer import QueryEnhancer
from app.rag.rag_tool import RAGTool
from app.services.image_service import ImageService
from app.services.table_service import TableService


class ChatAgent(BaseAgent):
    """
    Main document-intelligence orchestration agent.

    Routes document queries according to their intent so that
    images and tables do not get mixed with ordinary text retrieval.

    Explicit page references are extracted from the user's query
    and passed as exact metadata filters instead of relying on
    semantic similarity.
    """

    def __init__(self):
        self._classifier = QueryClassifier()
        self._enhancer = QueryEnhancer()
        self._rag = RAGTool()
        self._image_service = ImageService()
        self._table_service = TableService()

    def execute(
        self,
        query: str,
        project_name: str | None = None,
        document_name: str | None = None,
    ):
        intent = self._classifier.classify(query)

        if intent == QueryIntent.GREETING:
            return {
                "intent": intent,
                "response": (
                    "Hello! 👋 Upload one or more documents "
                    "and ask me anything related to them."
                ),
            }

        if intent == QueryIntent.OUT_OF_SCOPE:
            return {
                "intent": intent,
                "response": (
                    "I can answer questions only from uploaded "
                    "documents."
                ),
            }

        enhanced_query = self._enhancer.enhance(query)

        # ---------------------------------------------------------
        # Extract explicit page reference.
        #
        # Examples:
        #
        #   "show me table on page 24"
        #   "show the image from page 37"
        #   "figure on page 10"
        #
        # If no explicit page is present, page_number remains None
        # and normal semantic retrieval is used.
        # ---------------------------------------------------------

        page_number = self._extract_page_number(
            query
        )

        # ---------------------------------------------------------
        # IMAGE SEARCH
        # ---------------------------------------------------------

        if intent == QueryIntent.SEARCH_IMAGE:
            results = self._image_service.get_images(
                query=enhanced_query,
                project_name=project_name,
                document_name=document_name,
                limit=5,
                page_number=page_number,
            )

            return {
                "intent": intent,
                "results": results,
            }

        # ---------------------------------------------------------
        # TABLE SEARCH
        # ---------------------------------------------------------

        if intent == QueryIntent.SEARCH_TABLE:
            results = self._table_service.get_tables(
                query=enhanced_query,
                project_name=project_name,
                document_name=document_name,
                limit=5,
                page_number=page_number,
            )

            return {
                "intent": intent,
                "results": results,
            }

        # ---------------------------------------------------------
        # NORMAL DOCUMENT SEARCH
        # ---------------------------------------------------------

        if intent in {
            QueryIntent.RAG_FACTUAL,
            QueryIntent.RAG_SYNTHESIS,
            QueryIntent.RAG_SEARCH,
        }:
            results = self._rag.search(
                query=enhanced_query,
                limit=5,
                project_name=project_name,
                document_name=document_name,
            )

            return {
                "intent": intent,
                "results": results,
            }

        return {
            "intent": intent,
            "query": enhanced_query,
        }

    @staticmethod
    def _extract_page_number(
        query: str,
    ) -> int | None:
        """
        Extract an explicit page number from a natural-language query.

        Supported examples:

            page 24
            Page 24
            on page 24
            from page 24
            pages 24

        Returns None when the user did not explicitly specify
        a page.
        """

        match = re.search(
            r"\bpages?\s+(\d+)\b",
            query,
            flags=re.IGNORECASE,
        )

        if not match:
            return None

        try:
            page_number = int(
                match.group(1)
            )
        except ValueError:
            return None

        if page_number <= 0:
            return None

        return page_number