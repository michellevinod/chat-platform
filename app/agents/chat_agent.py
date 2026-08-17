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
        # IMAGE SEARCH
        # ---------------------------------------------------------

        if intent == QueryIntent.SEARCH_IMAGE:
            results = self._image_service.get_images(
                query=enhanced_query,
                project_name=project_name,
                document_name=document_name,
                limit=5,
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