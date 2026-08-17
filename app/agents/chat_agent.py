from app.agents.base_agent import BaseAgent
from app.agents.query_classifier import (
    QueryClassifier,
    QueryIntent,
)
from app.agents.query_enhancer import QueryEnhancer
from app.rag.rag_tool import RAGTool


class ChatAgent(BaseAgent):
    """
    Main orchestration agent.

    Responsible for:
    - Greeting handling
    - Out-of-scope detection
    - Query enhancement
    - RAG search routing

    Future:
    - Document Summary
    - Project Summary
    - Image Search
    - Table Search
    """

    def __init__(self):

        self._classifier = QueryClassifier()

        self._enhancer = QueryEnhancer()

        self._rag = RAGTool()

    def execute(
        self,
        query: str,
    ):

        intent = self._classifier.classify(query)

        if intent == QueryIntent.GREETING:

            return {
                "intent": intent,
                "response":
                    "Hello! 👋 Upload one or more documents and ask me anything related to them."
            }

        if intent == QueryIntent.OUT_OF_SCOPE:

            return {
                "intent": intent,
                "response":
                    "I can answer questions only from uploaded documents."
            }

        enhanced_query = self._enhancer.enhance(query)

        if intent == QueryIntent.RAG_SEARCH:

            return {
                "intent": intent,
                "results": self._rag.search(
                    query=enhanced_query,
                    limit=5,
                ),
            }

        return {
            "intent": intent,
            "query": enhanced_query,
        }