from app.rag.rag_tool import RAGTool


class ChatService:
    """
    Main chat orchestration service.

    Responsibilities:
    - Greeting detection
    - Reject unrelated questions
    - Invoke RAG
    - Format citations
    """

    GREETINGS = {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
    }

    def __init__(self) -> None:
        self._rag = RAGTool()

    def chat(
        self,
        query: str,
    ) -> dict:

        query = query.strip()

        if not query:
            return {
                "success": False,
                "message": "Query cannot be empty.",
            }

        if self._is_greeting(query):
            return {
                "success": True,
                "response": "Hello! 👋 Upload one or more documents and ask me anything related to them.",
                "citations": [],
            }

        if self._is_out_of_scope(query):
            return {
                "success": True,
                "response": (
                    "I can answer questions only from uploaded documents. "
                    "Please upload a document and ask questions related to it."
                ),
                "citations": [],
            }

        results = self._rag.search(
            query=query,
            limit=5,
        )

        if not results:
            return {
                "success": True,
                "response": "I couldn't find relevant information in the uploaded documents.",
                "citations": [],
            }

        answer = "\n\n".join(
            chunk.text
            for chunk in results
        )

        citations = []

        seen = set()

        for chunk in results:

            key = (
                chunk.document_name,
                chunk.page_number,
            )

            if key in seen:
                continue

            seen.add(key)

            citations.append(
                {
                    "document": chunk.document_name,
                    "page": chunk.page_number,
                }
            )

        return {
            "success": True,
            "response": answer,
            "citations": citations,
        }

    def _is_greeting(
        self,
        query: str,
    ) -> bool:

        return query.lower() in self.GREETINGS

    def _is_out_of_scope(
        self,
        query: str,
    ) -> bool:

        lowered = query.lower()

        blocked = [
            "capital of",
            "who is",
            "weather",
            "news",
            "cricket",
            "football",
            "movie",
            "recipe",
        ]

        return any(word in lowered for word in blocked)