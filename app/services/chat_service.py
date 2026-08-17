from app.agents.chat_agent import ChatAgent
from app.agents.query_classifier import QueryIntent


class ChatService:
    """
    Main chat orchestration service.

    Passes project/document scope from the API into the
    document-intelligence agent.
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
        self._agent = ChatAgent()

    def chat(
        self,
        query: str,
        project_name: str | None = None,
        document_name: str | None = None,
        session_id: str | None = None,
        conversation_id: str | None = None,
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
                "response": (
                    "Hello! 👋 Upload one or more documents "
                    "and ask me anything related to them."
                ),
                "citations": [],
                "session_id": session_id,
            }

        if self._is_out_of_scope(query):
            return {
                "success": True,
                "response": (
                    "I can answer questions only from uploaded "
                    "documents."
                ),
                "citations": [],
                "session_id": session_id,
            }

        agent_response = self._agent.execute(
            query=query,
            project_name=project_name,
            document_name=document_name,
        )

        intent = agent_response["intent"]

        if intent in {
            QueryIntent.GREETING,
            QueryIntent.OUT_OF_SCOPE,
        }:
            return {
                "success": True,
                "response": agent_response.get(
                    "response",
                    "",
                ),
                "citations": [],
                "session_id": session_id,
            }

        results = agent_response.get(
            "results",
            [],
        )

        if not results:
            return {
                "success": True,
                "response": (
                    "I couldn't find relevant information "
                    "in the uploaded documents."
                ),
                "citations": [],
                "session_id": session_id,
            }

        # Keep the retrieved chunks available to the response
        # builder instead of flattening them into plain strings.
        citations = []

        seen = set()

        for chunk in results:
            key = (
                chunk.document_name,
                chunk.page_number,
                chunk.chunk_type,
            )

            if key in seen:
                continue

            seen.add(key)

            citations.append(
                {
                    "project": project_name,
                    "document": chunk.document_name,
                    "page": chunk.page_number,
                    "source": chunk.source,
                    "chunk_type": chunk.chunk_type,
                }
            )

        # ---------------------------------------------------------
        # IMAGE RESPONSE
        # ---------------------------------------------------------

        if intent == QueryIntent.SEARCH_IMAGE:
            chunk = results[0]

            image_path = (
                getattr(chunk, "image_path", None)
                or getattr(chunk, "image_id", None)
            )

            response = (
                f"### Figure / Image\n\n"
                f"**Document:** `{chunk.document_name}`  \n"
                f"**Page:** {chunk.page_number}  \n\n"
                f"![Figure]({image_path})"
            )

            return {
                "success": True,
                "response": response,
                "citations": citations,
                "session_id": session_id,
            }

        # ---------------------------------------------------------
        # TABLE RESPONSE
        # ---------------------------------------------------------

        if intent == QueryIntent.SEARCH_TABLE:
            tables = []

            for chunk in results[:2]:
                tables.append(
                    f"### Table from `{chunk.document_name}` "
                    f"(Page {chunk.page_number})\n\n"
                    f"{chunk.text.strip()}"
                )

            return {
                "success": True,
                "response": "\n\n".join(tables),
                "citations": citations,
                "session_id": session_id,
            }

        # ---------------------------------------------------------
        # NORMAL DOCUMENT RESPONSE
        # ---------------------------------------------------------

        unique_chunks = []
        seen_text = set()

        for chunk in results[:3]:
            text = chunk.text.strip()

            if not text or text in seen_text:
                continue

            seen_text.add(text)
            unique_chunks.append(text)

        return {
            "success": True,
            "response": "\n\n".join(unique_chunks),
            "citations": citations,
            "session_id": session_id,
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

        blocked = {
            "capital of",
            "capital city",
            "weather in",
            "weather today",
            "weather",
            "movie",
            "recipe",
            "how to cook",
            "how to make",
            "idli",
            "dosa",
            "cooking",
            "football",
            "cricket",
            "ipl",
            "fifa",
            "president of",
            "prime minister",
            "election",
            "politics",
            "actor",
            "actress",
            "lyrics",
            "tell me a joke",
            "horoscope",
            "bitcoin",
        }

        return any(
            word in lowered
            for word in blocked
        )