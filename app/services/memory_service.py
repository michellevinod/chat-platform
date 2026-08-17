from typing import Any
import uuid


class MemoryService:
    """
    Lightweight in-memory conversation memory for short-term context.
    Tracks session history, last referenced document/project, and resolves follow-ups.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, list[dict[str, Any]]] = {}
        self._last_prompt: dict[str, str] = {}  # e.g., "ask_document"

    def get_or_create_session(self, session_id: str | None) -> str:
        if not session_id or not session_id.strip():
            session_id = str(uuid.uuid4())
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        return session_id

    def get_history(self, session_id: str) -> list[dict[str, Any]]:
        return self._sessions.get(session_id, [])

    def set_pending_prompt(self, session_id: str, prompt_type: str) -> None:
        self._last_prompt[session_id] = prompt_type

    def clear_pending_prompt(self, session_id: str) -> None:
        self._last_prompt.pop(session_id, None)

    def get_last_context(self, session_id: str) -> dict[str, Any] | None:
        history = self._sessions.get(session_id, [])
        if history:
            return history[-1]
        return None

    def resolve_query(
        self,
        session_id: str,
        query: str,
        explicit_doc: str | None = None,
        explicit_proj: str | None = None,
    ) -> tuple[str, str | None, str | None]:
        """
        Resolves follow-ups and pending document prompts.
        Returns (effective_query, effective_document, effective_project).
        """
        query_strip = query.strip()
        last_turn = self.get_last_context(session_id)
        pending = self._last_prompt.get(session_id)

        resolved_doc = explicit_doc or (last_turn.get("document") if last_turn else None)
        resolved_proj = explicit_proj or (last_turn.get("project") if last_turn else None)

        # 1. If previous turn asked "Which document would you like me to summarize?"
        if pending == "ask_document_summary" and last_turn:
            self.clear_pending_prompt(session_id)
            doc_name = query_strip
            return f"Summarize document {doc_name}", doc_name, resolved_proj

        if pending == "ask_document" and last_turn:
            self.clear_pending_prompt(session_id)
            prev_q = last_turn.get("query", "")
            return f"{prev_q} in {query_strip}", query_strip, resolved_proj

        # 2. Check for follow-up patterns like "Explain that further", "tell me more", "why is that"
        lowered = query_strip.lower()
        follow_up_phrases = [
            "explain that further",
            "explain further",
            "tell me more",
            "elaborate",
            "more details",
            "why is that",
            "what about that",
            "explain it",
            "continue",
        ]

        if any(phrase in lowered for phrase in follow_up_phrases) and last_turn:
            prev_query = last_turn.get("query", "")
            prev_resp = last_turn.get("response", "")
            combined_query = f"{prev_query}. Context: {prev_resp[:200]}. {query_strip}"
            return combined_query, resolved_doc, resolved_proj

        return query_strip, explicit_doc, explicit_proj

    def save_turn(
        self,
        session_id: str,
        query: str,
        response: str,
        project: str | None = None,
        document: str | None = None,
        citations: list[dict[str, Any]] | None = None,
    ) -> None:
        if session_id not in self._sessions:
            self._sessions[session_id] = []

        self._sessions[session_id].append(
            {
                "query": query,
                "response": response,
                "project": project,
                "document": document,
                "citations": citations or [],
            }
        )

        # Cap history to last 20 turns
        if len(self._sessions[session_id]) > 20:
            self._sessions[session_id] = self._sessions[session_id][-20:]
