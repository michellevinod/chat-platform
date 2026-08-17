from app.services.document_intelligence_service import DocumentIntelligenceService


class ChatService:
    """
    Main chat orchestration service.
    Delegates to DocumentIntelligenceService for grounded RAG,
    filtering, ambiguity resolution, and Gemini synthesis.
    """

    def __init__(self) -> None:
        self._doc_intelligence = DocumentIntelligenceService()

    def chat(
        self,
        query: str,
        project_name: str | None = None,
        document_name: str | None = None,
        session_id: str | None = None,
    ) -> dict:
        return self._doc_intelligence.answer(
            question=query,
            project_name=project_name,
            document_name=document_name,
            session_id=session_id,
        )