from app.rag.rag_tool import RAGTool
from app.services.context_builder import ContextBuilder
from app.services.llm_service import LLMService


class SummaryService:
    """
    Service for generating grounded document and project summaries using Gemini.
    """

    def __init__(self) -> None:
        self._rag = RAGTool()
        self._llm = LLMService()
        self._context_builder = ContextBuilder()

    def summarize_document(
        self,
        document_name: str,
        project_name: str | None = None,
    ) -> str:
        chunks = self._rag.search(
            query="summary overview purpose introduction conclusion findings key points",
            limit=15,
            project_name=project_name,
            document_name=document_name,
        )
        if not chunks:
            return "I couldn't find relevant information in the uploaded documents."

        context = self._context_builder.build(chunks)
        return self._llm.generate_answer(
            question=f"Provide a comprehensive structured summary of document {document_name}.",
            context=context,
        )

    def summarize_project(
        self,
        project_name: str,
    ) -> str:
        chunks = self._rag.search(
            query="project summary overview objectives results status documents",
            limit=20,
            project_name=project_name,
        )
        if not chunks:
            return "I couldn't find relevant information in the uploaded documents."

        context = self._context_builder.build(chunks)
        return self._llm.generate_answer(
            question=f"Provide an executive project summary for project {project_name}.",
            context=context,
        )
