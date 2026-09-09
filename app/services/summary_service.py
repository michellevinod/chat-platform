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
        document_names = self._rag.get_distinct_documents(
            project_name=project_name,
        )

        if not document_names:
            return (
                f"I couldn't find any documents in project \"{project_name}\"."
            )

        context_sections = [
            (
                f"Project: {project_name}\n"
                f"Document count: {len(document_names)}\n"
                "Documents in this project:\n"
                + "\n".join(
                    f"{index}. {document_name}"
                    for index, document_name in enumerate(document_names, start=1)
                )
            )
        ]

        for index, document_name in enumerate(document_names, start=1):
            chunks = self._rag.get_representative_document_content(
                document_name=document_name,
                project_name=project_name,
                limit=6,
            )

            if not chunks:
                context_sections.append(
                    f"Document {index}: {document_name}\n"
                    "No representative text chunks were found for this document."
                )
                continue

            context_sections.append(
                f"Document {index}: {document_name}\n"
                f"{self._context_builder.build(chunks)}"
            )

        summary_context = "\n\n---\n\n".join(context_sections)

        return self._llm.generate_answer(
            question=(
                f"Create a grounded AI project summary for project \"{project_name}\". "
                "Use only the supplied project context. "
                "Include: "
                "1) project overview, "
                "2) total number of documents/files in the project, "
                "3) a list of every document/file in the project, "
                "4) a short grounded overview of what each document contains, "
                "5) major topics or themes across the project, "
                "6) useful overall or cross-document observations where supported. "
                "Do not invent information, do not include documents outside this project, "
                "and do not rely on any outside knowledge. "
                "Format the response with clear headings."
            ),
            context=summary_context,
        )
