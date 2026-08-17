from typing import Any
from app.rag.retrieved_chunk import RetrievedChunk
from app.schemas.chat_schema import Citation


class CitationService:
    """
    Builds clean, deduplicated user-facing citations.
    Never exposes internal UUIDs or database IDs.
    """

    def build_citations(
        self,
        chunks: list[RetrievedChunk],
    ) -> list[Citation]:
        citations: list[Citation] = []
        seen = set()

        for chunk in chunks:
            proj = chunk.project_name or "Default Project"
            doc = chunk.document_name
            page = chunk.page_number
            src = chunk.source
            chunk_type = chunk.chunk_type

            key = (proj, doc, page)
            if key in seen:
                continue

            seen.add(key)
            citations.append(
                Citation(
                    project=proj,
                    document=doc,
                    page=page,
                    source=src,
                    chunk_type=chunk_type,
                )
            )

        return citations

    def format_citations_markdown(
        self,
        citations: list[Citation],
    ) -> str:
        if not citations:
            return ""

        lines = ["\n\n### Sources & Citations\n"]
        for idx, cite in enumerate(citations, start=1):
            proj_str = f"**Project:** {cite.project} | " if cite.project else ""
            lines.append(f"{idx}. {proj_str}**Document:** `{cite.document}` | **Page:** {cite.page}")

        return "\n".join(lines)
