class ContextBuilder:
    """
    Converts retrieved chunks into structured LLM context.
    """

    def build(
        self,
        chunks,
    ) -> str:

        seen = set()
        context = []

        for chunk in chunks:

            key = (
                chunk.document_name,
                chunk.page_number,
                chunk.chunk_number,
            )

            if key in seen:
                continue

            seen.add(key)

            proj = getattr(chunk, "project_name", "Default Project")
            doc = chunk.document_name
            page = chunk.page_number
            chunk_type = getattr(chunk, "chunk_type", "text")

            context.append(
                f"""Project: {proj}
Document: {doc}
Page: {page} (Type: {chunk_type})
Content:
{chunk.text}"""
            )

        return "\n\n---\n\n".join(context)