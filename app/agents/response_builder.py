from app.rag.retrieved_chunk import RetrievedChunk


class ResponseBuilder:
    """
    Builds clean Markdown responses from retrieved chunks or synthesis output.
    Preserves tables in Markdown and image references.
    """

    def build_factual_response(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> str:
        """
        Formats factual answers directly from retrieved chunks without LLM hallucination.
        """
        if not chunks:
            return "I couldn't find relevant information in the uploaded documents."

        # Check if top chunk is a table
        top_chunk = chunks[0]
        if top_chunk.chunk_type == "table" or "|" in top_chunk.text and "---" in top_chunk.text:
            return f"### Information retrieved from `{top_chunk.document_name}` (Page {top_chunk.page_number})\n\n{top_chunk.text.strip()}"

        # Check if top chunk is an image
        if top_chunk.chunk_type == "image":
            img_ref = getattr(top_chunk, "image_path", None) or getattr(top_chunk, "image_id", "")
            return f"### Figure / Image on Page {top_chunk.page_number} of `{top_chunk.document_name}`\n\n{top_chunk.text}\n\n*Reference:* `{img_ref}`"

        seen_text = set()
        clean_sections = []

        for chunk in chunks[:3]:
            txt = chunk.text.strip()
            if txt in seen_text:
                continue
            seen_text.add(txt)
            clean_sections.append(txt)

        if not clean_sections:
            return "I couldn't find relevant information in the uploaded documents."

        answer_body = "\n\n".join(clean_sections)
        return f"{answer_body}"

    def build_table_response(
        self,
        chunks: list[RetrievedChunk],
    ) -> str:
        """
        Extracts and formats tables found in retrieved chunks.
        """
        table_chunks = [c for c in chunks if c.chunk_type == "table" or ("|" in c.text and "---" in c.text)]
        if not table_chunks:
            return self.build_factual_response("", chunks)

        res = []
        for tc in table_chunks[:2]:
            res.append(f"### Table from `{tc.document_name}` (Page {tc.page_number})\n\n{tc.text.strip()}")

        return "\n\n".join(res)
