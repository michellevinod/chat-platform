from app.rag.retrieved_chunk import RetrievedChunk


class Reranker:
    """
    Reranks candidate retrieved chunks by score and diversity.
    """

    def rerank(
        self,
        chunks: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        # Sort descending by score
        sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)

        # Deduplicate identical text
        seen = set()
        deduped = []
        for c in sorted_chunks:
            t = c.text.strip()
            if t in seen:
                continue
            seen.add(t)
            deduped.append(c)

        return deduped[:top_k]
