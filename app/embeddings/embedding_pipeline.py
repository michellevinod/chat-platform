from app.chunking.chunk_models import DocumentChunk
from app.embeddings.sentence_transformer_embedding_service import (
    SentenceTransformerEmbeddingService,
)


class EmbeddingPipeline:
    """
    Generates embeddings for document chunks.
    """

    def __init__(self) -> None:
        self._embedding_service = SentenceTransformerEmbeddingService()

    def generate(
        self,
        chunks: list[DocumentChunk],
    ) -> list[DocumentChunk]:

        for chunk in chunks:

            chunk.embedding = self._embedding_service.generate_embedding(
                chunk.text
            )

        return chunks
    