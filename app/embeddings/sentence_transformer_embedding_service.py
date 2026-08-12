from sentence_transformers import SentenceTransformer

from app.embeddings.base_embedding_service import BaseEmbeddingService


class SentenceTransformerEmbeddingService(BaseEmbeddingService):
    """
    Generates embeddings using SentenceTransformers.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:

        self._model = SentenceTransformer(model_name)

    def generate_embedding(
        self,
        text: str,
    ) -> list[float]:

        embedding = self._model.encode(
            text,
            normalize_embeddings=True,
        )

        return embedding.tolist()