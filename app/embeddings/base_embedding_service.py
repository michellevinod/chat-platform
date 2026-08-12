from abc import ABC, abstractmethod


class BaseEmbeddingService(ABC):
    """
    Base interface for all embedding providers.
    """

    @abstractmethod
    def generate_embedding(
        self,
        text: str,
    ) -> list[float]:
        """
        Generate an embedding for a text.
        """
        raise NotImplementedError