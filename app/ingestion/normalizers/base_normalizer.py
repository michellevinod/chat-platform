from abc import ABC, abstractmethod

from app.ingestion.extractors.raw_models import RawDocument


class BaseNormalizer(ABC):

    @abstractmethod
    def normalize(
        self,
        document: RawDocument,
    ) -> RawDocument:
        pass