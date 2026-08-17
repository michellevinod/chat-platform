from abc import ABC
from abc import abstractmethod


class BaseAgent(ABC):

    @abstractmethod
    def execute(
        self,
        query: str,
    ):
        ...