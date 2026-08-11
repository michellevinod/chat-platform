from pathlib import Path

from app.ingestion.parsers.base_parser import BaseParser


class ParserFactory:
    """
    Factory responsible for registering and creating
    document parsers.
    """

    _registry: dict[str, type[BaseParser]] = {}

    @classmethod
    def register(
        cls,
        extension: str,
    ):
        """
        Decorator used by parsers to register themselves.

        Example:
            @ParserFactory.register(".pdf")
            class PDFParser(BaseParser):
                ...
        """

        def decorator(parser_class: type[BaseParser]):
            cls._registry[extension.lower()] = parser_class
            return parser_class

        return decorator

    @classmethod
    def get_parser(
        cls,
        file_path: Path,
    ) -> BaseParser:
        """
        Returns the parser corresponding to a file.

        Args:
            file_path:
                Uploaded document path.

        Raises:
            ValueError:
                If parser is unavailable.
        """

        extension = file_path.suffix.lower()

        parser_class = cls._registry.get(extension)

        if parser_class is None:
            raise ValueError(
                f"No parser registered for '{extension}'."
            )

        return parser_class()