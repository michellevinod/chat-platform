from __future__ import annotations


class QueryEnhancer:
    """
    Lightweight query normalizer.

    The user's original wording is preserved because the retrieval
    system should search for what the user actually asked.

    Metadata such as project, document, page number, and chunk type
    should be handled by the retrieval layer rather than by rewriting
    the query text.
    """

    @staticmethod
    def enhance(query: str) -> str:
        if not query:
            return ""

        # Preserve the user's actual query.
        # Only normalize surrounding/repeated whitespace.
        return " ".join(query.strip().split())