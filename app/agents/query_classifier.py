from enum import Enum
import re


class QueryIntent(str, Enum):
    GREETING = "greeting"

    OUT_OF_SCOPE = "out_of_scope"

    DOCUMENT_SUMMARY = "document_summary"

    PROJECT_SUMMARY = "project_summary"

    SEARCH_TABLE = "search_table"

    SEARCH_IMAGE = "search_image"

    AMBIGUOUS_DOCUMENT = "ambiguous_document"

    RAG_FACTUAL = "rag_factual"

    RAG_SYNTHESIS = "rag_synthesis"

    RAG_SEARCH = "rag_search"


class QueryClassifier:
    """
    Classifies document-chat queries before retrieval.

    The classifier is intentionally deterministic for routing decisions.
    It does not call an LLM.

    Image queries are recognized from:
    - image-related words
    - figure-related words
    - image filenames
    - image extensions
    - explicit page + image requests
    """

    GREETINGS = {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "greetings",
        "howdy",
    }

    BLOCKED = {
        "capital of",
        "capital city",
        "weather in",
        "weather today",
        "weather",
        "movie",
        "recipe",
        "how to cook",
        "how to make",
        "idli",
        "dosa",
        "cooking",
        "football",
        "cricket",
        "ipl",
        "fifa",
        "president of",
        "prime minister",
        "election",
        "politics",
        "actor",
        "actress",
        "lyrics",
        "tell me a joke",
        "horoscope",
        "bitcoin",
    }

    SYNTHESIS_KEYWORDS = {
        "summarize",
        "summary",
        "overview",
        "compare",
        "comparison",
        "difference",
        "differences",
        "analyse",
        "analyze",
        "analysis",
        "explain",
        "synthesize",
        "relationship",
        "insights",
        "evaluate",
        "pros and cons",
    }

    IMAGE_KEYWORDS = {
        "image",
        "images",
        "figure",
        "fig.",
        "fig ",
        "diagram",
        "photo",
        "photograph",
        "picture",
        "pictures",
        "illustration",
        "chart",
        "graph",
        "plot",
        "show me the image",
        "show the image",
        "show me a figure",
        "show the figure",
        "display the image",
        "display the figure",
    }

    IMAGE_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".gif",
        ".bmp",
    }

    def classify(
        self,
        query: str,
    ) -> QueryIntent:

        lowered = query.lower().strip()

        # ---------------------------------------------------------
        # Greeting
        # ---------------------------------------------------------

        if lowered in self.GREETINGS:
            return QueryIntent.GREETING

        # ---------------------------------------------------------
        # Out of scope
        # ---------------------------------------------------------

        if any(
            word in lowered
            for word in self.BLOCKED
        ):
            return QueryIntent.OUT_OF_SCOPE

        # ---------------------------------------------------------
        # Explicit image filename
        #
        # Examples:
        #
        # img_f28db4f9_p37_2.png
        # figure_01.jpg
        # diagram.webp
        # ---------------------------------------------------------

        if QueryClassifier._contains_image_filename(
            lowered
        ):
            return QueryIntent.SEARCH_IMAGE

        # ---------------------------------------------------------
        # Explicit page + image request
        #
        # Examples:
        #
        # image on page 44
        # figure on page 18
        # picture from page 32
        # show page 50 image
        # ---------------------------------------------------------

        if (
            QueryClassifier._contains_page_reference(
                lowered
            )
            and any(
                keyword in lowered
                for keyword in {
                    "image",
                    "figure",
                    "diagram",
                    "photo",
                    "picture",
                    "chart",
                    "graph",
            }
            )
        ):
            return QueryIntent.SEARCH_IMAGE

        # ---------------------------------------------------------
        # Project summary
        # ---------------------------------------------------------

        if (
            "project summary" in lowered
            or "summarize project" in lowered
        ):
            return QueryIntent.PROJECT_SUMMARY

        # ---------------------------------------------------------
        # Ambiguous summary
        # ---------------------------------------------------------

        if (
            lowered == "summarize"
            or lowered == "summary"
            or lowered == "summarize this"
            or lowered == "overview"
            or lowered == "give me a summary"
            or lowered == "summarize the document"
        ):
            return QueryIntent.AMBIGUOUS_DOCUMENT

        # ---------------------------------------------------------
        # Document summary
        # ---------------------------------------------------------

        if any(
            keyword in lowered
            for keyword in [
                "summary",
                "summarize",
                "overview",
            ]
        ):
            return QueryIntent.DOCUMENT_SUMMARY

        # ---------------------------------------------------------
        # Table
        # ---------------------------------------------------------

        if any(
            keyword in lowered
            for keyword in [
                "table",
                "tabular",
                "spreadsheet",
            ]
        ):
            return QueryIntent.SEARCH_TABLE

        # ---------------------------------------------------------
        # Image / figure / diagram
        # ---------------------------------------------------------

        if any(
            keyword in lowered
            for keyword in self.IMAGE_KEYWORDS
        ):
            return QueryIntent.SEARCH_IMAGE

        # ---------------------------------------------------------
        # Synthesis
        # ---------------------------------------------------------

        if any(
            keyword in lowered
            for keyword in self.SYNTHESIS_KEYWORDS
        ):
            return QueryIntent.RAG_SYNTHESIS

        # ---------------------------------------------------------
        # Default factual document search
        # ---------------------------------------------------------

        return QueryIntent.RAG_FACTUAL

    @staticmethod
    def _contains_page_reference(
        query: str,
    ) -> bool:
        """
        Detect explicit page references.

        Examples:
            page 44
            page 18
            pg 12
            p. 37
        """

        return bool(
            re.search(
                r"\b(?:page|pages|pg|p\.)\s*\d+\b",
                query,
            )
        )

    @classmethod
    def _contains_image_filename(
        cls,
        query: str,
    ) -> bool:
        """
        Detect an image filename or path.

        Examples:
            img_f28db4f9_p37_2.png
            figure_1.jpg
            diagram.webp
            storage/images/test.png
        """

        for extension in cls.IMAGE_EXTENSIONS:
            if extension in query:
                return True

        # Also recognize common image identifiers even if the
        # extension has been omitted.
        if re.search(
            r"\bimg[_-][a-z0-9_-]+\b",
            query,
        ):
            return True

        return False