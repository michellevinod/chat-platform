from enum import Enum


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

    def classify(
        self,
        query: str,
    ) -> QueryIntent:

        lowered = query.lower().strip()

        if lowered in self.GREETINGS or any(lowered == g for g in self.GREETINGS):
            return QueryIntent.GREETING

        if any(word in lowered for word in self.BLOCKED):
            return QueryIntent.OUT_OF_SCOPE

        if (
            "project summary" in lowered
            or "summarize project" in lowered
        ):
            return QueryIntent.PROJECT_SUMMARY

        if (
            lowered == "summarize"
            or lowered == "summary"
            or lowered == "summarize this"
            or lowered == "overview"
            or lowered == "give me a summary"
            or lowered == "summarize the document"
        ):
            return QueryIntent.AMBIGUOUS_DOCUMENT

        if any(k in lowered for k in ["summary", "summarize", "overview"]):
            return QueryIntent.DOCUMENT_SUMMARY

        if any(k in lowered for k in ["table", "tabular", "spreadsheet"]):
            return QueryIntent.SEARCH_TABLE

        if any(k in lowered for k in ["image", "figure", "diagram", "photo", "chart"]):
            return QueryIntent.SEARCH_IMAGE

        if any(k in lowered for k in self.SYNTHESIS_KEYWORDS):
            return QueryIntent.RAG_SYNTHESIS

        return QueryIntent.RAG_FACTUAL