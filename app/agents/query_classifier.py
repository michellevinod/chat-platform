from __future__ import annotations

import re
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
    RAG_ENUMERATION = "rag_enumeration"


class QueryClassifier:
    """
    Classifies document-chat queries without relying on any
    domain-specific vocabulary.

    Important design rule:
    A word such as "cricket", "weather", "oil", "medicine",
    "finance", etc. may legitimately exist inside an uploaded
    document. Therefore vocabulary alone must never determine
    that a query is out of scope.
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

    SUMMARY_PHRASES = (
    "what is this document about",
    "what does this document cover",
    "what does the document cover",
    "what is the document about",
    "give me an overview",
    "give an overview",
    "provide an overview",
    "overview of the document",
    )

    SUMMARY_PATTERN = re.compile(
        r"\b(?:what|tell me|can you tell me|give me|provide)\b"
        r".*\b(?:document|file|report|paper|manual)\b"
        r".*\b(?:about|contain|cover|overview|summary)\b"
    )

    SYNTHESIS_TERMS = {
        "summarize",
        "summarise",
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
        "synthesise",
        "relationship",
        "relationships",
        "insights",
        "evaluate",
        "evaluation",
        "pros and cons",
        "advantages and disadvantages",
        "contrast",
    }

    TABLE_TERMS = {
        "table",
        "tables",
        "tabular",
        "spreadsheet",
    }

    IMAGE_TERMS = {
        "image",
        "images",
        "figure",
        "figures",
        "diagram",
        "diagrams",
        "photo",
        "photos",
        "chart",
        "charts",
        "graph",
        "graphs",
    }

    @classmethod
    def classify(cls, query: str) -> QueryIntent:
        """
        Determine the routing intent for a document query.

        This classifier intentionally does NOT maintain a list of
        forbidden domain words. Whether a question is answerable
        must ultimately be determined by document retrieval/evidence,
        not by the vocabulary used in the question.
        """

        lowered = cls._normalize(query)

        if not lowered:
            return QueryIntent.RAG_FACTUAL

        # ---------------------------------------------------------
        # GREETING
        # ---------------------------------------------------------

        if lowered in cls.GREETINGS:
            return QueryIntent.GREETING

        # ---------------------------------------------------------
        # PROJECT SUMMARY
        # ---------------------------------------------------------

        if (
            "project summary" in lowered
            or "summary of the project" in lowered
            or "summarize project" in lowered
            or "summarise project" in lowered
            or "overview of the project" in lowered
            or re.search(r"\b(?:summarize|summarise|overview)\b.*\bproject\b", lowered)
        ):
            return QueryIntent.PROJECT_SUMMARY

        # ---------------------------------------------------------
        # AMBIGUOUS SUMMARY
        #
        # "summarize" by itself needs document/project context.
        # The agent/service can resolve that from the selected scope.
        # ---------------------------------------------------------

        if lowered in {
            "summarize",
            "summarise",
            "summary",
            "summarize this",
            "summarise this",
            "overview",
            "give me a summary",
            "give me an overview",
            "summarize the document",
            "summarise the document",
            "overview of the document",
        }:
            return QueryIntent.AMBIGUOUS_DOCUMENT

        # ---------------------------------------------------------
        # DOCUMENT SUMMARY
        # ---------------------------------------------------------

        if cls._contains_any(
            lowered,
            cls.SUMMARY_PHRASES,
        ) or cls.SUMMARY_PATTERN.search(lowered):
            return QueryIntent.DOCUMENT_SUMMARY

        if (
            re.search(r"\b(?:summarize|summarise|summary|overview)\b", lowered)
            and (
                re.search(r"\b(?:document|file|report|paper|manual)\b", lowered)
                or re.search(r"\b(?:pdf|docx?|pptx?|xlsx?|txt|md)\b", lowered)
            )
        ):
            return QueryIntent.DOCUMENT_SUMMARY

        # ---------------------------------------------------------
        # TABLE
        # ---------------------------------------------------------

        if cls._contains_any(
            lowered,
            cls.TABLE_TERMS,
        ):
            return QueryIntent.SEARCH_TABLE

        if (
            cls._contains_any(lowered, cls.IMAGE_TERMS)
            and cls._contains_any_phrase(lowered, cls.SYNTHESIS_TERMS)
        ):
            return QueryIntent.RAG_SYNTHESIS

        # ---------------------------------------------------------
        # IMAGE / FIGURE
        # ---------------------------------------------------------

        if cls._contains_any(
            lowered,
            cls.IMAGE_TERMS,
        ):
            return QueryIntent.SEARCH_IMAGE

        # ---------------------------------------------------------
        # SYNTHESIS / REASONING
        # ---------------------------------------------------------

        if cls._contains_any_phrase(
            lowered,
            cls.SYNTHESIS_TERMS,
        ):
            return QueryIntent.RAG_SYNTHESIS

        if (
            re.search(r"\b(list|enumerate|names|all)\b", lowered)
            or re.search(r"\bwhat are\b|\bwhich are\b", lowered)
            or re.search(r"\bgive me\b", lowered)
            or re.search(r"\bwhat\s+[a-z][a-z -]{1,35}\b(?:does|do|did|can)\b", lowered)
        ):
            return QueryIntent.RAG_ENUMERATION

        # ---------------------------------------------------------
        # DEFAULT
        # ---------------------------------------------------------

        return QueryIntent.RAG_FACTUAL

    @staticmethod
    def _contains_any(
        text: str,
        terms: set[str],
    ) -> bool:
        """
        Match complete words/phrases instead of arbitrary substrings.
        """

        for term in terms:
            if " " in term:
                if term in text:
                    return True
                continue

            if re.search(
                rf"\b{re.escape(term)}\b",
                text,
            ):
                return True

        return False

    @staticmethod
    def _contains_any_phrase(
        text: str,
        terms: set[str],
    ) -> bool:
        """
        Match synthesis expressions while avoiding accidental
        substring matches.
        """

        for term in terms:
            if " " in term:
                if term in text:
                    return True
                continue

            if re.search(
                rf"\b{re.escape(term)}\b",
                text,
            ):
                return True

        return False

    @staticmethod
    def _normalize(query: str) -> str:
        """Normalize common conversational spelling before routing."""
        normalized = (query or "").lower().replace("’", "'")
        normalized = re.sub(r"\bwhat's\b|\bwhats\b", "what is", normalized)
        normalized = re.sub(r"\bwho's\b", "who is", normalized)
        normalized = re.sub(r"\bwhere's\b", "where is", normalized)
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        return " ".join(normalized.split())