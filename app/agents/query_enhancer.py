from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class QueryPlan:
    """Retrieval-oriented representations of one user information need."""

    original: str
    normalized: str
    variants: tuple[str, ...] = field(default_factory=tuple)
    terms: tuple[str, ...] = field(default_factory=tuple)
    requests_synthesis: bool = False
    requests_visual: bool = False
    requests_table: bool = False
    focus_terms: tuple[str, ...] = field(default_factory=tuple)


class QueryEnhancer:
    """
    Build bounded, deterministic retrieval formulations.

    The original wording remains the primary query. Additional variants
    remove conversational scaffolding and expose useful lexical/structural
    signals without relying on document-specific vocabulary.
    """

    _STOP_WORDS = {
        "a", "an", "and", "are", "be", "can", "did", "do", "does",
        "for", "from", "give", "has", "have", "how", "in", "is", "it",
        "me", "of", "on", "please", "say", "tell", "the", "this", "that",
        "to", "was", "what", "when", "where", "which", "who", "why", "with",
    }

    _VISUAL_TERMS = {
        "chart", "diagram", "figure", "graph", "image", "photo", "picture",
        "screenshot", "visual", "certificate",
    }
    _TABLE_TERMS = {"cell", "column", "row", "spreadsheet", "table", "tabular"}
    _SYNTHESIS_TERMS = {
        "analyze", "analyse", "compare", "comparison", "explain", "overview",
        "summarise", "summarize", "summary", "synthesize", "synthesise",
    }
    _ATTRIBUTE_GROUPS = (
        ("employer", "company", "organization", "workplace", "work", "worked", "internship"),
        ("author", "written", "created", "prepared", "presented", "person"),
        ("date", "time", "when", "year", "month"),
        ("location", "where", "place", "site", "address", "country", "city"),
        ("amount", "cost", "price", "value", "number", "total", "quantity"),
    )

    @classmethod
    def build_plan(cls, query: str) -> QueryPlan:
        original = " ".join((query or "").strip().split())
        if not original:
            return QueryPlan("", "")

        normalized = cls._normalize(original)
        tokens = tuple(
            token for token in re.findall(r"[a-z0-9]+", normalized)
            if len(token) > 2 and token not in cls._STOP_WORDS
        )

        variants: list[str] = [original]
        if normalized and normalized != original.lower():
            variants.append(normalized)

        lexical = " ".join(tokens)
        if lexical and lexical not in variants:
            variants.append(lexical)

        # A compact structural formulation improves matching headings,
        # captions, and metadata while keeping retrieval bounded.
        structural_terms = [
            token for token in tokens
            if token in cls._VISUAL_TERMS or token in cls._TABLE_TERMS
        ]
        if structural_terms and lexical:
            structural = f"{lexical} {' '.join(structural_terms)}"
            if structural not in variants:
                variants.append(structural)

        focus_terms: tuple[str, ...] = ()
        query_terms = set(tokens)
        for group in cls._ATTRIBUTE_GROUPS:
            if query_terms.intersection(group):
                focus_terms = group
                focus_query = " ".join(group)
                if focus_query not in variants:
                    variants.append(focus_query)
                break

        return QueryPlan(
            original=original,
            normalized=normalized,
            variants=tuple(variants[:4]),
            terms=tokens,
            requests_synthesis=bool(set(tokens).intersection(cls._SYNTHESIS_TERMS)),
            requests_visual=bool(set(tokens).intersection(cls._VISUAL_TERMS)),
            requests_table=bool(set(tokens).intersection(cls._TABLE_TERMS)),
            focus_terms=focus_terms,
        )

    @classmethod
    def enhance(cls, query: str) -> str:
        return cls.build_plan(query).normalized

    @staticmethod
    def _normalize(query: str) -> str:
        normalized = query.lower().replace("’", "'")
        normalized = re.sub(r"\bwhat's\b", "what is", normalized)
        normalized = re.sub(r"\bwhats\b", "what is", normalized)
        normalized = re.sub(r"\bwho's\b", "who is", normalized)
        normalized = re.sub(r"\bwhere's\b", "where is", normalized)
        normalized = re.sub(r"\bcan't\b", "cannot", normalized)
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        return " ".join(normalized.split())