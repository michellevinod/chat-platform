class FollowupResolver:
    """
    Resolves ambiguities and matches documents referenced in queries.
    """

    def extract_document_mention(
        self,
        query: str,
        available_documents: list[str],
    ) -> str | None:
        """
        Checks if the query explicitly mentions one of the available document names or stems.
        """
        lowered_q = query.lower()
        for doc in available_documents:
            doc_lower = doc.lower()
            stem = doc_lower.rsplit(".", 1)[0]
            if doc_lower in lowered_q or (len(stem) > 3 and stem in lowered_q):
                return doc

        return None

    def is_ambiguous_summary(
        self,
        query: str,
        selected_doc: str | None,
        available_documents: list[str],
    ) -> bool:
        """
        Returns True if user asks for a summary without specifying which document,
        and there are multiple available documents.
        """
        if selected_doc:
            return False

        lowered = query.lower().strip()
        generic_summary_phrases = [
            "summarize",
            "summary",
            "summarize this",
            "summarize the document",
            "give me a summary",
            "overview",
            "document summary",
        ]

        if lowered in generic_summary_phrases:
            return len(available_documents) > 1

        return False
