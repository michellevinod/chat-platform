from __future__ import annotations

from app.agents.chat_agent import ChatAgent
from app.agents.query_classifier import (
    QueryClassifier,
    QueryIntent,
)


class ChatService:
    """
    Main chat orchestration service.

    Responsibilities:
    - Validate the incoming query.
    - Classify the query through the central QueryClassifier.
    - Pass project/document scope into ChatAgent.
    - Build citations.
    - Render images using the public image endpoint.
    - Render structured tables as Markdown.
    - Return normal document responses.
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

    def __init__(self) -> None:
        self._agent = ChatAgent()
        self._classifier = QueryClassifier()

    def chat(
        self,
        query: str,
        project_name: str | None = None,
        document_name: str | None = None,
        session_id: str | None = None,
        conversation_id: str | None = None,
    ) -> dict:
        """
        Process one document-chat request.

        No LLM is called by this service itself.
        """

        query = query.strip()

        # ---------------------------------------------------------
        # EMPTY QUERY
        # ---------------------------------------------------------

        if not query:
            return {
                "success": False,
                "message": "Query cannot be empty.",
            }

        # ---------------------------------------------------------
        # CENTRAL CLASSIFICATION
        # ---------------------------------------------------------

        intent = self._classifier.classify(query)

        # ---------------------------------------------------------
        # GREETING
        # ---------------------------------------------------------

        if intent == QueryIntent.GREETING:
            return {
                "success": True,
                "response": (
                    "Hello! 👋 Upload one or more documents "
                    "and ask me anything related to them."
                ),
                "citations": [],
                "session_id": session_id,
            }

        # ---------------------------------------------------------
        # OUT OF SCOPE
        # ---------------------------------------------------------

        if intent == QueryIntent.OUT_OF_SCOPE:
            return {
                "success": True,
                "response": (
                    "I can answer questions only from uploaded "
                    "documents."
                ),
                "citations": [],
                "session_id": session_id,
            }

        # ---------------------------------------------------------
        # DOCUMENT AGENT
        # ---------------------------------------------------------

        agent_response = self._agent.execute(
            query=query,
            project_name=project_name,
            document_name=document_name,
        )

        agent_intent = agent_response.get(
            "intent",
            intent,
        )

        # ---------------------------------------------------------
        # AGENT-LEVEL GREETING / OUT OF SCOPE
        # ---------------------------------------------------------

        if agent_intent in {
            QueryIntent.GREETING,
            QueryIntent.OUT_OF_SCOPE,
        }:
            return {
                "success": True,
                "response": agent_response.get(
                    "response",
                    "",
                ),
                "citations": [],
                "session_id": session_id,
            }

        # ---------------------------------------------------------
        # RETRIEVED RESULTS
        # ---------------------------------------------------------

        results = agent_response.get(
            "results",
            [],
        )

        if not results:
            return {
                "success": True,
                "response": (
                    "I couldn't find relevant information "
                    "in the uploaded documents."
                ),
                "citations": [],
                "session_id": session_id,
            }

        # ---------------------------------------------------------
        # IMAGE RESPONSE
        # ---------------------------------------------------------

        if agent_intent == QueryIntent.SEARCH_IMAGE:

            # For an image request, the first result is the
            # selected/best image.
            chunk = results[0]

            image_path = getattr(
                chunk,
                "image_path",
                None,
            )

            image_id = getattr(
                chunk,
                "image_id",
                None,
            )

            image_name = self._extract_image_name(
                image_path=image_path,
                image_id=image_id,
            )

            # If an image was retrieved but we cannot determine
            # its public filename, return a safe text response
            # rather than exposing an internal filesystem path.
            if not image_name:
                return {
                    "success": True,
                    "response": (
                        "### Figure / Image\n\n"
                        f"**Document:** "
                        f"`{chunk.document_name}`  \n"
                        f"**Page:** "
                        f"{chunk.page_number}"
                    ),
                    "citations": [
                        self._build_citation(
                            chunk,
                            project_name,
                        )
                    ],
                    "session_id": session_id,
                }

            # Public API route.
            #
            # The frontend/browser can request:
            #
            # /images/<filename>
            public_image_url = (
                f"/images/{image_name}"
            )

            response = (
                "### Figure / Image\n\n"
                f"**Document:** "
                f"`{chunk.document_name}`  \n"
                f"**Page:** "
                f"{chunk.page_number}  \n\n"
                f"![Figure]({public_image_url})"
            )

            # For one selected image, cite ONLY that image.
            citation = self._build_citation(
                chunk,
                project_name,
            )

            return {
                "success": True,
                "response": response,
                "citations": [citation],
                "session_id": session_id,
            }

        # ---------------------------------------------------------
        # TABLE RESPONSE
        # ---------------------------------------------------------

        if agent_intent == QueryIntent.SEARCH_TABLE:

            tables = []

            # Keep the number of returned tables limited.
            for chunk in results[:2]:

                markdown_table = (
                    self._render_table_markdown(
                        chunk
                    )
                )

                if not markdown_table:
                    continue

                tables.append(
                    f"### Table from "
                    f"`{chunk.document_name}` "
                    f"(Page {chunk.page_number})\n\n"
                    f"{markdown_table}"
                )

            citations = self._build_citations(
                results,
                project_name,
            )

            return {
                "success": True,
                "response": (
                    "\n\n".join(tables)
                    if tables
                    else (
                        "I couldn't find a usable table "
                        "in the uploaded documents."
                    )
                ),
                "citations": citations,
                "session_id": session_id,
            }

        # ---------------------------------------------------------
        # NORMAL DOCUMENT RESPONSE
        # ---------------------------------------------------------

        unique_chunks = []
        seen_text = set()

        for chunk in results[:3]:

            text = getattr(
                chunk,
                "text",
                "",
            ).strip()

            if not text:
                continue

            if text in seen_text:
                continue

            seen_text.add(text)
            unique_chunks.append(text)

        citations = self._build_citations(
            results,
            project_name,
        )

        return {
            "success": True,
            "response": "\n\n".join(
                unique_chunks
            ),
            "citations": citations,
            "session_id": session_id,
        }

    # =============================================================
    # TABLE RENDERING
    # =============================================================

    @staticmethod
    def _render_table_markdown(
        chunk,
    ) -> str:
        """
        Render a retrieved table using its structured headers
        and rows instead of the flattened semantic-search text.

        The semantic-search text is intended for retrieval.
        table_headers/table_rows are the canonical user-facing
        representation.
        """

        headers = getattr(
            chunk,
            "table_headers",
            None,
        ) or []

        rows = getattr(
            chunk,
            "table_rows",
            None,
        ) or []

        # ---------------------------------------------------------
        # Normalize headers
        # ---------------------------------------------------------

        headers = [
            ChatService._clean_table_cell(
                header
            )
            for header in headers
        ]

        # ---------------------------------------------------------
        # Normalize rows
        # ---------------------------------------------------------

        normalized_rows = []

        for row in rows:

            if not row:
                continue

            normalized_row = [
                ChatService._clean_table_cell(
                    cell
                )
                for cell in row
            ]

            # Ignore completely empty rows.
            if not any(
                cell.strip()
                for cell in normalized_row
            ):
                continue

            normalized_rows.append(
                normalized_row
            )

        # ---------------------------------------------------------
        # Determine table width
        # ---------------------------------------------------------

        column_count = max(
            [len(headers)]
            + [
                len(row)
                for row in normalized_rows
            ]
            + [0]
        )

        # If structured data is unavailable,
        # fall back to the original table text.
        if column_count == 0:
            return (
                getattr(
                    chunk,
                    "text",
                    "",
                )
                or ""
            ).strip()

        # ---------------------------------------------------------
        # Missing headers
        # ---------------------------------------------------------

        if not headers:
            headers = [
                f"Column {index + 1}"
                for index in range(
                    column_count
                )
            ]

        # Pad headers if necessary.
        if len(headers) < column_count:
            headers.extend(
                [
                    ""
                    for _ in range(
                        column_count
                        - len(headers)
                    )
                ]
            )

        # ---------------------------------------------------------
        # Markdown header
        # ---------------------------------------------------------

        lines = []

        lines.append(
            "| "
            + " | ".join(
                headers[:column_count]
            )
            + " |"
        )

        lines.append(
            "| "
            + " | ".join(
                "---"
                for _ in range(
                    column_count
                )
            )
            + " |"
        )

        # ---------------------------------------------------------
        # Markdown rows
        # ---------------------------------------------------------

        for row in normalized_rows:

            if len(row) < column_count:
                row = row + [
                    ""
                    for _ in range(
                        column_count
                        - len(row)
                    )
                ]

            elif len(row) > column_count:
                row = row[:column_count]

            lines.append(
                "| "
                + " | ".join(row)
                + " |"
            )

        return "\n".join(lines)

    @staticmethod
    def _clean_table_cell(
        value,
    ) -> str:
        """
        Normalize one table cell for Markdown output.
        """

        if value is None:
            return ""

        text = str(value)

        # Collapse line breaks and repeated whitespace.
        text = " ".join(
            text.split()
        )

        # Escape Markdown column separators.
        text = text.replace(
            "|",
            "\\|",
        )

        return text

    # =============================================================
    # CITATIONS
    # =============================================================

    @staticmethod
    def _build_citation(
        chunk,
        project_name: str | None,
    ) -> dict:
        """
        Build a clean user-facing citation.
        """

        return {
            "project": (
                getattr(
                    chunk,
                    "project_name",
                    None,
                )
                or project_name
            ),
            "document": getattr(
                chunk,
                "document_name",
                None,
            ),
            "page": getattr(
                chunk,
                "page_number",
                None,
            ),
            "source": getattr(
                chunk,
                "source",
                "upload",
            ),
            "chunk_type": getattr(
                chunk,
                "chunk_type",
                "text",
            ),
        }

    def _build_citations(
        self,
        results,
        project_name: str | None,
    ) -> list[dict]:
        """
        Deduplicate citations by document/page/type.
        """

        citations = []
        seen = set()

        for chunk in results:

            citation = self._build_citation(
                chunk,
                project_name,
            )

            key = (
                citation["document"],
                citation["page"],
                citation["chunk_type"],
            )

            if key in seen:
                continue

            seen.add(key)
            citations.append(
                citation
            )

        return citations

    # =============================================================
    # IMAGE HELPERS
    # =============================================================

    @staticmethod
    def _extract_image_name(
        image_path: str | None,
        image_id: str | None,
    ) -> str | None:
        """
        Extract only the filename from image metadata.

        Handles Windows paths and Unix-style paths.

        Examples:

            storage\\images\\img_abc.png
            storage/images/img_abc.png
            img_abc.png
        """

        candidate = (
            image_path
            or image_id
        )

        if not candidate:
            return None

        candidate = str(
            candidate
        ).strip()

        if not candidate:
            return None

        # Normalize Windows separators.
        candidate = candidate.replace(
            "\\",
            "/",
        )

        # Return only the filename.
        filename = candidate.rsplit(
            "/",
            1,
        )[-1]

        return (
            filename
            if filename
            else None
        )