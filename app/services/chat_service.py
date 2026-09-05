from __future__ import annotations

import re
from typing import Any

from app.agents.chat_agent import ChatAgent
from app.agents.query_classifier import (
    QueryClassifier,
    QueryIntent,
)
from app.services.llm_service import LLMService


class ChatService:
    """
    Main chat orchestration service.

    Retrieval-first architecture:

        User Query
            |
            v
        QueryClassifier
            |
            v
        ChatAgent
            |
            v
        Qdrant retrieval
            |
            +---- Image/Table -> direct rendering
            |
            +---- Factual -> direct grounded response
            |
            +---- Summary/Synthesis -> Gemini using retrieved evidence

    Important:
    Gemini is NEVER given the original document directly.
    Gemini receives only retrieved evidence from Qdrant.
    """

    NO_RESULTS_MESSAGE = (
        "I couldn't find relevant information in the uploaded documents. "
        "I can answer only from uploaded documents."
    )

    def __init__(self) -> None:
        self._agent = ChatAgent()
        self._classifier = QueryClassifier()
        self._llm = LLMService()

    # =================================================================
        # MAIN CHAT ENTRY POINT
    # =================================================================

    def chat(
        self,
        query: str,
        project_name: str | None = None,
        document_name: str | None = None,
        document_names: list[str] | None = None,
        session_id: str | None = None,
        conversation_id: str | None = None,
        ) -> dict:
        """
        Process one document-chat request.

        The service is retrieval-first.

        Simple factual questions do NOT call Gemini.

        Gemini is used only when the query genuinely requires
        synthesis, summarization, comparison, or explanation.
        """

        query = (query or "").strip()

        # -------------------------------------------------------------
        # EMPTY QUERY
        # -------------------------------------------------------------

        if not query:
            return {
                "success": False,
                "message": "Query cannot be empty.",
            }

        # -------------------------------------------------------------
        # CLASSIFY
        # -------------------------------------------------------------

        intent = self._classifier.classify(query)

        # -------------------------------------------------------------
        # GREETING
        # -------------------------------------------------------------

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

        # -------------------------------------------------------------
        # OUT OF SCOPE
        # -------------------------------------------------------------

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

        if (
            intent == QueryIntent.AMBIGUOUS_DOCUMENT
            and not document_name
            and not document_names
        ):
            return {
                "success": True,
                "response": "Which document would you like me to summarize?",
                "citations": [],
                "session_id": session_id,
            }

        # -------------------------------------------------------------
        # RETRIEVE EVIDENCE
        # -------------------------------------------------------------

        agent_response = self._agent.execute(
            query=query,
            project_name=project_name,
            document_name=document_name,
            document_names=document_names,
        )

        agent_intent = agent_response.get(
            "intent",
            intent,
        )

        # -------------------------------------------------------------
        # AGENT-LEVEL GREETING / OUT OF SCOPE
        # -------------------------------------------------------------

        if agent_intent in {
            QueryIntent.GREETING,
            QueryIntent.OUT_OF_SCOPE,
        }:
            return {
                "success": True,
                "response": agent_response.get(
                    "direct_response",
                    agent_response.get("response", ""),
                ),
                "citations": [],
                "session_id": session_id,
            }

        results = agent_response.get("results") or []

        # -------------------------------------------------------------
        # NO EVIDENCE
        # -------------------------------------------------------------

        if not results:
            return self._no_results(
                session_id=session_id,
                original_question=query,
            )

        # A vector neighbour is not evidence by itself. Exact page/table/image
        # requests are already deterministic; ordinary document questions need
        # a lexical or strong semantic basis before any answer/LLM call.
        if (
            agent_intent in {QueryIntent.RAG_FACTUAL, QueryIntent.RAG_SEARCH, QueryIntent.RAG_SYNTHESIS, QueryIntent.RAG_ENUMERATION}
            and not self._has_sufficient_evidence(query, results)
        ):
            return self._no_results(session_id=session_id, original_question=query)

        # -------------------------------------------------------------
        # IMAGE
        # -------------------------------------------------------------

        if agent_intent == QueryIntent.SEARCH_IMAGE:
            return self._build_image_response(
                results=results,
                project_name=project_name,
                session_id=session_id,
            )

        # -------------------------------------------------------------
        # TABLE
        # -------------------------------------------------------------

        if agent_intent == QueryIntent.SEARCH_TABLE:
            return self._build_table_response(
                results=results,
                project_name=project_name,
                session_id=session_id,
            )

        if agent_response.get("page_number") is not None:
            return self._build_page_response(
                results=results,
                project_name=project_name,
                session_id=session_id,
            )

        # -------------------------------------------------------------
        # SUMMARY / SYNTHESIS
        #
        # These are the cases where Gemini is actually useful.
        # Gemini receives ONLY retrieved evidence.
        # -------------------------------------------------------------

        if agent_intent in {
            QueryIntent.DOCUMENT_SUMMARY,
            QueryIntent.PROJECT_SUMMARY,
            QueryIntent.RAG_SYNTHESIS,
            QueryIntent.RAG_ENUMERATION,
        }:
            return self._build_synthesis_response(
                query=query,
                results=results,
                project_name=project_name,
                session_id=session_id,
            )

        # -------------------------------------------------------------
        # NORMAL FACTUAL / SEARCH RESPONSE
        #
        # IMPORTANT:
        # Do NOT concatenate several unrelated chunks.
        #
        # The top result is the highest-ranked evidence returned by
        # the retrieval layer.
        # -------------------------------------------------------------

        return self._build_factual_response(
            query=query,
            results=results,
            project_name=project_name,
            session_id=session_id,
        )

    # =================================================================
    # FACTUAL RESPONSE
    # =================================================================

    def _build_factual_response(
        self,
        query: str,
        results: list[Any],
        project_name: str | None,
        session_id: str | None,
        ) -> dict:
        """
        Return the strongest retrieved evidence directly.

        No LLM call is made here.

        This is intentional:
        - conserves Gemini quota
        - avoids unnecessary hallucination
        - preserves exact document wording
        - keeps factual retrieval deterministic
        """

        table_chunk = self._select_relevant_table(
            query,
            results,
        )

        if table_chunk is not None:
            table_context = self._build_llm_context(
                [table_chunk]
            )

            response_text = self._llm.generate_answer(
                question=query,
                context=table_context,
            )

            if self._looks_like_no_result(response_text):
                return self._no_results(
                    session_id=session_id,
                    original_question=query,
                )

            if (
                response_text
                and response_text.strip()
                and not self._looks_like_no_result(response_text)
            ):
                supporting_table = self._render_table_markdown(
                    table_chunk
                )

                if supporting_table:
                    response_text = (
                        f"{response_text.strip()}\n\n"
                        f"### Supporting table\n\n"
                        f"{supporting_table}"
                    )

            return {
                "success": True,
                "response": response_text.strip(),
                "citations": self._build_citations(
                    [table_chunk],
                    project_name,
                ),
                "session_id": session_id,
            }

        best_chunk = self._select_best_chunk(results)

        if best_chunk is None:
            return self._no_results(
                session_id=session_id,
                original_question=query,
            )

        text = self._clean_response_text(
            getattr(best_chunk, "text", "")
        )

        if not text:
            return self._no_results(
                session_id=session_id,
                original_question=query,
            )

        supporting_markdown, supporting_chunks = (
            self._render_supporting_evidence(results, best_chunk)
        )
        if supporting_markdown:
            text = f"{text}\n\n{supporting_markdown}"

        citations = self._build_citations(
            [best_chunk, *supporting_chunks],
            project_name,
        )

        return {
            "success": True,
            "response": text,
            "citations": citations,
            "session_id": session_id,
        }

    def _build_page_response(
        self,
        results: list[Any],
        project_name: str | None,
        session_id: str | None,
        ) -> dict:
        """Render the selected page in indexed reading order, without an LLM."""
        text_chunks = [
            chunk for chunk in results
            if (getattr(chunk, "chunk_type", "text") or "text").lower() in {"text", "ocr"}
            and self._clean_response_text(getattr(chunk, "text", ""))
        ]
        if not text_chunks:
            return self._no_results(session_id, "page request")
        content = "\n\n".join(self._clean_response_text(chunk.text) for chunk in text_chunks)
        supporting_markdown, supporting = self._render_supporting_evidence(results, text_chunks[0])
        if supporting_markdown:
            content = f"{content}\n\n{supporting_markdown}"
        return {
            "success": True,
            "response": content,
            "citations": self._build_citations([*text_chunks, *supporting], project_name),
            "session_id": session_id,
        }

    @staticmethod
    def _has_sufficient_evidence(query: str, results: list[Any]) -> bool:
        """Reject unrelated vector neighbours without domain-specific rules."""
        stop_words = {
        "what", "which", "where", "when", "why", "how", "does", "did", "is", "are",
        "the", "a", "an", "this", "that", "document", "say", "about", "show", "me",
        "tell", "please", "on", "in", "of", "for", "to", "and", "with", "page", "from", "into", "than", "then",
        }
        terms = {
        token for token in re.findall(r"[a-z0-9]+", query.lower())
        if len(token) >= 3 and token not in stop_words and not token.isdigit()
        }
        if not terms:
            return bool(results)

        for chunk in results:
            searchable = " ".join(
                str(getattr(chunk, field, "") or "")
                for field in ("text", "heading", "section", "table_caption", "image_caption")
            ).lower()
            if terms.intersection(re.findall(r"[a-z0-9]+", searchable)):
                return True
            if float(getattr(chunk, "score", 0.0) or 0.0) >= 0.72:
                return True
        return False

    @staticmethod
    def _select_relevant_table(query: str, results: list[Any]) -> Any | None:
        """Prefer a table when its actual cells support a factual question."""
        stop_words = {"what", "which", "where", "when", "why", "how", "does", "is", "are", "the", "a", "an", "this", "that", "document", "say", "about", "show", "me", "on", "in", "of", "for", "to", "and", "with"}
        terms = {
        token for token in re.findall(r"[a-z0-9]+", query.lower())
        if len(token) >= 3 and token not in stop_words
        }
        if not terms:
            return None
        for chunk in results:
            if (getattr(chunk, "chunk_type", "") or "").lower() != "table":
                continue
            content = " ".join(
                [str(getattr(chunk, "table_caption", "") or ""), str(getattr(chunk, "text", "") or "")]
            ).lower()
            if terms.intersection(re.findall(r"[a-z0-9]+", content)):
                return chunk
        return None

    def _render_supporting_evidence(
        self,
        results: list[Any],
        best_chunk: Any,
        ) -> tuple[str, list[Any]]:
        """Attach only visual/table chunks already retrieved for the query."""
        parts: list[str] = []
        supporting: list[Any] = []

        for chunk in results:
            if chunk is best_chunk:
                continue

            chunk_type = (getattr(chunk, "chunk_type", "") or "").lower()
            if chunk_type == "table" and not any(
                (getattr(item, "chunk_type", "") or "").lower() == "table"
                for item in supporting
            ):
                table = self._render_table_markdown(chunk)
                if table:
                    caption = getattr(chunk, "table_caption", None) or "Supporting table"
                    parts.append(f"### {caption}\n\n{table}")
                    supporting.append(chunk)

            if chunk_type == "image" and not any(
                (getattr(item, "chunk_type", "") or "").lower() == "image"
                for item in supporting
            ):
                image_name = self._extract_image_name(
                    getattr(chunk, "image_path", None),
                    getattr(chunk, "image_id", None),
                )
                if image_name:
                    caption = getattr(chunk, "image_caption", None) or "Supporting figure"
                    parts.append(f"### {caption}\n\n![Figure](/images/{image_name})")
                    supporting.append(chunk)

            if len(supporting) == 2:
                break

        return "\n\n".join(parts), supporting

    # =================================================================
    # SYNTHESIS RESPONSE
    # =================================================================

    def _build_synthesis_response(
        self,
        query: str,
        results: list[Any],
        project_name: str | None,
        session_id: str | None,
        ) -> dict:
        """
        Use Gemini only for genuine synthesis.

        The model receives retrieved evidence and nothing else.
        """

        evidence = self._build_llm_context(results)

        if not evidence.strip():
            return self._no_results(
                session_id=session_id,
                original_question=query,
            )

        response_text = self._llm.generate_answer(
            question=query,
            context=evidence,
        )

        # The LLM service already has a safe fallback.
        if (
            not response_text
            or not response_text.strip()
            or self._looks_like_no_result(response_text)
        ):
            return self._no_results(
                session_id=session_id,
                original_question=query,
            )

        citations = self._build_citations(
            results,
            project_name,
        )

        return {
            "success": True,
            "response": response_text.strip(),
            "citations": citations,
            "session_id": session_id,
        }

    # =================================================================
    # LLM CONTEXT
    # =================================================================

    @staticmethod
    def _build_llm_context(
        results: list[Any],
        ) -> str:
        """
        Convert retrieved chunks into a controlled evidence context.

        Internal database identifiers are deliberately excluded.
        """

        sections: list[str] = []
        seen: set[tuple] = set()

        # Limit evidence so we do not waste Gemini quota sending
        # excessive duplicate material.
        for index, chunk in enumerate(results[:24], start=1):
            text = ChatService._clean_response_text(
                getattr(chunk, "text", "")
            )

            if not text:
                continue

            document = getattr(
                chunk,
                "document_name",
                None,
            )

            page = getattr(
                chunk,
                "page_number",
                None,
            )

            chunk_type = getattr(
                chunk,
                "chunk_type",
                None,
            ) or "text"

            key = (
                document,
                page,
                chunk_type,
                text,
            )

            if key in seen:
                continue

            seen.add(key)

            location_parts = []

            if document:
                location_parts.append(
                    f"Document: {document}"
                )

            if page is not None:
                location_parts.append(
                    f"Page: {page}"
                )

            location_parts.append(
                f"Type: {chunk_type}"
            )

            location = " | ".join(
                location_parts
            )

            sections.append(
                f"[Evidence {index}]\n"
                f"{location}\n"
                f"{text}"
            )

        return "\n\n".join(sections)

    # =================================================================
    # IMAGE RESPONSE
    # =================================================================

    def _build_image_response(
        self,
        results: list[Any],
        project_name: str | None,
        session_id: str | None,
        ) -> dict:
        """
        Render the best retrieved image.

        Only the public filename is exposed.
        Internal filesystem paths are never returned.
        """

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

        citation = self._build_citation(
            chunk,
            project_name,
        )

        # -------------------------------------------------------------
        # Image metadata exists but no public filename
        # -------------------------------------------------------------

        if not image_name:
            return {
                "success": True,
                "response": (
                    "### Figure / Image\n\n"
                    f"**Document:** "
                    f"`{getattr(chunk, 'document_name', 'Unknown')}`  \n"
                    f"**Page:** "
                    f"{getattr(chunk, 'page_number', 'Unknown')}"
                ),
                "citations": [citation],
                "session_id": session_id,
            }

        # -------------------------------------------------------------
        # Public image endpoint
        # -------------------------------------------------------------

        public_image_url = (
            f"/images/{image_name}"
        )

        document_name = getattr(
            chunk,
            "document_name",
            None,
        )

        page_number = getattr(
            chunk,
            "page_number",
            None,
        )

        response = (
            "### Figure / Image\n\n"
            f"**Document:** `{document_name}`  \n"
            f"**Page:** {page_number}  \n\n"
            f"![Figure]({public_image_url})"
        )

        return {
            "success": True,
            "response": response,
            "citations": [citation],
            "session_id": session_id,
        }

    # =================================================================
    # TABLE RESPONSE
    # =================================================================

    def _build_table_response(
        self,
        results: list[Any],
        project_name: str | None,
        session_id: str | None,
        ) -> dict:
        """
        Render structured table data directly.

        Tables do not need Gemini for normal retrieval/display.
        """

        tables: list[str] = []

        seen: set[tuple] = set()

        for chunk in results[:3]:
            document_name = getattr(
                chunk,
                "document_name",
                None,
            )

            page_number = getattr(
                chunk,
                "page_number",
                None,
            )

            table_id = getattr(
                chunk,
                "table_id",
                None,
            )

            key = (
                document_name,
                page_number,
                table_id,
            )

            if key in seen:
                continue

            seen.add(key)

            markdown_table = (
                self._render_table_markdown(
                    chunk
                )
            )

            if not markdown_table:
                continue

            tables.append(
                f"### Table from `{document_name}` "
                f"(Page {page_number})\n\n"
                f"{markdown_table}"
            )

        if not tables:
            return {
                "success": True,
                "response": (
                    "I couldn't find a usable table "
                    "in the uploaded documents."
                ),
                "citations": [],
                "session_id": session_id,
            }

        citations = self._build_citations(
            results,
            project_name,
        )

        return {
            "success": True,
            "response": "\n\n".join(tables),
            "citations": citations,
            "session_id": session_id,
        }

    # =================================================================
    # TABLE MARKDOWN
    # =================================================================

    @staticmethod
    def _render_table_markdown(
        chunk,
        ) -> str:
        """
        Render structured table data.

        Preferred source:
            table_headers
            table_rows

        Fallback:
            chunk.text
        """

        headers = (
            getattr(
                chunk,
                "table_headers",
                None,
            )
            or []
        )

        rows = (
            getattr(
                chunk,
                "table_rows",
                None,
            )
            or []
        )

        # -------------------------------------------------------------
        # Normalize headers
        # -------------------------------------------------------------

        headers = [
            ChatService._clean_table_cell(
                header
            )
            for header in headers
        ]

        # -------------------------------------------------------------
        # Normalize rows
        # -------------------------------------------------------------

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

            if not any(
                cell.strip()
                for cell in normalized_row
            ):
                continue

            normalized_rows.append(
                normalized_row
            )

        # -------------------------------------------------------------
        # Determine width
        # -------------------------------------------------------------

        column_count = max(
            [len(headers)]
            + [
                len(row)
                for row in normalized_rows
            ]
            + [0]
        )

        # -------------------------------------------------------------
        # Structured data unavailable
        # -------------------------------------------------------------

        if column_count == 0:
            return (
                ChatService._clean_response_text(
                    getattr(
                        chunk,
                        "text",
                        "",
                    )
                )
            )

        # -------------------------------------------------------------
        # Create generic headers when needed
        # -------------------------------------------------------------

        if not headers:
            headers = [
                f"Column {index + 1}"
                for index in range(
                    column_count
                )
            ]

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

        # -------------------------------------------------------------
        # Markdown
        # -------------------------------------------------------------

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

    # =================================================================
    # TABLE CELL CLEANING
    # =================================================================

    @staticmethod
    def _clean_table_cell(
        value,
        ) -> str:
        """
        Normalize a single table cell.
        """

        if value is None:
            return ""

        text = str(value)

        text = " ".join(
            text.split()
        )

        text = text.replace(
            "|",
            "\\|",
        )

        return text


    @staticmethod
    def _looks_like_no_result(
        text: str,
        ) -> bool:
        """
        Detect an empty or standard grounded-retrieval refusal.

        No document/domain-specific terms are used.
        """
        normalized = " ".join(
            (text or "").lower().split()
        )

        if not normalized:
            return True

        refusal_phrases = (
            "i couldn't find relevant information",
            "i could not find relevant information",
        )

        return any(
            phrase in normalized
            for phrase in refusal_phrases
        )

    # =================================================================
    # BEST RESULT SELECTION
    # =================================================================

    @staticmethod
    def _select_best_chunk(
        results: list[Any],
        ) -> Any | None:
        """
        Select the highest-ranked usable text result.

        Qdrant returns results ordered by relevance, so the first
        usable text chunk is preferred.

        Image/table chunks are skipped for normal factual responses.
        """

        usable = []
        for chunk in results:
            chunk_type = (
                getattr(
                    chunk,
                    "chunk_type",
                    None,
                )
                or "text"
            ).lower()

            if chunk_type in {
                "image",
                "table",
            }:
                continue

            text = (
                getattr(
                    chunk,
                    "text",
                    "",
                )
                or ""
            ).strip()

            if text:
                usable.append(chunk)

        if not usable:
            return None

        best = usable[0]
        best_text = str(getattr(best, "text", "") or "").strip()
        if len(best_text) < 100:
            same_page = [
                chunk for chunk in usable[:8]
                if getattr(chunk, "document_name", None) == getattr(best, "document_name", None)
                and getattr(chunk, "page_number", None) == getattr(best, "page_number", None)
                and len(str(getattr(chunk, "text", "") or "").strip()) > len(best_text)
            ]
            if same_page:
                return max(same_page, key=lambda chunk: len(str(getattr(chunk, "text", "") or "")))

        return best

    # =================================================================
    # RESPONSE TEXT CLEANING
    # =================================================================

    @staticmethod
    def _clean_response_text(
        text: Any,
        ) -> str:
        """
        Clean retrieved text before showing it to the user.

        This does NOT rewrite or summarize document content.
        """

        if text is None:
            return ""

        text = str(text).strip()

        if not text:
            return ""

        # Normalize excessive blank lines.
        lines = [
            line.rstrip()
            for line in text.splitlines()
        ]

        cleaned_lines = []

        previous_blank = False

        for line in lines:
            if not line.strip():
                if previous_blank:
                    continue

                cleaned_lines.append("")
                previous_blank = True
                continue

            cleaned_lines.append(line)
            previous_blank = False

        return "\n".join(
            cleaned_lines
        ).strip()

    # =================================================================
    # CITATIONS
    # =================================================================

    @staticmethod
    def _build_citation(
        chunk,
        project_name: str | None,
        ) -> dict:
        """
        Build a clean user-facing citation.

        Internal database IDs and filesystem paths are excluded.
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
        results: list[Any],
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
            citations.append(citation)

        return citations

    # =================================================================
    # NO RESULTS
    # =================================================================

    def _no_results(
        self,
        session_id: str | None,
        original_question: str,
        ) -> dict:
        """
        Safe response when retrieval produced no usable evidence.
        """

        return {
            "success": True,
            "response": self.NO_RESULTS_MESSAGE,
            "citations": [],
            "session_id": session_id,
        }

    # =================================================================
    # IMAGE HELPERS
    # =================================================================

    @staticmethod
    def _extract_image_name(
        image_path: str | None,
        image_id: str | None,
        ) -> str | None:
        """
        Extract ONLY the public image filename.

        Handles:

            storage\\images\\img_abc.png
            storage/images/img_abc.png
            img_abc.png

        Never returns the internal directory path.
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

        candidate = candidate.replace(
            "\\",
            "/",
        )

        filename = candidate.rsplit(
            "/",
            1,
        )[-1]

        return (
            filename
            if filename
            else None
        )
