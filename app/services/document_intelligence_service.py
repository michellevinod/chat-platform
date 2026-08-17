from app.agents.followup_resolver import FollowupResolver
from app.agents.query_classifier import QueryClassifier, QueryIntent
from app.agents.response_builder import ResponseBuilder
from app.rag.citation_service import CitationService
from app.rag.rag_tool import RAGTool
from app.rag.retrieved_chunk import RetrievedChunk
from app.services.context_builder import ContextBuilder
from app.services.decision_service import DecisionService
from app.services.llm_service import LLMService
from app.services.memory_service import MemoryService


class DocumentIntelligenceService:
    """
    Main Document Intelligence Service.
    Flow:
    User Question -> Session & Followup Resolution -> Intent Classification ->
    Document Ambiguity Check -> Scoped Qdrant Retrieval -> Relevance Filtering ->
    Direct Markdown / Table Response or Gemini Synthesis (only when needed) ->
    Clean Citations -> Session Memory Update.
    """

    MINIMUM_SCORE = 0.40

    def __init__(self):
        self._rag = RAGTool()
        self._llm = LLMService()
        self._decision = DecisionService()
        self._classifier = QueryClassifier()
        self._context_builder = ContextBuilder()
        self._citation_service = CitationService()
        self._followup_resolver = FollowupResolver()
        self._response_builder = ResponseBuilder()
        self._memory = MemoryService()

    def answer(
        self,
        question: str,
        project_name: str | None = None,
        document_name: str | None = None,
        session_id: str | None = None,
    ) -> dict:

        session_id = self._memory.get_or_create_session(session_id)
        original_question = question.strip()

        if not original_question:
            return {
                "success": False,
                "response": "Query cannot be empty.",
                "citations": [],
                "session_id": session_id,
            }

        # 1. Resolve follow-ups from conversation memory
        effective_query, resolved_doc, resolved_proj = self._memory.resolve_query(
            session_id=session_id,
            query=original_question,
            explicit_doc=document_name,
            explicit_proj=project_name,
        )

        project_name = resolved_proj
        document_name = resolved_doc

        # 2. Greeting check
        if self._decision.is_greeting(effective_query):
            response_text = "Hello! Upload one or more documents and ask me anything related to them."
            self._memory.save_turn(session_id, original_question, response_text)
            return {
                "success": True,
                "response": response_text,
                "citations": [],
                "session_id": session_id,
            }


        # 3. Out-of-scope check
        if self._decision.is_out_of_scope(effective_query):
            response_text = "I can answer only from uploaded documents. Please upload a document and ask questions related to it."
            self._memory.save_turn(session_id, original_question, response_text)
            return {
                "success": True,
                "response": response_text,
                "citations": [],
                "session_id": session_id,
            }

        # 4. Extract document mention from query if not explicitly set (unless comparing multiple docs)
        is_comparison = any(w in effective_query.lower() for w in ["compare", "comparison", "difference between", "versus", " vs "])
        available_docs = self._rag.get_distinct_documents(project_name=project_name)
        if not document_name and not is_comparison:
            mentioned_doc = self._followup_resolver.extract_document_mention(
                query=effective_query,
                available_documents=available_docs,
            )
            if mentioned_doc:
                document_name = mentioned_doc

        # 5. Check for ambiguous document summary request
        if self._followup_resolver.is_ambiguous_summary(
            query=effective_query,
            selected_doc=document_name,
            available_documents=available_docs,
        ):
            self._memory.set_pending_prompt(session_id, "ask_document_summary")
            response_text = "Which document would you like me to summarize?"
            self._memory.save_turn(session_id, original_question, response_text, project=project_name)
            return {
                "success": True,
                "response": response_text,
                "citations": [],
                "session_id": session_id,
            }


        # 6. Intent classification
        intent = self._classifier.classify(effective_query)

        # 7. Retrieve from Qdrant
        limit = 12 if intent in [QueryIntent.DOCUMENT_SUMMARY, QueryIntent.PROJECT_SUMMARY, QueryIntent.RAG_SYNTHESIS] else 6

        # Determine chunk_type filter if user specifically asked for table/image
        chunk_type_filter = None
        if intent == QueryIntent.SEARCH_TABLE:
            chunk_type_filter = "table"
        elif intent == QueryIntent.SEARCH_IMAGE:
            chunk_type_filter = "image"

        search_query = effective_query
        if intent == QueryIntent.DOCUMENT_SUMMARY and document_name:
            search_query = f"{effective_query} summary overview purpose introduction report findings"
        elif intent == QueryIntent.PROJECT_SUMMARY and project_name:
            search_query = f"{effective_query} project overview summary status objectives"

        results = self._rag.search(
            query=search_query,
            limit=limit,
            project_name=project_name,
            document_name=document_name,
            chunk_type=chunk_type_filter,
        )

        # If table/image search with filter gave nothing, fallback to general search
        if not results and chunk_type_filter:
            results = self._rag.search(
                query=search_query,
                limit=limit,
                project_name=project_name,
                document_name=document_name,
            )

        # 8. Check relevance & confidence
        if not results:
            return self._no_results(session_id, original_question)

        # Threshold: if document/project is explicitly selected, relax threshold for summaries/lookups within that document
        effective_threshold = 0.20 if (document_name or project_name) else self.MINIMUM_SCORE
        top_score = max(r.score for r in results)
        if top_score < effective_threshold:
            return self._no_results(session_id, original_question)

        # Filter out chunks far below threshold
        relevant_chunks = [r for r in results if r.score >= (effective_threshold - 0.05)]
        if not relevant_chunks:
            relevant_chunks = results[:3]


        # 9. Build citations
        citations = self._citation_service.build_citations(relevant_chunks)
        citations_payload = [c.model_dump() for c in citations]

        # 10. Generate response
        # Case A: Table response
        if intent == QueryIntent.SEARCH_TABLE or (relevant_chunks[0].chunk_type == "table" and not self._decision.should_use_llm(effective_query)):
            response_text = self._response_builder.build_table_response(relevant_chunks)

        # Case B: LLM Synthesis (summary, comparison, multi-hop reasoning)
        elif self._decision.should_use_llm(effective_query) or intent in [
            QueryIntent.DOCUMENT_SUMMARY,
            QueryIntent.PROJECT_SUMMARY,
            QueryIntent.RAG_SYNTHESIS,
        ]:
            context = self._context_builder.build(relevant_chunks)
            response_text = self._llm.generate_answer(
                question=effective_query,
                context=context,
            )

        # Case C: Factual Lookup (RAG only - no LLM call)
        else:
            response_text = self._response_builder.build_factual_response(
                query=effective_query,
                chunks=relevant_chunks,
            )

        # Save turn to conversation memory
        resolved_doc_final = relevant_chunks[0].document_name if relevant_chunks else document_name
        resolved_proj_final = relevant_chunks[0].project_name if relevant_chunks else project_name
        self._memory.save_turn(
            session_id=session_id,
            query=original_question,
            response=response_text,
            project=resolved_proj_final,
            document=resolved_doc_final,
            citations=citations_payload,
        )

        return {
            "success": True,
            "response": response_text,
            "citations": citations_payload,
            "session_id": session_id,
        }

    def _no_results(self, session_id: str, original_question: str) -> dict:
        response_text = "I couldn't find relevant information in the uploaded documents."
        self._memory.save_turn(session_id, original_question, response_text)
        return {
            "success": True,
            "response": response_text,
            "citations": [],
            "session_id": session_id,
        }