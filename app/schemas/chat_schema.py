from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    Chat request payload.
    """

    query: str = Field(
        ...,
        description="User question",
        min_length=1,
    )
    project_name: str | None = Field(
        default=None,
        description="Optional project filter",
    )
    document_name: str | None = Field(
        default=None,
        description="Optional document filter",
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session ID for conversational memory",
    )
    conversation_id: str | None = Field(
        default=None,
        description="Alias for session_id",
    )


class Citation(BaseModel):
    """
    Citation returned with the response.
    """

    project: str | None = None
    document: str
    page: int
    source: str | None = None
    chunk_type: str | None = None


class ChatResponse(BaseModel):
    """
    Chat response payload.
    """

    success: bool
    response: str
    citations: list[Citation] = Field(default_factory=list)
    session_id: str | None = None