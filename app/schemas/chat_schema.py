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


class Citation(BaseModel):
    """
    Citation returned with the response.
    """

    document: str
    page: int


class ChatResponse(BaseModel):
    """
    Chat response payload.
    """

    success: bool
    response: str
    citations: list[Citation] = Field(default_factory=list)