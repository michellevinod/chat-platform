from fastapi import APIRouter

from app.schemas.chat_schema import (
    ChatRequest,
    ChatResponse,
)
from app.services.chat_service import ChatService


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)

_service = ChatService()


@router.post(
    "",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
) -> ChatResponse:
    """
    Chat with uploaded documents.
    """

    result = _service.chat(request.query)

    return ChatResponse(**result)