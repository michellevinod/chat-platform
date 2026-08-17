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

    result = _service.chat(
        query=request.query,
        project_name=request.project_name,
        document_name=request.document_name,
        session_id=request.session_id,
        conversation_id=request.conversation_id,
    )

    return ChatResponse(**result)