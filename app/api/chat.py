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

    session_id = request.session_id or request.conversation_id
    result = _service.chat(
        query=request.query,
        project_name=request.project_name,
        document_name=request.document_name,
        session_id=session_id,
    )

    return ChatResponse(**result)