from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.chat_service import get_chat_service, ChatService
from app.schemas.chat import ChatMessageRequest, MessageResponse

router = APIRouter()


@router.post("/message", response_model=MessageResponse, status_code=status.HTTP_200_OK)
async def send_chat_message(
    payload: ChatMessageRequest,
    db: AsyncSession = Depends(get_db),
    chat_service: ChatService = Depends(get_chat_service),
):
    """
    Context-Aware Multimodal Chatbot Completion:
    1. Loads dialogue history from the session.
    2. Resolves pronouns & coreferences across dialogue turns.
    3. Runs VQA or extracts direct YOLO spatial telemetry.
    4. Fuses telemetry and dialogue history into natural response.
    5. Persists exchange and returns structured completion.
    """
    return await chat_service.handle_chat_message(db=db, payload=payload)
