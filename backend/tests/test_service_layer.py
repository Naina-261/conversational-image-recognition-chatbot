import pytest
from uuid import uuid4
from PIL import Image
from unittest.mock import MagicMock
import io

from app.core.database import AsyncSessionLocal
from app.services.session_service import SessionService
from app.services.chat_service import ChatService
from app.services.image_service import ImageService
from app.schemas.chat import ChatMessageRequest
from app.core.exceptions import EntityNotFoundError


@pytest.mark.asyncio
async def test_session_service_crud_and_clear():
    async with AsyncSessionLocal() as db:
        # 1. Create
        session = await SessionService.create_session(db, title="Service Test Session")
        assert session.id is not None
        assert session.title == "Service Test Session"

        # 2. List
        sessions = await SessionService.list_sessions(db)
        assert any(s.id == session.id for s in sessions)

        # 3. Get detail
        detail = await SessionService.get_session_detail(db, session.id)
        assert detail.id == session.id

        # 4. Clear history (starts empty -> 0 cleared)
        clear_res = await SessionService.clear_session_history(db, session.id)
        assert clear_res.cleared_messages_count == 0

        # 5. Delete
        await SessionService.delete_session(db, session.id)

        # 6. Verify EntityNotFoundError
        with pytest.raises(EntityNotFoundError):
            await SessionService.get_session_detail(db, session.id)


@pytest.mark.asyncio
async def test_chat_service_multimodal_flow(monkeypatch):
    async with AsyncSessionLocal() as db:
        session = await SessionService.create_session(db, title="Chat Flow Test")

        chat_svc = ChatService()
        # Mock VQA client so test does not call external APIs
        mock_vqa = MagicMock()
        mock_vqa.answer_question.return_value = {
            "answer": "a blue car",
            "error": None,
            "latency_ms": 10.0,
            "device": "mock",
            "model": "mock",
        }
        monkeypatch.setattr(chat_svc, "vqa_client", mock_vqa)

        # 1. First turn: general query
        req1 = ChatMessageRequest(
            session_id=session.id,
            content="Hello assistant, what can you do?",
        )
        res1 = await chat_svc.handle_chat_message(db, req1)
        assert res1.sender_type == "assistant"
        assert len(res1.content) > 0

        # 2. Second turn: follow-up query
        req2 = ChatMessageRequest(
            session_id=session.id,
            content="Can you explain that again?",
        )
        res2 = await chat_svc.handle_chat_message(db, req2)
        assert res2.sender_type == "assistant"
        assert len(res2.content) > 0

        # Cleanup
        await SessionService.delete_session(db, session.id)


def test_image_service_missing_file_raises():
    with pytest.raises(EntityNotFoundError):
        ImageService.get_image_file_path("non_existent_file_99999.jpg")
