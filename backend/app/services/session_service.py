from typing import List, Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.repositories.session_repo import SessionRepository
from app.repositories.message_repo import MessageRepository
from app.models.session import SessionModel
from app.schemas.session import SessionClearResponse
from app.core.exceptions import EntityNotFoundError

logger = logging.getLogger("bel_chatbot.session_service")


class SessionService:
    """Service layer managing conversation session lifecycle and dialogue memory."""

    @staticmethod
    async def create_session(
        db: AsyncSession,
        title: Optional[str] = None,
        session_id: Optional[UUID] = None,
    ) -> SessionModel:
        clean_title = title.strip() if title and title.strip() else "New Conversation"
        session = await SessionRepository.create(db, title=clean_title, session_id=session_id)
        logger.info(f"Created session {session.id} ('{session.title}')")
        return session

    @staticmethod
    async def list_sessions(
        db: AsyncSession,
        limit: int = 50,
        offset: int = 0,
    ) -> List[SessionModel]:
        return await SessionRepository.list_recent(db, limit=limit, offset=offset)

    @staticmethod
    async def get_session_detail(
        db: AsyncSession,
        session_id: UUID,
    ) -> SessionModel:
        session = await SessionRepository.get_by_id(db, session_id, load_relations=True)
        if not session:
            raise EntityNotFoundError(f"Session {session_id} not found.")
        return session

    @staticmethod
    async def delete_session(
        db: AsyncSession,
        session_id: UUID,
    ) -> None:
        session = await SessionRepository.get_by_id(db, session_id, load_relations=False)
        if not session:
            raise EntityNotFoundError(f"Session {session_id} not found.")
        await SessionRepository.delete(db, session)
        logger.info(f"Deleted session {session_id}")

    @staticmethod
    async def clear_session_history(
        db: AsyncSession,
        session_id: UUID,
    ) -> SessionClearResponse:
        """
        Clears conversation history (messages) for a session while preserving
        the session itself and any associated imagery or object telemetry.
        """
        session = await SessionRepository.get_by_id(db, session_id, load_relations=False)
        if not session:
            raise EntityNotFoundError(f"Session {session_id} not found.")
        cleared_count = await MessageRepository.delete_by_session(db, session_id)
        logger.info(f"Cleared {cleared_count} messages for session {session_id}")
        return SessionClearResponse(
            session_id=session_id,
            cleared_messages_count=cleared_count,
        )


_session_service_instance: Optional[SessionService] = None


def get_session_service() -> SessionService:
    global _session_service_instance
    if _session_service_instance is None:
        _session_service_instance = SessionService()
    return _session_service_instance
