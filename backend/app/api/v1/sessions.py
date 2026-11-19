from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID
from typing import List, Optional

from app.core.database import get_db
from app.services.session_service import get_session_service, SessionService
from app.schemas.session import (
    SessionCreate,
    SessionResponse,
    SessionDetailResponse,
    SessionClearResponse,
)

router = APIRouter()


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    payload: Optional[SessionCreate] = None,
    db: AsyncSession = Depends(get_db),
    session_service: SessionService = Depends(get_session_service),
):
    """Creates a new conversation session."""
    title = payload.title if payload else None
    return await session_service.create_session(db, title=title)


@router.get("", response_model=List[SessionResponse])
async def list_sessions(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    session_service: SessionService = Depends(get_session_service),
):
    """Lists recent conversation sessions with pagination."""
    return await session_service.list_sessions(db, limit=limit, offset=offset)


@router.get("/{session_id}", response_model=SessionDetailResponse)
async def get_session_detail(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
    session_service: SessionService = Depends(get_session_service),
):
    """Retrieves full session detail including message dialogue turns and uploaded images."""
    return await session_service.get_session_detail(db, session_id=session_id)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
    session_service: SessionService = Depends(get_session_service),
):
    """Permanently deletes a session and cascades associated artifacts."""
    await session_service.delete_session(db, session_id=session_id)
    return None


@router.delete("/{session_id}/history", response_model=SessionClearResponse, status_code=status.HTTP_200_OK)
async def clear_session_history(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
    session_service: SessionService = Depends(get_session_service),
):
    """
    Clears all dialogue messages from the specified conversation session
    while preserving the session record, uploaded images, and object detections.
    """
    return await session_service.clear_session_history(db, session_id=session_id)
