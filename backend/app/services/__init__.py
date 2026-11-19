"""
Backend services package.
Encapsulates all domain and business logic away from HTTP controllers.
"""
from app.services.session_service import SessionService, get_session_service
from app.services.image_service import ImageService, get_image_service
from app.services.chat_service import ChatService, get_chat_service
from app.services.context_manager import ConversationalContextManager, ContextManager, get_context_manager
from app.services.fusion_engine import MultiModalFusionEngine, FusionEngine, get_fusion_engine
from app.services.color_analyzer import ObjectColorAnalyzer, get_color_analyzer

__all__ = [
    "SessionService",
    "get_session_service",
    "ImageService",
    "get_image_service",
    "ChatService",
    "get_chat_service",
    "ConversationalContextManager",
    "ContextManager",
    "get_context_manager",
    "MultiModalFusionEngine",
    "FusionEngine",
    "get_fusion_engine",
    "ObjectColorAnalyzer",
    "get_color_analyzer",
]
