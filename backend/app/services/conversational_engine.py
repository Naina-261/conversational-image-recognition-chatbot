"""
Conversational Engine Facade & Re-export Module.
Modularized into app.services.context_manager and app.services.fusion_engine.
Maintains full backward compatibility for existing callers and tests.
"""
from app.services.context_manager import (
    QuestionIntent,
    DialogueTurn,
    CLASS_SYNONYMS,
    CATEGORY_GROUPS,
    INVENTORY_TRIGGERS,
    COUNT_TRIGGERS,
    SPATIAL_TRIGGERS,
    ATTRIBUTE_TRIGGERS,
    ConversationalContextManager,
    ContextManager,
    get_context_manager,
)
from app.services.fusion_engine import (
    MultiModalFusionEngine,
    FusionEngine,
    get_fusion_engine,
)

__all__ = [
    "QuestionIntent",
    "DialogueTurn",
    "CLASS_SYNONYMS",
    "CATEGORY_GROUPS",
    "INVENTORY_TRIGGERS",
    "COUNT_TRIGGERS",
    "SPATIAL_TRIGGERS",
    "ATTRIBUTE_TRIGGERS",
    "ConversationalContextManager",
    "ContextManager",
    "get_context_manager",
    "MultiModalFusionEngine",
    "FusionEngine",
    "get_fusion_engine",
]
