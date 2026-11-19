"""
Configuration re-export module for backward compatibility.
Centralized settings have moved to app.core.config.
"""
from app.core.config import Settings, get_settings

__all__ = ["Settings", "get_settings"]
