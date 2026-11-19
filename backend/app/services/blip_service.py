"""
Visual Question Answering Service wrapper.
Delegates to app.clients.vqa_client.VQAClient with exponential backoff and multi-tier fallbacks.
"""
from app.clients.vqa_client import VQAClient, get_vqa_client

# Maintain backward-compatible aliases
BLIPService = VQAClient
get_blip_service = get_vqa_client

__all__ = ["BLIPService", "get_blip_service", "VQAClient", "get_vqa_client"]
