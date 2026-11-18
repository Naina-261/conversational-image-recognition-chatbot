"""
YOLO Object Detection Service wrapper.
Delegates to app.clients.yolo_client.YOLOClient with MPS/CUDA acceleration and concurrency control.
"""
from app.clients.yolo_client import YOLOClient, get_yolo_client

# Maintain backward-compatible aliases
YOLOService = YOLOClient
get_yolo_service = get_yolo_client

__all__ = ["YOLOService", "get_yolo_service", "YOLOClient", "get_yolo_client"]
