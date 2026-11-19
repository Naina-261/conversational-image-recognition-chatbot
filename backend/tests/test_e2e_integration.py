import pytest
import io
from PIL import Image
from httpx import AsyncClient, ASGITransport
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.main import app


def generate_recon_sample_image() -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (640, 480), color=(70, 90, 60))
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_full_multimodal_e2e_pipeline(monkeypatch):
    transport = ASGITransport(app=app)

    # Mock AI clients (YOLO and VQA) to ensure isolated, deterministic testing without live APIs
    async def mock_detect(*args, **kwargs):
        return {
            "detections": [
                {
                    "id": "e2e00000-0000-0000-0000-000000000001",
                    "class_id": 1,
                    "class_name": "military vehicle",
                    "confidence": 0.95,
                    "confidence_percentage": 95.0,
                    "spatial_sector": "center",
                    "area_ratio": 0.25,
                    "bbox": {"x_min": 150.0, "y_min": 120.0, "x_max": 490.0, "y_max": 380.0},
                    "normalized_bbox": {"x_min": 0.234, "y_min": 0.25, "x_max": 0.765, "y_max": 0.791},
                }
            ],
            "count": 1,
            "image_dimensions": {"width": 640, "height": 480},
            "latency_ms": 12.0,
            "device": "cpu",
            "model": "yolov8n.pt",
        }

    def mock_vqa(*args, **kwargs):
        return {
            "answer": "military vehicle",
            "model": "gemini-2.0-flash",
            "latency_ms": 25.0,
            "error": None,
        }

    monkeypatch.setattr("app.clients.yolo_client.YOLOClient.detect_objects_async", mock_detect)
    monkeypatch.setattr("app.clients.vqa_client.VQAClient.answer_question", mock_vqa)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. Create a new conversation session
            session_res = await client.post("/api/v1/sessions", json={"title": "BEL Tactical E2E Surveillance"})
            assert session_res.status_code == 201
            session_id = session_res.json()["id"]

            # 2. Upload an image to the session (triggers real YOLO detection pass)
            img_bytes = generate_recon_sample_image()
            files = {"file": ("tactical_surveillance.jpg", img_bytes, "image/jpeg")}
            upload_res = await client.post(
                "/api/v1/images/upload",
                files=files,
                data={"session_id": session_id},
            )
            assert upload_res.status_code == 201
            upload_data = upload_res.json()
            image_id = upload_data["id"]
            assert upload_data["session_id"] == session_id
            assert upload_data["width"] == 640
            assert upload_data["height"] == 480

            # 3. First dialogue turn: Ask about objects
            q1_res = await client.post(
                "/api/v1/chat/message",
                json={
                    "session_id": session_id,
                    "image_id": image_id,
                    "content": "What objects or equipment do you observe in this image?",
                },
            )
            assert q1_res.status_code == 200
            q1_data = q1_res.json()
            assert q1_data["sender_type"] == "assistant"
            assert len(q1_data["content"]) > 0

            # 4. Second dialogue turn: Multi-turn follow-up with pronoun resolution
            q2_res = await client.post(
                "/api/v1/chat/message",
                json={
                    "session_id": session_id,
                    "image_id": image_id,
                    "content": "Where is it located?",
                },
            )
            assert q2_res.status_code == 200
            q2_data = q2_res.json()
            assert q2_data["sender_type"] == "assistant"

            # 5. Fetch full session details with messages and images
            detail_res = await client.get(f"/api/v1/sessions/{session_id}")
            assert detail_res.status_code == 200
            detail = detail_res.json()
            assert len(detail["messages"]) >= 4  # 2 user turns + 2 assistant turns
            assert len(detail["images"]) >= 1

            # 6. Cleanup
            del_res = await client.delete(f"/api/v1/sessions/{session_id}")
            assert del_res.status_code == 204
