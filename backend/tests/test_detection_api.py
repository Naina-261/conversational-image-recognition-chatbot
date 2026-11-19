import pytest
import io
from PIL import Image
from httpx import AsyncClient, ASGITransport
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.main import app


def create_test_image(format="JPEG", size=(320, 240), color="green") -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_upload_and_detect_api(monkeypatch):
    transport = ASGITransport(app=app)

    async def mock_detect(*args, **kwargs):
        return {
            "detections": [
                {
                    "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                    "class_id": 2,
                    "class_name": "car",
                    "confidence": 0.95,
                    "confidence_percentage": 95.0,
                    "spatial_sector": "center",
                    "area_ratio": 0.15,
                    "bbox": {"x_min": 10.0, "y_min": 20.0, "x_max": 100.0, "y_max": 120.0},
                    "normalized_bbox": {"x_min": 0.03, "y_min": 0.08, "x_max": 0.31, "y_max": 0.5},
                }
            ],
            "count": 1,
            "image_dimensions": {"width": 320, "height": 240},
            "latency_ms": 1.5,
            "device": "cpu",
            "model": "mock",
        }

    monkeypatch.setattr("app.clients.yolo_client.YOLOClient.detect_objects_async", mock_detect)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Upload and detect
        img_bytes = create_test_image("JPEG")
        files = {"file": ("tactical_feed.jpg", img_bytes, "image/jpeg")}
        res = await client.post("/api/v1/images/upload", files=files)
        assert res.status_code == 201
        data = res.json()
        assert "detected_objects" in data
        assert isinstance(data["detected_objects"], list)
        image_id = data["id"]

        # 2. Re-run detection with threshold
        rerun_res = await client.post(f"/api/v1/images/{image_id}/detect?confidence_threshold=0.5")
        assert rerun_res.status_code == 200
        rerun_data = rerun_res.json()
        assert rerun_data["id"] == image_id
        assert "detected_objects" in rerun_data
