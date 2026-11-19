import sys
import logging
from pathlib import Path
from typing import AsyncGenerator
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.core.config import get_settings
from app.core.database import Base, engine, get_db, AsyncSessionLocal
import app.models  # ensure models are registered
from app.main import app

logger = logging.getLogger("bel_chatbot.tests")


@pytest_asyncio.fixture(scope="function", autouse=True)
async def setup_test_database():
    """
    Creates all database tables before the test session begins,
    and cleans them up after the session ends.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture(scope="function")
async def client() -> AsyncGenerator[AsyncClient, None]:
    """
    FastAPI HTTP test client for making API requests.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_yolo_detections():
    """Mock detection payload returned by YOLO client."""
    return {
        "detections": [
            {
                "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "class_id": 2,
                "class_name": "car",
                "confidence": 0.942,
                "confidence_percentage": 94.2,
                "spatial_sector": "center",
                "area_ratio": 0.15,
                "bbox": {"x_min": 100.0, "y_min": 150.0, "x_max": 300.0, "y_max": 350.0},
                "normalized_bbox": {"x_min": 0.156, "y_min": 0.312, "x_max": 0.468, "y_max": 0.729},
            }
        ],
        "count": 1,
        "image_dimensions": {"width": 640, "height": 480},
        "latency_ms": 12.5,
        "device": "cpu",
        "model": "yolov8n.pt",
    }
