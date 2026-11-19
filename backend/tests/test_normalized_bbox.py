from unittest.mock import MagicMock
import pytest
import sys
from pathlib import Path
from PIL import Image

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.services.yolo_service import YOLOService


def test_spatial_sector_calculation():
    # Top-left box
    pos = YOLOService.calculate_spatial_position(10, 10, 50, 50, 1000, 1000)
    assert pos == "top-left"

    # Bottom-right box
    pos_br = YOLOService.calculate_spatial_position(800, 800, 950, 950, 1000, 1000)
    assert pos_br == "bottom-right"

    # Center box
    pos_center = YOLOService.calculate_spatial_position(450, 450, 550, 550, 1000, 1000)
    assert pos_center == "center"


def test_normalized_bbox_generation(monkeypatch):
    service = YOLOService(model_name="yolov8n.pt")

    # Mock YOLO model predictions to test bounding box normalization without downloading weights
    mock_box = MagicMock()
    mock_box.xyxy = [[100.0, 150.0, 300.0, 400.0]]
    mock_box.conf = [0.92]
    mock_box.cls = [0]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_result.names = {0: "person"}
    mock_result.orig_shape = (480, 640)

    service.model = MagicMock()
    service.model.predict.return_value = [mock_result]
    service._is_loaded = True
    monkeypatch.setattr(service, "load_model", lambda: None)

    test_img = Image.new("RGB", (640, 480), color=(100, 150, 200))
    result = service.detect_objects(test_img)

    assert "image_dimensions" in result
    assert result["image_dimensions"]["width"] == 640
    assert result["image_dimensions"]["height"] == 480
    assert result["count"] == 1

    det = result["detections"][0]
    assert det["class_name"] == "person"
    assert det["confidence"] == 0.92
    assert "normalized_bbox" in det
    assert det["normalized_bbox"]["x_min"] == pytest.approx(100.0 / 640.0, rel=1e-3)
    assert det["normalized_bbox"]["y_min"] == pytest.approx(150.0 / 480.0, rel=1e-3)
    assert det["normalized_bbox"]["x_max"] == pytest.approx(300.0 / 640.0, rel=1e-3)
    assert det["normalized_bbox"]["y_max"] == pytest.approx(400.0 / 480.0, rel=1e-3)
    assert det["spatial_sector"] is not None
