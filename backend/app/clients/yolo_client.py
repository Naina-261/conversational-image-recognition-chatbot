import logging
import asyncio
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
from uuid import uuid4

from app.core.config import get_settings

logger = logging.getLogger("bel_chatbot.yolo_client")
settings = get_settings()


class YOLOClient:
    """
    Ultralytics YOLOv8 Deep-Learning Object Detection Engine.

    Features:
    - Supports Apple Silicon MPS acceleration, CUDA, and single-threaded CPU execution.
    - Lazy model weight loading to prevent cold-start overheads in memory-constrained environments.
    - Normalizes coordinates to [0, 1] and rescales to original image dimensions.
    - Resolves spatial sectors (e.g. 'top-left', 'center', 'bottom-right').
    - NMS IoU & containment filtering to suppress duplicate nested bounding boxes.
    - Concurrency-controlled async execution to prevent concurrent memory spikes.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.YOLO_MODEL_NAME
        self.device = settings.INFERENCE_DEVICE
        self.model = None
        self._is_loaded = False
        self._lock: Optional[asyncio.Lock] = None

    @property
    def lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def load_model(self) -> None:
        """Loads YOLO model weights and applies device-specific optimizations."""
        if self._is_loaded and self.model is not None:
            return

        logger.info(f"Loading YOLOv8 weights ({self.model_name}) on device: {self.device.upper()}...")
        try:
            import torch
            try:
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
            except Exception:
                pass

            from ultralytics import YOLO

            self.model = YOLO(self.model_name)
            try:
                self.model.fuse()
            except Exception:
                pass

            if self.device == "mps" and not torch.backends.mps.is_available():
                logger.warning("MPS requested but not available. Falling back to CPU.")
                self.device = "cpu"

            self._is_loaded = True
            logger.info(f"YOLOv8 ({self.model_name}) successfully loaded on {self.device.upper()}.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model weights: {e}")
            raise e

    def is_ready(self) -> bool:
        """Returns True if the underlying model weights are loaded and ready for inference."""
        return self._is_loaded and self.model is not None

    @staticmethod
    def calculate_spatial_position(
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
        img_w: int,
        img_h: int,
    ) -> str:
        """Determines intuitive spatial sector tags (e.g., 'top-left', 'center') within the image."""
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0

        horiz = "left" if cx < img_w * 0.38 else "right" if cx > img_w * 0.62 else "center"
        vert = "top" if cy < img_h * 0.38 else "bottom" if cy > img_h * 0.62 else "middle"

        if horiz == "center" and vert == "middle":
            return "center"
        return f"{vert}-{horiz}"

    @staticmethod
    def suppress_duplicate_nested_boxes(
        detections: List[Dict[str, Any]],
        iou_thresh: float = 0.45,
        containment_thresh: float = 0.70,
    ) -> List[Dict[str, Any]]:
        """Suppresses redundant duplicate bounding boxes and nested sub-parts of the same object class."""
        if len(detections) <= 1:
            return detections

        sorted_dets = sorted(detections, key=lambda d: d.get("confidence", 0), reverse=True)
        kept = []
        for d in sorted_dets:
            b = d.get("bbox", {})
            w = max(0.0, b.get("x_max", 0) - b.get("x_min", 0))
            h = max(0.0, b.get("y_max", 0) - b.get("y_min", 0))
            area = w * h
            if area <= 0:
                continue

            duplicate = False
            for k in kept:
                if k.get("class_name") != d.get("class_name"):
                    continue
                kb = k.get("bbox", {})
                kw = max(0.0, kb.get("x_max", 0) - kb.get("x_min", 0))
                kh = max(0.0, kb.get("y_max", 0) - kb.get("y_min", 0))
                karea = kw * kh
                if karea <= 0:
                    continue

                inter_w = max(
                    0.0,
                    min(b.get("x_max", 0), kb.get("x_max", 0)) - max(b.get("x_min", 0), kb.get("x_min", 0)),
                )
                inter_h = max(
                    0.0,
                    min(b.get("y_max", 0), kb.get("y_max", 0)) - max(b.get("y_min", 0), kb.get("y_min", 0)),
                )
                inter_area = inter_w * inter_h

                if inter_area <= 0:
                    continue

                iou = inter_area / (area + karea - inter_area)
                containment = inter_area / min(area, karea)

                if iou >= iou_thresh or containment >= containment_thresh:
                    duplicate = True
                    break

            if not duplicate:
                kept.append(d)

        return kept

    def detect_objects(
        self,
        image_source: Any,
        conf_threshold: float = 0.35,
        classes_filter: Optional[List[str]] = None,
        orig_dimensions: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """
        Executes deep-learning object detection on an image.
        Returns detected classes, confidence scores, bounding boxes, normalized coordinates,
        spatial sector tags, and latency telemetry.
        """
        self.load_model()
        import time
        try:
            import torch
            inference_ctx = torch.inference_mode()
        except ImportError:
            from contextlib import nullcontext
            inference_ctx = nullcontext()

        start_time = time.perf_counter()
        with inference_ctx:
            results = self.model.predict(
                source=image_source,
                conf=conf_threshold,
                iou=0.45,
                imgsz=640,
                device=self.device,
                verbose=False,
            )
        latency_ms = (time.perf_counter() - start_time) * 1000

        detections = []
        result = results[0]
        boxes = result.boxes
        orig_shape = result.orig_shape
        img_h, img_w = orig_shape[0], orig_shape[1]

        target_w = int(orig_dimensions[0]) if orig_dimensions and orig_dimensions[0] > 0 else img_w
        target_h = int(orig_dimensions[1]) if orig_dimensions and orig_dimensions[1] > 0 else img_h

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                raw_cls = box.cls[0]
                cls_id = int(raw_cls.item()) if hasattr(raw_cls, "item") else int(raw_cls)
                class_name = result.names[cls_id]

                if classes_filter and class_name not in classes_filter:
                    continue

                raw_conf = box.conf[0]
                conf = float(raw_conf.item()) if hasattr(raw_conf, "item") else float(raw_conf)
                raw_xyxy = box.xyxy[0]
                xyxy = raw_xyxy.tolist() if hasattr(raw_xyxy, "tolist") else list(raw_xyxy)

                src_x_min, src_y_min, src_x_max, src_y_max = xyxy[0], xyxy[1], xyxy[2], xyxy[3]

                norm_x_min = max(0.0, min(1.0, src_x_min / img_w)) if img_w > 0 else 0.0
                norm_y_min = max(0.0, min(1.0, src_y_min / img_h)) if img_h > 0 else 0.0
                norm_x_max = max(0.0, min(1.0, src_x_max / img_w)) if img_w > 0 else 0.0
                norm_y_max = max(0.0, min(1.0, src_y_max / img_h)) if img_h > 0 else 0.0

                target_x_min = norm_x_min * target_w
                target_y_min = norm_y_min * target_h
                target_x_max = norm_x_max * target_w
                target_y_max = norm_y_max * target_h

                bbox_w = target_x_max - target_x_min
                bbox_h = target_y_max - target_y_min
                area_ratio = (
                    round((bbox_w * bbox_h) / (target_w * target_h), 4)
                    if (target_w * target_h) > 0
                    else 0.0
                )

                spatial_sector = self.calculate_spatial_position(
                    target_x_min, target_y_min, target_x_max, target_y_max, target_w, target_h
                )

                detections.append({
                    "id": str(uuid4()),
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": round(conf, 4),
                    "confidence_percentage": round(conf * 100, 1),
                    "spatial_sector": spatial_sector,
                    "area_ratio": area_ratio,
                    "bbox": {
                        "x_min": round(target_x_min, 2),
                        "y_min": round(target_y_min, 2),
                        "x_max": round(target_x_max, 2),
                        "y_max": round(target_y_max, 2),
                    },
                    "normalized_bbox": {
                        "x_min": round(norm_x_min, 4),
                        "y_min": round(norm_y_min, 4),
                        "x_max": round(norm_x_max, 4),
                        "y_max": round(norm_y_max, 4),
                    },
                })

        detections = self.suppress_duplicate_nested_boxes(detections)

        logger.info(
            f"YOLO detected {len(detections)} objects on {self.device.upper()} in {latency_ms:.2f}ms"
        )

        from app.utils.memory import trim_process_memory
        trim_process_memory()

        return {
            "detections": detections,
            "count": len(detections),
            "image_dimensions": {"width": target_w, "height": target_h},
            "latency_ms": round(latency_ms, 2),
            "device": self.device,
            "model": self.model_name,
        }

    async def detect_objects_async(
        self,
        image_source: Any,
        conf_threshold: float = 0.35,
        classes_filter: Optional[List[str]] = None,
        orig_dimensions: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """Concurrency-controlled async wrapper for YOLO detection."""
        async with self.lock:
            return await asyncio.to_thread(
                self.detect_objects,
                image_source=image_source,
                conf_threshold=conf_threshold,
                classes_filter=classes_filter,
                orig_dimensions=orig_dimensions,
            )


_yolo_instance: Optional[YOLOClient] = None


def get_yolo_client() -> YOLOClient:
    """Singleton accessor for YOLOClient."""
    global _yolo_instance
    if _yolo_instance is None:
        _yolo_instance = YOLOClient()
    return _yolo_instance
