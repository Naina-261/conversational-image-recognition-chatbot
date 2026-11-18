import time
import asyncio
import logging
from typing import Optional, List, Dict, Any
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.repositories.session_repo import SessionRepository
from app.repositories.image_repo import ImageRepository
from app.repositories.message_repo import MessageRepository
from app.services.context_manager import get_context_manager, QuestionIntent
from app.services.fusion_engine import get_fusion_engine
from app.clients.vqa_client import get_vqa_client, VQAClient
from app.clients.yolo_client import YOLOClient
from app.services.color_analyzer import get_color_analyzer
from app.schemas.chat import ChatMessageRequest, MessageResponse

logger = logging.getLogger("bel_chatbot.chat_service")
settings = get_settings()


class ChatService:
    """
    Chat Service Layer.
    Encapsulates multimodal conversational intelligence:
    - Session management & user message persistence
    - Multi-turn dialogue context and pronoun coreference resolution
    - Spatial telemetry extraction from YOLO detections
    - External VQA dispatch (Gemini/HF) with local CV color extraction fallback
    - Multimodal response synthesis via Fusion Engine
    - Assistant message persistence and response formatting
    """

    def __init__(
        self,
        vqa_client: Optional[VQAClient] = None,
        context_manager=None,
        fusion_engine=None,
    ):
        self.vqa_client = vqa_client or get_vqa_client()
        self.context_manager = context_manager or get_context_manager()
        self.fusion_engine = fusion_engine or get_fusion_engine()

    async def handle_chat_message(
        self,
        db: AsyncSession,
        payload: ChatMessageRequest,
    ) -> MessageResponse:
        """
        Executes end-to-end multimodal chat completion.
        """
        start_time = time.perf_counter()

        # 1. Fetch or create session
        session = await SessionRepository.get_by_id(db, payload.session_id, load_relations=True)
        if not session:
            session = await SessionRepository.create(
                db, title=payload.content[:40], session_id=payload.session_id
            )
            session_messages = []
        else:
            from sqlalchemy import inspect
            insp = inspect(session)
            session_messages = session.messages if "messages" in insp.dict else []

        # 2. Persist User Message
        await MessageRepository.create(
            db,
            session_id=payload.session_id,
            sender_type="user",
            content=payload.content,
            image_id=payload.image_id,
        )

        # 3. Context & Multimodal Processing
        history_turns = self.context_manager.build_history_window(session_messages)

        yolo_detections: List[Dict[str, Any]] = []
        vqa_answer = ""
        vqa_error = None
        vqa_latency = 0.0
        conf_threshold = (
            payload.confidence_threshold
            if payload.confidence_threshold is not None
            else settings.CONFIDENCE_THRESHOLD
        )

        # Resolve target image: explicit image_id or most recent session image
        target_image_id = payload.image_id
        if not target_image_id and session:
            from sqlalchemy import select
            from app.models.image import ImageModel
            img_query = (
                select(ImageModel.id)
                .where(ImageModel.session_id == session.id)
                .order_by(ImageModel.created_at.desc())
                .limit(1)
            )
            res = await db.execute(img_query)
            target_image_id = res.scalar_one_or_none()

        if target_image_id:
            image = await ImageRepository.get_by_id(db, target_image_id, load_objects=True)
            if image:
                img_w = image.width or 640
                img_h = image.height or 480
                yolo_detections = [
                    {
                        "class_name": o.class_name,
                        "confidence": o.confidence_score,
                        "spatial_sector": YOLOClient.calculate_spatial_position(
                            o.bbox_x_min, o.bbox_y_min, o.bbox_x_max, o.bbox_y_max, img_w, img_h
                        ),
                        "bbox": {
                            "x_min": o.bbox_x_min,
                            "y_min": o.bbox_y_min,
                            "x_max": o.bbox_x_max,
                            "y_max": o.bbox_y_max,
                        },
                    }
                    for o in image.detected_objects
                ]

                (
                    intent,
                    resolved_query,
                    target_class,
                    matched_classes,
                    spatial_direction,
                    explicit_threshold,
                    is_follow_up,
                ) = self.context_manager.classify_and_resolve(
                    payload.content, history_turns, yolo_detections
                )

                # Route to external VQA only for visual attributes/general queries
                if intent in (QuestionIntent.VISUAL_ATTRIBUTE, QuestionIntent.GENERAL):
                    vqa_res = await asyncio.to_thread(
                        self.vqa_client.answer_question, image.storage_path, resolved_query
                    )
                    vqa_answer = vqa_res.get("answer") or ""
                    vqa_error = vqa_res.get("error")
                    vqa_latency = vqa_res.get("latency_ms", 0.0)

                    # Fallback to local CV pixel color analyzer if VQA unconfigured/unavailable
                    is_color_q = any(
                        w in payload.content.lower()
                        for w in ["color", "colour", "shade", "paint", "tint"]
                    )
                    if is_color_q and target_class and not vqa_answer:
                        classes_to_match = set(matched_classes or [target_class])
                        matching_objs = [d for d in yolo_detections if d.get("class_name") in classes_to_match]
                        if matching_objs:
                            analyzer = get_color_analyzer()
                            color_res = analyzer.analyze_object_color_from_file(
                                image.storage_path,
                                matching_objs[0]["bbox"],
                                target_dimensions=(img_w, img_h),
                            )
                            if (
                                color_res
                                and color_res.get("color_phrase")
                                and color_res["color_phrase"] not in ("unknown", "indeterminate")
                            ):
                                vqa_answer = color_res["color_phrase"]
                                vqa_error = None
                            elif color_res.get("error"):
                                vqa_error = color_res["error"]

        # 4. Generate Fused Response
        fused = self.fusion_engine.generate_fused_response(
            query=payload.content,
            vqa_answer=vqa_answer,
            yolo_detections=yolo_detections,
            dialogue_history=history_turns,
            conf_threshold=conf_threshold,
            vqa_error=vqa_error,
        )

        total_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        meta_payload = {
            "vqa_mode": bool(vqa_answer),
            "vqa_answer": vqa_answer,
            "detected_classes": fused["detected_classes"],
            "primary_subject": fused.get("primary_subject"),
            "intent": fused.get("intent"),
            "latency_ms": total_latency_ms,
            "vqa_latency_ms": vqa_latency,
            "confidence_threshold": conf_threshold,
        }

        # 5. Persist Assistant Response
        assistant_msg = await MessageRepository.create(
            db,
            session_id=payload.session_id,
            sender_type="assistant",
            content=fused["content"],
            image_id=payload.image_id,
            metadata=meta_payload,
        )

        return MessageResponse(
            id=assistant_msg.id,
            session_id=assistant_msg.session_id,
            image_id=assistant_msg.image_id,
            sender_type="assistant",
            content=assistant_msg.content,
            metadata=meta_payload,
            created_at=assistant_msg.created_at,
        )


_chat_service_instance: Optional[ChatService] = None


def get_chat_service() -> ChatService:
    global _chat_service_instance
    if _chat_service_instance is None:
        _chat_service_instance = ChatService()
    return _chat_service_instance
