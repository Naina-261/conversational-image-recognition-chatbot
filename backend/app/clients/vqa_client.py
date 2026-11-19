import time
import logging
import os
import base64
import io
import requests
from typing import Optional, Union, Dict, Any, List
from pathlib import Path
from PIL import Image

from app.core.config import get_settings
from app.clients.base import retry_with_backoff

logger = logging.getLogger("bel_chatbot.vqa_client")
settings = get_settings()


class VQAClient:
    """
    Resilient Visual Question Answering (VQA) Client.

    Features:
    - Never loads local heavy transformer weights into container RAM (guaranteeing 0 risk of 512MB RAM OOM-kills).
    - Tiered fallback architecture:
        1. Google Gemini API (preferred: fast, free tier, multimodal via gemini-2.0-flash)
        2. Hugging Face Serverless Inference API (fallback: Salesforce/blip-vqa-base)
    - Automatic exponential backoff retries on transient network errors and 429 rate limits.
    - Clear, non-hallucinating error reporting when external APIs are unconfigured or unavailable.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        timeout: float = 5.0,
        max_retries: int = 2,
    ):
        self.model_name = model_name or settings.VQA_MODEL_NAME
        self.timeout = timeout
        self.max_retries = max_retries

        # Credentials resolved from settings or environment
        self.hf_token = os.getenv("HF_TOKEN") or getattr(settings, "HF_TOKEN", None)
        self.gemini_key = os.getenv("GEMINI_API_KEY") or getattr(settings, "GEMINI_API_KEY", None)

        self.hf_endpoints = [
            f"https://router.huggingface.co/hf-inference/models/{self.model_name}",
            f"https://api-inference.huggingface.co/models/{self.model_name}",
        ]

    def is_configured(self) -> bool:
        """Returns True if any external VQA inference credentials are configured."""
        return bool(
            (self.gemini_key and self.gemini_key.strip())
            or (self.hf_token and self.hf_token.strip())
        )

    def _image_to_base64_jpeg(self, pil_image: Image.Image, max_dim: int = 640) -> str:
        """Downscale and serialize PIL image to optimized base64 JPEG."""
        img_copy = pil_image.copy()
        if max(img_copy.size) > max_dim:
            img_copy.thumbnail((max_dim, max_dim), Image.Resampling.BILINEAR)
        buf = io.BytesIO()
        img_copy.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        del img_copy
        buf.close()
        return b64

    @retry_with_backoff(max_retries=2, initial_delay=0.5, backoff_factor=2.0)
    def _execute_gemini_request(self, url: str, payload: dict) -> requests.Response:
        """Executes HTTP POST to Google Gemini API with retry-on-transient-failure."""
        return requests.post(
            url,
            json=payload,
            timeout=self.timeout,
            headers={"Content-Type": "application/json"},
        )

    def _query_gemini_api(self, pil_image: Image.Image, question: str) -> Optional[Dict[str, Any]]:
        """Queries Google Gemini 2.0 Flash for concise visual analysis."""
        if not self.gemini_key or not self.gemini_key.strip():
            return None

        img_b64 = self._image_to_base64_jpeg(pil_image)
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.0-flash:generateContent?key={self.gemini_key.strip()}"
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                "You are a visual analysis assistant for an object detection reconnaissance application. "
                                f"Answer the following question about this image concisely and accurately in 1-2 sentences. "
                                f"Question: {question}"
                            )
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": img_b64,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 150,
                "temperature": 0.2,
            },
        }

        start_time = time.perf_counter()
        try:
            res = self._execute_gemini_request(url, payload)
            latency_ms = (time.perf_counter() - start_time) * 1000

            if res.status_code == 200:
                data = res.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    answer = parts[0].get("text", "").strip() if parts else ""
                    if answer:
                        logger.info(
                            f"Gemini VQA answered '{question}' -> '{answer[:80]}...' ({latency_ms:.0f}ms)"
                        )
                        return {
                            "question": question,
                            "answer": answer,
                            "error": None,
                            "latency_ms": round(latency_ms, 2),
                            "device": "gemini_api",
                            "model": "gemini-2.0-flash",
                        }
            else:
                logger.warning(f"Gemini API returned HTTP {res.status_code}: {res.text[:200]}")
        except Exception as e:
            logger.warning(f"Gemini API execution error: {e}")

        return None

    @retry_with_backoff(max_retries=2, initial_delay=0.5, backoff_factor=2.0)
    def _execute_hf_request(self, endpoint: str, headers: dict, payload: dict) -> requests.Response:
        """Executes HTTP POST to Hugging Face Serverless API with retry-on-transient-failure."""
        return requests.post(endpoint, headers=headers, json=payload, timeout=self.timeout)

    def _query_hf_api(self, pil_image: Image.Image, question: str) -> Optional[Dict[str, Any]]:
        """Queries Hugging Face Serverless Inference API as fallback."""
        if not self.hf_token or not self.hf_token.strip():
            logger.info("External VQA (HF) skipped: HF_TOKEN not configured.")
            return None

        img_b64 = self._image_to_base64_jpeg(pil_image)
        headers = {
            "Authorization": f"Bearer {self.hf_token.strip()}",
            "Content-Type": "application/json",
        }
        payload = {"inputs": {"image": img_b64, "question": question}}

        start_time = time.perf_counter()
        for endpoint in self.hf_endpoints:
            try:
                res = self._execute_hf_request(endpoint, headers, payload)
                if res.status_code == 200:
                    data = res.json()
                    raw_answer = ""
                    if isinstance(data, list) and len(data) > 0:
                        raw_answer = data[0].get("answer", "")
                    elif isinstance(data, dict):
                        raw_answer = data.get("answer", "")

                    latency_ms = (time.perf_counter() - start_time) * 1000
                    logger.info(
                        f"HF VQA answered '{question}' -> '{raw_answer}' ({latency_ms:.1f}ms via {endpoint})"
                    )
                    return {
                        "question": question,
                        "answer": raw_answer.strip() if raw_answer else None,
                        "error": None,
                        "latency_ms": round(latency_ms, 2),
                        "device": "external_api",
                        "model": self.model_name,
                    }
                else:
                    logger.warning(f"HF endpoint {endpoint} returned HTTP {res.status_code}: {res.text[:200]}")
            except Exception as e:
                logger.warning(f"HF endpoint {endpoint} connection error: {e}")

        return None

    def answer_question(
        self,
        image: Union[Image.Image, str, Path],
        question: str,
    ) -> Dict[str, Any]:
        """
        Executes Visual Question Answering on an image and natural language query.
        Tries Google Gemini API first, then falls back to Hugging Face Serverless API.
        Returns clean, non-hallucinated response structure with execution metadata.
        """
        start_time = time.perf_counter()

        try:
            if isinstance(image, (str, Path)):
                with Image.open(str(image)) as img:
                    pil_image = img.convert("RGB")
            else:
                pil_image = image.convert("RGB")
        except Exception as e:
            return {
                "question": question,
                "answer": None,
                "error": f"Failed to load image for visual analysis: {str(e)}",
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
                "device": "external_api",
                "model": self.model_name,
            }

        # 1. Preferred: Google Gemini 2.0 Flash
        gemini_result = self._query_gemini_api(pil_image, question)
        if gemini_result and gemini_result.get("answer"):
            return gemini_result

        # 2. Fallback: Hugging Face Serverless Inference API
        hf_result = self._query_hf_api(pil_image, question)
        if hf_result and hf_result.get("answer"):
            return hf_result

        # 3. Unconfigured or service unavailable reporting
        latency_ms = (time.perf_counter() - start_time) * 1000
        configured_services: List[str] = []
        if self.gemini_key:
            configured_services.append("Gemini")
        if self.hf_token:
            configured_services.append("HuggingFace")

        if not configured_services:
            error_msg = (
                "Visual question answering is currently unavailable (API key required). "
                "Set GEMINI_API_KEY (free at aistudio.google.com) or HF_TOKEN in environment variables. "
                "You can still ask about detected objects, counts, and spatial positions."
            )
        else:
            error_msg = (
                f"External VQA services ({', '.join(configured_services)}) are temporarily unavailable or returned errors. "
                "You can still ask about detected objects, counts, and spatial positions."
            )

        logger.info(f"VQA unavailable for query '{question}': {error_msg}")
        return {
            "question": question,
            "answer": None,
            "error": error_msg,
            "latency_ms": round(latency_ms, 2),
            "device": "external_api",
            "model": self.model_name,
        }


_vqa_instance: Optional[VQAClient] = None


def get_vqa_client() -> VQAClient:
    """Singleton accessor for VQAClient."""
    global _vqa_instance
    if _vqa_instance is None:
        _vqa_instance = VQAClient()
    return _vqa_instance
