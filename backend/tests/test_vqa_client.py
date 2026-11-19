import pytest
from PIL import Image
from unittest.mock import MagicMock
import requests

from app.clients.vqa_client import VQAClient
from app.clients.base import retry_with_backoff


def test_vqa_client_unconfigured(tmp_path):
    client = VQAClient()
    client.gemini_key = None
    client.hf_token = None
    assert client.is_configured() is False

    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (64, 64), color="blue").save(img_path)

    res = client.answer_question(img_path, "What color is this?")
    assert res["answer"] is None
    assert "currently unavailable" in res["error"].lower()
    assert "api key required" in res["error"].lower()


def test_vqa_client_gemini_success(tmp_path, monkeypatch):
    client = VQAClient()
    client.gemini_key = "mock_gemini_key"

    img_path = tmp_path / "gemini_test.jpg"
    Image.new("RGB", (64, 64), color="green").save(img_path)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "The object is green."}]
                }
            }
        ]
    }
    monkeypatch.setattr(client, "_execute_gemini_request", lambda url, payload: mock_resp)

    res = client.answer_question(img_path, "What color is the object?")
    assert res["answer"] == "The object is green."
    assert res["model"] == "gemini-2.0-flash"
    assert res["error"] is None


def test_vqa_client_hf_fallback_success(tmp_path, monkeypatch):
    client = VQAClient()
    client.gemini_key = None  # Force fallback to HF
    client.hf_token = "mock_hf_token"

    img_path = tmp_path / "hf_test.jpg"
    Image.new("RGB", (64, 64), color="yellow").save(img_path)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [{"answer": "yellow"}]
    monkeypatch.setattr(client, "_execute_hf_request", lambda endpoint, headers, payload: mock_resp)

    res = client.answer_question(img_path, "What color is this?")
    assert res["answer"] == "yellow"
    assert res["error"] is None


def test_retry_with_backoff_success_after_failure():
    call_count = 0

    @retry_with_backoff(max_retries=3, initial_delay=0.01, backoff_factor=1.5)
    def flaky_network_call():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            resp = requests.Response()
            resp.status_code = 429
            return resp
        resp = requests.Response()
        resp.status_code = 200
        return resp

    result = flaky_network_call()
    assert result.status_code == 200
    assert call_count == 2
