# Conversational Image Recognition Chatbot
**Enterprise Multi-Modal Conversational AI System**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch MPS](https://img.shields.io/badge/PyTorch-2.2.0%20(Apple%20MPS)-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00ffff.svg)](https://ultralytics.com)
[![React Vite](https://img.shields.io/badge/React%2018-Vite%205-61DAFB.svg?logo=react&logoColor=black)](https://vitejs.dev)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.4-38B2AC.svg?logo=tailwind-css&logoColor=white)](https://tailwindcss.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20%2F%2018-336791.svg?logo=postgresql&logoColor=white)](https://postgresql.org)
[![Tests](https://img.shields.io/badge/Tests-Pytest%20Passing-brightgreen.svg)](https://docs.pytest.org)

An enterprise-grade, multi-modal conversational AI reconnaissance system designed to detect and recognize objects in uploaded imagery, answer complex visual questions in natural language, retain conversational memory across multi-turn dialogues, and visualize detections via an interactive HTML5 canvas with real-time bounding boxes and confidence telemetry.

---

## Table of Contents
1. [Architectural Overview](#architectural-overview)
2. [Folder Structure](#folder-structure)
3. [Technology Stack](#technology-stack)
4. [Quickstart Setup Guide (< 10 Minutes)](#quickstart-setup-guide--10-minutes)
5. [Environment Variables Reference](#environment-variables-reference)
6. [API Specification & Example Payloads](#api-specification--example-payloads)
7. [Standardized Error Handling](#standardized-error-handling)
8. [Automated Testing & Mocking](#automated-testing--mocking)
9. [Multi-Turn Dialogue & Context Fusion](#multi-turn-dialogue--context-fusion)
10. [Security & Production Hardening](#security--production-hardening)

---

## Architectural Overview

The application is structured into decoupled, single-responsibility layers following enterprise clean-architecture principles:

```mermaid
flowchart TD
    Client["React 18 Frontend / API Client"] -->|HTTP / REST| Middleware["Structured Logging Middleware\n(Request ID Tracing & Latency Telemetry)"]
    Middleware --> Routers["API Route Controllers (app/api/v1/)\n(HTTP validation, status codes, dependency injection)"]
    
    Routers --> Services["Service Layer (app/services/)\n(Session, Image, Chat, ContextManager & FusionEngine)"]
    
    Services --> Repositories["Repository Layer (app/repositories/)\n(Database CRUD & Transactions)"]
    Repositories --> DB[("Database\n(PostgreSQL with SQLite local dev fallback)")]
    
    Services --> Clients["Resilient AI Clients (app/clients/)\n(Exponential backoff retry & circuit handling)"]
    Clients --> YOLO["YOLOClient (Ultralytics YOLOv8)\n(MPS/CUDA/CPU Hardware Acceleration)"]
    Clients --> VQA["VQAClient (Gemini 2.0 Flash / HF Inference)\n(Visual Question Answering)"]
    
    Services --> Utils["Validation & Utilities (app/utils/)\n(Streaming chunk size validation & Magic byte check)"]
```

### Architectural Principles Applied:
- **Zero Business Logic in Routes:** Route handlers act solely as HTTP dispatchers, validating schemas and delegating orchestration to the service layer.
- **Service Layer Isolation:** Chat dialogue, image storage, and session lifecycles are decoupled into testable services (`ChatService`, `ImageService`, `SessionService`).
- **Resilient AI Clients:** External AI calls are isolated in `app/clients/` with exponential backoff retries, jitter, and non-hallucinating failure fallbacks.
- **Database Portability:** Custom `GUID` type decorators and dialect checks support both production PostgreSQL and zero-configuration SQLite for local testing.
- **Memory-Safe Streaming Validation:** Uploads are inspected in 64 KB memory chunks, rejecting payloads exceeding size limits or matching corrupt/malicious headers before storing to disk.

---

## Folder Structure

```
conversational-image-recognition-chatbot/
├── .env.example                     # Canonical environment variables template
├── .gitignore                       # Git exclusions (caches, uploads, databases, keys)
├── README.md                        # Documentation & setup guide
├── docker-compose.yml               # Multi-container local production stack
├── pytest.ini                       # Pytest test discovery & asyncio settings
│
├── backend/
│   ├── requirements.txt             # Pinned backend dependencies
│   ├── alembic.ini                  # Alembic database migration config
│   ├── alembic/                     # Database version migration scripts
│   ├── uploads/                     # Local storage directory for validated images
│   ├── app/
│   │   ├── main.py                  # ASGI FastAPI application entry point
│   │   ├── api/                     # HTTP Endpoints (Lean controllers)
│   │   │   └── v1/
│   │   │       ├── chat.py          # POST /api/v1/chat/message
│   │   │       ├── images.py        # Upload, detect, and stream images
│   │   │       ├── sessions.py      # Session CRUD & DELETE /{id}/history
│   │   │       └── health.py        # System health, live, and readiness checks
│   │   ├── clients/                 # Wrappers around external vision/LLM APIs
│   │   │   ├── base.py              # Exponential backoff retry decorators
│   │   │   ├── vqa_client.py        # Gemini 2.0 Flash + HuggingFace VQA client
│   │   │   └── yolo_client.py       # Ultralytics YOLOv8 object detection client
│   │   ├── core/                    # Application core configuration
│   │   │   ├── config.py            # Pydantic Settings loaded from .env
│   │   │   ├── database.py          # SQLAlchemy Async engine, session & GUID type
│   │   │   ├── exceptions.py        # Domain exceptions & global FastAPI handlers
│   │   │   ├── logging.py           # Structured logging configuration
│   │   │   └── logging_middleware.py # Request ID tracing & HTTP access logging
│   │   ├── models/                  # Declarative SQLAlchemy ORM entities
│   │   │   ├── session.py           # SessionModel
│   │   │   ├── image.py             # ImageModel, DetectedObjectModel
│   │   │   └── message.py           # MessageModel
│   │   ├── repositories/            # Data access layer (PostgreSQL / SQLite)
│   │   │   ├── session_repo.py      # Session CRUD queries
│   │   │   ├── image_repo.py        # Image & object detection transaction queries
│   │   │   └── message_repo.py      # Message queries & history deletion
│   │   ├── schemas/                 # Pydantic request/response validation
│   │   │   ├── common.py            # Standard ErrorResponse schema
│   │   │   ├── session.py           # SessionCreate, SessionDetail, SessionClear
│   │   │   ├── image.py             # ImageResponse, DetectedObjectResponse
│   │   │   └── chat.py              # ChatRequest, ChatResponse
│   │   ├── services/                # Business logic layer
│   │   │   ├── session_service.py   # Session creation, retrieval, and history clear
│   │   │   ├── image_service.py     # Image validation, storage & detection
│   │   │   ├── chat_service.py      # Multi-turn chat orchestration & VQA routing
│   │   │   ├── context_manager.py   # Coreference & pronoun resolution
│   │   │   ├── fusion_engine.py     # Multimodal rule and spatial telemetry fusion
│   │   │   └── color_analyzer.py    # Local fallback pixel color analyzer
│   │   └── utils/                   # Helpers and validators
│   │       └── file_validation.py   # Chunked streaming validation & magic byte check
│   └── tests/                       # Complete automated test suite
│       ├── conftest.py              # Test database fixtures and AI mocks
│       ├── test_service_layer.py    # Unit tests for Session, Image, and Chat services
│       ├── test_vqa_client.py       # Unit tests for Gemini/HF client & retries
│       ├── test_file_validation.py  # Unit tests for magic byte & size validation
│       ├── test_sessions.py         # Session lifecycle & history clearing tests
│       ├── test_image_upload.py     # Upload validation & format rejection tests
│       ├── test_chat_endpoint.py    # Chat dialog endpoint tests
│       ├── test_detection_api.py    # Detection API tests
│       ├── test_e2e_integration.py  # End-to-end multimodal pipeline test
│       └── ...
│
└── frontend/                        # React 18 + Vite Web Application
    ├── package.json                 # Pinned frontend dependencies
    ├── src/                         # React components, canvas, and API clients
    └── vite.config.ts               # Vite bundler configuration
```

---

## Technology Stack

- **Backend Framework:** FastAPI 0.111.0, Starlette, Uvicorn (ASGI)
- **Data Validation & Settings:** Pydantic v2, `pydantic-settings`
- **Database & ORM:** PostgreSQL 16+ / SQLite fallback, SQLAlchemy 2.0 (AsyncIO), Alembic
- **Deep Learning & Computer Vision:** Ultralytics YOLOv8, PyTorch 2.2, Pillow
- **Visual Question Answering:** Google Gemini 2.0 Flash (`gemini-2.0-flash`), HuggingFace Serverless Inference (`Salesforce/blip-vqa-base`)
- **Frontend Framework:** React 18, Vite 5, Tailwind CSS, Lucide React, HTML5 Canvas API
- **Testing:** Pytest 8.x, `pytest-asyncio`, AnyIO, HTTPX

---

## Quickstart Setup Guide (< 10 Minutes)

### Option A: Local Development (Fastest Zero-Setup Mode)

The backend features **automatic database portability**: if PostgreSQL is not running locally, it seamlessly uses an isolated local SQLite database (`test.db`), allowing any developer to clone and run in minutes.

#### 1. Clone & Set Environment
```bash
git clone https://github.com/pr0zpect/conversational-image-recognition-chatbot.git
cd conversational-image-recognition-chatbot
cp .env.example .env
```

#### 2. Start Backend
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

# Start backend server with hot-reload:
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
* Interactive Swagger Docs: `http://localhost:8000/docs`
* Health Check: `http://localhost:8000/api/v1/health`

#### 3. Start Frontend
```bash
# In a new terminal:
cd frontend
npm install
npm run dev
```
* Web Application: `http://localhost:5173`

---

### Option B: Containerized Docker Deployment

To spin up the entire production stack (PostgreSQL 16 + FastAPI Backend + React Web UI) in a single command:

```bash
docker compose up --build
```
* **Web UI:** `http://localhost:3000`
* **API Endpoints:** `http://localhost:8000`
* **Interactive API Documentation:** `http://localhost:8000/docs`

---

## Environment Variables Reference

All application configuration is managed via environment variables and loaded into `app.core.config.Settings`. Configure them in `.env`:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `PROJECT_NAME` | `Conversational Image Recognition Chatbot` | Application display name |
| `ENVIRONMENT` | `development` | Runtime environment (`development`, `production`, `test`) |
| `DEBUG` | `false` | Enable verbose debug output |
| `DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@localhost:5432/bel_chatbot_db` | Async SQLAlchemy database URI (auto-falls back to SQLite if asyncpg/pg is unavailable) |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:3000` | Comma-separated allowed CORS origins |
| `GEMINI_API_KEY` | *(empty string)* | Free Google Gemini API Key from [aistudio.google.com](https://aistudio.google.com) for visual question answering |
| `HF_TOKEN` | *(empty string)* | HuggingFace Access Token used as secondary VQA fallback |
| `VQA_MODEL_NAME` | `Salesforce/blip-vqa-base` | Fallback VQA model identifier |
| `YOLO_MODEL_NAME` | `yolov8s.pt` | Ultralytics model checkpoint (`yolov8n.pt`, `yolov8s.pt`, etc.) |
| `INFERENCE_DEVICE` | `mps` (macOS) / `cpu` | Hardware acceleration target (`mps`, `cuda`, `cpu`) |
| `MAX_UPLOAD_SIZE_BYTES` | `15728640` (15 MB) | Maximum accepted image upload size |
| `ALLOWED_IMAGE_EXTENSIONS`| `jpg,jpeg,png,webp` | Permitted image file extensions |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

---

## API Specification & Example Payloads

### 1. Health & Readiness Check
- **`GET /api/v1/health`**
- Returns database connectivity, hardware acceleration, and AI client status.

```json
{
  "status": "healthy",
  "database": "connected",
  "database_dialect": "postgresql",
  "acceleration": {
    "mps_available": true,
    "active_device": "mps"
  },
  "models": {
    "yolo": "yolov8s.pt",
    "vqa": "gemini-2.0-flash / Salesforce/blip-vqa-base",
    "vqa_configured": true
  }
}
```

### 2. Session Management
- **`POST /api/v1/sessions`**
  - **Request Body:** `{"title": "Surveillance Session A"}`
  - **Response (201 Created):**
  ```json
  {
    "id": "c1f7a28e-59b3-4f51-b842-88f5a5e30201",
    "title": "Surveillance Session A",
    "created_at": "2026-11-18T18:00:00Z",
    "updated_at": "2026-11-18T18:00:00Z"
  }
  ```
- **`GET /api/v1/sessions`** — List all sessions.
- **`GET /api/v1/sessions/{id}`** — Retrieve full session details including all messages, images, and detected objects.
- **`DELETE /api/v1/sessions/{id}`** — Delete a session and its associated data (204 No Content).
- **`DELETE /api/v1/sessions/{id}/history`** — Clear dialogue messages for a session while preserving uploaded imagery and detection metadata.
  - **Response (200 OK):**
  ```json
  {
    "session_id": "c1f7a28e-59b3-4f51-b842-88f5a5e30201",
    "deleted_messages_count": 8,
    "message": "Session conversation history cleared successfully."
  }
  ```

### 3. Image Upload & Detection
- **`POST /api/v1/images/upload`** (Multipart Form: `file`, optional `session_id`)
  - Validates magic bytes, checks streaming file size, saves locally, and executes YOLO detection.
  - **Response (201 Created):**
  ```json
  {
    "id": "e98e21a2-11c7-4560-a292-6d2716382029",
    "session_id": "c1f7a28e-59b3-4f51-b842-88f5a5e30201",
    "filename": "recon_patrol.jpg",
    "mime_type": "image/jpeg",
    "width": 1920,
    "height": 1080,
    "url": "/api/v1/images/e98e21a2-11c7-4560-a292-6d2716382029/file",
    "detected_objects": [
      {
        "id": "893c7219-c70f-4886-b4b7-d4fa2ea6ecf5",
        "class_name": "truck",
        "confidence_score": 0.942,
        "bbox": {
          "x_min": 762.5,
          "y_min": 194.8,
          "x_max": 1138.8,
          "y_max": 653.3
        },
        "bbox_normalized": {
          "x_min": 0.397,
          "y_min": 0.180,
          "x_max": 0.593,
          "y_max": 0.605
        },
        "spatial_sector": "center",
        "area_ratio": 0.089
      }
    ]
  }
  ```

- **`POST /api/v1/images/{id}/detect?confidence_threshold=0.50`**
  - Re-evaluates detection with a custom confidence threshold.

- **`GET /api/v1/images/{id}/file`**
  - Streams the raw image binary.

### 4. Conversational Multi-Turn Chat
- **`POST /api/v1/chat/message`**
  - **Request Body:**
  ```json
  {
    "session_id": "c1f7a28e-59b3-4f51-b842-88f5a5e30201",
    "image_id": "e98e21a2-11c7-4560-a292-6d2716382029",
    "content": "Where is the truck and what color is it?"
  }
  ```
  - **Response (200 OK):**
  ```json
  {
    "id": "119842c1-d961-4122-b5e5-f55db4632822",
    "session_id": "c1f7a28e-59b3-4f51-b842-88f5a5e30201",
    "image_id": "e98e21a2-11c7-4560-a292-6d2716382029",
    "sender_type": "assistant",
    "content": "The truck is positioned in the center quadrant of the image (confidence: 94.2%). The vehicle appears dark olive green.",
    "metadata": {
      "intent": "visual_attribute",
      "model": "gemini-2.0-flash",
      "latency_ms": 38.4
    },
    "created_at": "2026-11-18T18:01:15Z"
  }
  ```

---

## Standardized Error Handling

All endpoint errors follow a consistent, non-hallucinating JSON format defined in `app.schemas.common.ErrorResponse`:

```json
{
  "error": "FileValidationError",
  "message": "File exceeds maximum permitted size of 15.0 MB.",
  "status_code": 413,
  "request_id": "req-98f5a5e3",
  "details": {
    "max_size_bytes": 15728640,
    "actual_size_bytes": 20971520
  }
}
```

| HTTP Status | Exception Class | Description |
| :---: | :--- | :--- |
| **400** | `InvalidInputError` / `FileValidationError` | Malformed parameters, unsupported image types, or corrupt files |
| **404** | `NotFoundError` | Session or Image ID not found |
| **413** | `FileValidationError` | Upload payload exceeds `MAX_UPLOAD_SIZE_BYTES` |
| **502 / 503** | `ExternalAPIError` | Upstream AI service unavailable (with graceful fallback explanation) |
| **500** | `AppException` | Unhandled internal exception with request tracing ID |

---

## Automated Testing & Mocking

The backend test suite is designed for **100% deterministic isolation**: tests never require live external AI APIs or a running PostgreSQL server.

```bash
# Run the complete test suite:
pytest backend/tests/ -v
```

### Test Suite Highlights:
- **Mocked AI Execution:** YOLOv8 and Google Gemini/BLIP VQA are mocked by default in API and integration tests, eliminating multi-gigabyte downloads or API rate limits.
- **Database Isolation:** Tests run against an isolated SQLite test database with automated table setup and teardown per test function.
- **Service Layer Testing:** Explicit unit test coverage for `SessionService`, `ImageService`, and `ChatService`.
- **Validation Testing:** Unit tests for magic byte detection (`JPEG`, `PNG`, `WebP`), file truncation, and size limits.
- **Resilience Testing:** Verification of exponential backoff retry logic on transient HTTP errors (429, 500, 502, 503).
- **End-to-End Multimodal Test:** Full flow testing from session creation to image upload, detection, dialogue questions, follow-up pronoun resolution, and deletion.

---

## Multi-Turn Dialogue & Context Fusion

The conversational engine employs a three-stage context fusion pipeline:
1. **Dialogue History Sliding Window:** Retains the preceding $N$ dialogue turns per session to maintain conversational continuity.
2. **Coreference & Pronoun Resolution:** Resolves ambiguous pronouns (*"it"*, *"they"*, *"that vehicle"*) to previously discussed object entities and classes.
3. **Intent-Specific Fusion Routing:**
   - **Counting Queries** (*"How many people are there?"*): Evaluates the deduplicated YOLO detection list directly.
   - **Spatial Queries** (*"Where is the car?"*): Synthesizes sector tags (`top-left`, `center`, `bottom-right`) from normalized bounding boxes.
   - **Visual Attribute Queries** (*"What is written on the sign?"*, *"What color is the car?"*): Dispatches the query and visual crop to Google Gemini 2.0 Flash (with HuggingFace serverless fallback and local pixel color analysis).

---

## Security & Production Hardening

- **No Hardcoded Credentials:** Zero secrets in source code; configuration is strictly mediated via environment variables.
- **Magic Byte Binary Verification:** Validates underlying image header bytes (`FF D8 FF` for JPEG, `89 50 4E 47` for PNG, `52 49 46 46` for WebP), neutralizing file-extension spoofing.
- **Streaming Upload Defense:** Validates file size while streaming to prevent memory exhaustion (DoS attacks via oversized payloads).
- **Transaction Safety:** Database operations wrap batch mutations in atomic transactions with automated rollbacks on failure.
- **CORS Protection:** Configurable origin allowlisting via `CORS_ORIGINS`.

---

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](file:///Users/pr0zpect/Documents/untitled%20folder/conversational-image-recognition-chatbot/LICENSE) file for details.
