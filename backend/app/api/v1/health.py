from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import platform
import os
from app.core.database import get_db
from app.core.config import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/live")
def get_liveness():
    """
    Ultra-fast, non-blocking liveness probe for cloud load balancers (Render, Kubernetes).
    Returns instantly in < 1ms with zero database overhead.
    """
    return {"status": "ok", "app": settings.APP_NAME}


@router.get("")
async def get_health_status(db: AsyncSession = Depends(get_db)):
    """
    Comprehensive system health check:
    - Verifies compute acceleration device (Apple MPS / CUDA / CPU)
    - Verifies database async connectivity (PostgreSQL or SQLite)
    - Reports system platform & memory metrics
    """
    # 1. Check Database connectivity
    db_status = "disconnected"
    db_version = "unknown"
    try:
        check_query = (
            text("SELECT sqlite_version();")
            if settings.DATABASE_URL.startswith("sqlite")
            else text("SELECT version();")
        )
        result = await db.execute(check_query)
        db_version = result.scalar()
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"

    # 2. Check Device Acceleration
    mps_available = False
    mps_built = False
    cuda_available = False
    torch_version = "unknown"
    try:
        import torch
        mps_available = torch.backends.mps.is_available()
        mps_built = torch.backends.mps.is_built()
        cuda_available = torch.cuda.is_available()
        torch_version = torch.__version__
    except Exception:
        pass

    active_device = settings.INFERENCE_DEVICE

    # 3. Process Memory Metrics
    mem_rss_mb = None
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        mem_rss_mb = round(proc.memory_info().rss / (1024 * 1024), 2)
    except Exception:
        pass

    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "memory": {
            "rss_mb": mem_rss_mb,
            "container_limit_mb": 512,
            "is_memory_safe": mem_rss_mb < 450 if mem_rss_mb else True,
        },
        "database": {
            "status": db_status,
            "version": db_version,
            "name": settings.DB_NAME,
        },
        "acceleration": {
            "active_device": active_device,
            "mps_available": mps_available,
            "mps_built": mps_built,
            "cuda_available": cuda_available,
            "torch_version": torch_version,
        },
        "platform": {
            "system": platform.system(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "models": {
            "yolo_model": settings.YOLO_MODEL_NAME,
            "vqa_model": settings.VQA_MODEL_NAME,
        },
    }
