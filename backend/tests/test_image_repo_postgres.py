import os
import asyncio
import asyncio
import pytest
from uuid import uuid4

DATABASE_URL = os.getenv("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="Set TEST_DATABASE_URL to run PostgreSQL integration tests",
)


def test_detection_replacement_postgres():
    asyncio.run(_test_detection_replacement_postgres())


async def _test_detection_replacement_postgres():
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

    from app.core.database import Base
    import app.models
    from app.models.session import SessionModel
    from app.models.image import ImageModel, DetectedObjectModel
    from app.repositories.image_repo import ImageRepository

    engine = create_async_engine(DATABASE_URL)

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

        async with SessionLocal() as db:
            session_id = uuid4()
            image_id = uuid4()

            db.add(SessionModel(id=session_id, title="Integration Test"))
            db.add(ImageModel(
                id=image_id,
                session_id=session_id,
                filename="test.jpg",
                storage_path="/tmp/test.jpg",
                mime_type="image/jpeg",
                file_size_bytes=1000,
                width=640,
                height=480,
            ))
            await db.commit()

            await ImageRepository.replace_detected_objects(
                db, image_id,
                [
                    {"class_name": "car", "confidence": 0.95},
                    {"class_name": "person", "confidence": 0.88},
                ],
            )

            result = await db.execute(
                select(DetectedObjectModel).where(
                    DetectedObjectModel.image_id == image_id
                )
            )
            assert len(result.scalars().all()) == 2

            await ImageRepository.replace_detected_objects(
                db, image_id,
                [{"class_name": "bicycle", "confidence": 0.91}],
            )

            result = await db.execute(
                select(DetectedObjectModel).where(
                    DetectedObjectModel.image_id == image_id
                )
            )
            objects = result.scalars().all()
            assert len(objects) == 1
            assert objects[0].class_name == "bicycle"

            await ImageRepository.replace_detected_objects(
                db, image_id, []
            )

            result = await db.execute(
                select(DetectedObjectModel).where(
                    DetectedObjectModel.image_id == image_id
                )
            )
            assert len(result.scalars().all()) == 0

            await db.delete(
                await db.get(ImageModel, image_id)
            )
            await db.delete(
                await db.get(SessionModel, session_id)
            )
            await db.commit()

    finally:
        await engine.dispose()
