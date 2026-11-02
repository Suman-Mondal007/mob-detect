import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from sqlalchemy.engine import URL
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Field, Session, SQLModel, create_engine, select


LOGGER = logging.getLogger("mob_alert.backend")
StoredCameraSourceType = Literal["rtsp", "file", "webcam"]


class Camera(SQLModel, table=True):
    id: str = Field(primary_key=True, max_length=64)
    name: str = Field(max_length=100)
    location: str = Field(max_length=200)
    source_type: str = Field(max_length=16, index=True)
    source_uri: str = Field(max_length=2048)
    enabled: bool = Field(default=True, index=True)
    auto_start: bool = Field(default=True, index=True)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class CameraStore:
    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        self._lock = threading.RLock()
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(
            URL.create("sqlite", database=str(self._database_path)),
            connect_args={"check_same_thread": False, "timeout": 30},
        )
        SQLModel.metadata.create_all(self._engine)
        self._migrate()
    def _migrate(self) -> None:
        try:
            with self._lock, Session(self._engine) as session:
                session.execute(__import__("sqlalchemy").text("ALTER TABLE camera ADD COLUMN auto_start BOOLEAN NOT NULL DEFAULT 1"))
                session.commit()
        except Exception:
            pass


    def create(self, camera: Camera) -> Camera:
        try:
            with self._lock, Session(self._engine) as session:
                session.add(camera)
                session.commit()
                session.refresh(camera)
                return camera
        except SQLAlchemyError as exc:
            LOGGER.exception("Could not create camera '%s'.", camera.id)
            raise RuntimeError(f"Could not create camera: {exc}") from exc

    def get(self, camera_id: str) -> Camera | None:
        try:
            with self._lock, Session(self._engine) as session:
                return session.get(Camera, camera_id)
        except SQLAlchemyError as exc:
            LOGGER.exception("Could not load camera '%s'.", camera_id)
            raise RuntimeError(f"Could not load camera: {exc}") from exc

    def list(self) -> list[Camera]:
        try:
            with self._lock, Session(self._engine) as session:
                return list(
                    session.exec(select(Camera).order_by(Camera.created_at, Camera.id))
                )
        except SQLAlchemyError as exc:
            LOGGER.exception("Could not list cameras.")
            raise RuntimeError(f"Could not list cameras: {exc}") from exc

    def update(
        self,
        camera_id: str,
        *,
        name: str,
        location: str,
        source_type: StoredCameraSourceType,
        source_uri: str,
        enabled: bool,
        auto_start: bool | None = None,
    ) -> Camera | None:
        try:
            with self._lock, Session(self._engine) as session:
                camera = session.get(Camera, camera_id)
                if camera is None:
                    return None
                camera.name = name
                camera.location = location
                camera.source_type = source_type
                camera.source_uri = source_uri
                camera.enabled = enabled
                if auto_start is not None:
                    camera.auto_start = auto_start
                session.add(camera)
                session.commit()
                session.refresh(camera)
                return camera
        except SQLAlchemyError as exc:
            LOGGER.exception("Could not update camera '%s'.", camera_id)
            raise RuntimeError(f"Could not update camera: {exc}") from exc

    def delete(self, camera_id: str) -> bool:
        try:
            with self._lock, Session(self._engine) as session:
                camera = session.get(Camera, camera_id)
                if camera is None:
                    return False
                session.delete(camera)
                session.commit()
                return True
        except SQLAlchemyError as exc:
            LOGGER.exception("Could not delete camera '%s'.", camera_id)
            raise RuntimeError(f"Could not delete camera: {exc}") from exc

    def close(self) -> None:
        self._engine.dispose()
