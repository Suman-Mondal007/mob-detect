import csv
import logging
import os
import queue
import tempfile
import threading
import uuid
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from sqlalchemy import func
from sqlalchemy.engine import URL
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.elements import ColumnElement
from sqlmodel import Field, Session, SQLModel, create_engine, select

from backend.app.schemas import IncidentListResponse, IncidentResponse


LOGGER = logging.getLogger("mob_alert.backend")
IncidentStatus = Literal["new", "acknowledged", "false-positive"]
EvidenceFrame = tuple[float, np.ndarray]


class IncidentRow(SQLModel, table=True):
    __tablename__ = "incident"

    id: int | None = Field(default=None, primary_key=True)
    camera_id: str = Field(index=True)
    start_time: str = Field(index=True)
    start_epoch: float = Field(index=True)
    end_time: str | None = None
    end_epoch: float | None = Field(default=None, index=True)
    peak_confidence: float
    snapshot_filename: str | None = None
    clip_filename: str | None = None
    status: str = Field(default="new", index=True)


class IncidentStore:
    def __init__(
        self,
        database_path: Path,
        evidence_dir: Path,
        *,
        retention_days: int = 30,
        ring_buffer_seconds: float = 10,
        evidence_fps: float = 5,
        cleanup_interval_seconds: float = 3600,
    ) -> None:
        if retention_days <= 0 or ring_buffer_seconds <= 0 or evidence_fps <= 0:
            raise ValueError("Incident retention and evidence settings must be positive.")
        if cleanup_interval_seconds <= 0:
            raise ValueError("Incident cleanup interval must be positive.")

        self._database_path = database_path
        self._evidence_dir = evidence_dir.resolve()
        self._retention_days = retention_days
        self._ring_buffer_seconds = ring_buffer_seconds
        self._evidence_fps = evidence_fps
        self._cleanup_interval_seconds = cleanup_interval_seconds
        self._lock = threading.RLock()
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._evidence_dir.mkdir(parents=True, exist_ok=True)
        database_url = URL.create("sqlite", database=str(self._database_path))
        self._engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False, "timeout": 30},
        )
        SQLModel.metadata.create_all(self._engine)
        self.cleanup_expired_evidence()

        self._clip_queue: queue.Queue[
            tuple[int, str, list[EvidenceFrame]] | None
        ] = queue.Queue(maxsize=64)
        self._clip_worker = threading.Thread(
            target=self._write_queued_clips,
            name="mob-alert-incident-evidence",
            daemon=True,
        )
        self._clip_worker.start()
        self._cleanup_stop = threading.Event()
        self._cleanup_worker = threading.Thread(
            target=self._cleanup_loop,
            name="mob-alert-incident-retention",
            daemon=True,
        )
        self._cleanup_worker.start()

    @property
    def ring_buffer_seconds(self) -> float:
        return self._ring_buffer_seconds

    def create_incident(
        self,
        *,
        camera_id: str,
        start_time: str,
        peak_confidence: float,
        snapshot_frame: np.ndarray,
    ) -> IncidentResponse:
        normalized_time, timestamp_epoch = self._normalize_timestamp(start_time)
        snapshot_filename = self._save_snapshot(snapshot_frame)
        row = IncidentRow(
            camera_id=camera_id,
            start_time=normalized_time,
            start_epoch=timestamp_epoch,
            peak_confidence=peak_confidence,
            snapshot_filename=snapshot_filename,
            status="new",
        )
        try:
            with self._lock, Session(self._engine) as session:
                session.add(row)
                session.commit()
                session.refresh(row)
                return self._to_response(row)
        except (OSError, RuntimeError, SQLAlchemyError) as exc:
            self._remove_evidence(snapshot_filename)
            LOGGER.exception("Could not persist camera incident.")
            raise RuntimeError(f"Could not persist camera incident: {exc}") from exc

    def update_peak(self, incident_id: int, confidence: float) -> None:
        with self._lock, Session(self._engine) as session:
            row = session.get(IncidentRow, incident_id)
            if row is None:
                raise RuntimeError(f"Incident {incident_id} was not found.")
            if confidence <= row.peak_confidence:
                return
            row.peak_confidence = confidence
            try:
                session.add(row)
                session.commit()
            except SQLAlchemyError as exc:
                LOGGER.exception("Could not update incident %s peak confidence.", incident_id)
                raise RuntimeError(f"Could not update incident peak confidence: {exc}") from exc

    def close_incident(
        self,
        incident_id: int,
        *,
        end_time: str,
        frames: list[EvidenceFrame],
    ) -> None:
        normalized_time, timestamp_epoch = self._normalize_timestamp(end_time)
        with self._lock, Session(self._engine) as session:
            row = session.get(IncidentRow, incident_id)
            if row is None:
                raise RuntimeError(f"Incident {incident_id} was not found.")
            if row.end_time is not None:
                return
            row.end_time = normalized_time
            row.end_epoch = timestamp_epoch
            try:
                session.add(row)
                session.commit()
            except SQLAlchemyError as exc:
                LOGGER.exception("Could not close incident %s.", incident_id)
                raise RuntimeError(f"Could not close incident: {exc}") from exc

        try:
            self._clip_queue.put_nowait((incident_id, normalized_time, frames))
        except queue.Full as exc:
            LOGGER.error(
                "Incident evidence queue is full; clip for incident %s was not queued.",
                incident_id,
            )
            raise RuntimeError("Incident evidence queue is full.") from exc

    def acknowledge(self, incident_id: int) -> IncidentResponse | None:
        return self._update_status(incident_id, "acknowledged", allowed={"new"})

    def mark_false_positive(self, incident_id: int) -> IncidentResponse | None:
        return self._update_status(
            incident_id,
            "false-positive",
            allowed={"new", "acknowledged"},
        )

    def get_incident(self, incident_id: int) -> IncidentResponse | None:
        with self._lock, Session(self._engine) as session:
            row = session.get(IncidentRow, incident_id)
            return self._to_response(row) if row is not None else None

    def list_incidents(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        camera_id: str | None = None,
        status: IncidentStatus | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        search: str | None = None,
    ) -> IncidentListResponse:
        filters = self._filters(
            camera_id=camera_id,
            status=status,
            start_time=start_time,
            end_time=end_time,
            search=search,
        )
        with self._lock, Session(self._engine) as session:
            count_query = select(func.count()).select_from(IncidentRow)
            total = int(session.exec(count_query.where(*filters)).one())
            rows = session.exec(
                select(IncidentRow)
                .where(*filters)
                .order_by(IncidentRow.start_epoch.desc(), IncidentRow.id.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            ).all()
        return IncidentListResponse(
            incidents=[self._to_response(row) for row in rows],
            total=total,
            page=page,
            page_size=page_size,
            total_pages=(total + page_size - 1) // page_size,
        )

    def csv_content(
        self,
        *,
        camera_id: str | None = None,
        status: IncidentStatus | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        search: str | None = None,
    ) -> str:
        filters = self._filters(
            camera_id=camera_id,
            status=status,
            start_time=start_time,
            end_time=end_time,
            search=search,
        )
        output = StringIO(newline="")
        fields = (
            "id",
            "camera_id",
            "start_time",
            "end_time",
            "peak_confidence",
            "status",
            "snapshot_url",
            "clip_url",
        )
        with self._lock, Session(self._engine) as session:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            rows = session.exec(
                select(IncidentRow)
                .where(*filters)
                .order_by(IncidentRow.start_epoch.desc(), IncidentRow.id.desc())
            )
            for row in rows:
                writer.writerow(self._to_response(row).model_dump())
        return output.getvalue()

    def get_snapshot_path(self, incident_id: int) -> Path | None:
        return self._get_evidence_path(incident_id, "snapshot_filename")

    def get_clip_path(self, incident_id: int) -> Path | None:
        return self._get_evidence_path(incident_id, "clip_filename")

    def cleanup_expired_evidence(self, *, now: datetime | None = None) -> int:
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        cutoff_epoch = (
            current.astimezone(timezone.utc) - timedelta(days=self._retention_days)
        ).timestamp()
        removed = 0
        with self._lock, Session(self._engine) as session:
            rows = session.exec(
                select(IncidentRow).where(IncidentRow.start_epoch < cutoff_epoch)
            ).all()
            for row in rows:
                changed = False
                for attribute in ("snapshot_filename", "clip_filename"):
                    filename = getattr(row, attribute)
                    if filename is None:
                        continue
                    evidence_path = self._safe_evidence_path(filename)
                    try:
                        evidence_path.unlink(missing_ok=True)
                    except OSError:
                        LOGGER.exception("Could not expire incident evidence %s.", filename)
                        continue
                    setattr(row, attribute, None)
                    changed = True
                    removed += 1
                if changed:
                    session.add(row)
            try:
                session.commit()
            except SQLAlchemyError as exc:
                LOGGER.exception("Could not update expired incident evidence metadata.")
                raise RuntimeError(f"Could not update expired evidence metadata: {exc}") from exc
        return removed

    def close(self) -> None:
        self._cleanup_stop.set()
        if self._cleanup_worker.is_alive():
            self._cleanup_worker.join(timeout=2)
        self._clip_queue.join()
        self._clip_queue.put(None)
        if self._clip_worker.is_alive():
            self._clip_worker.join(timeout=10)
        self._engine.dispose()

    def _cleanup_loop(self) -> None:
        while not self._cleanup_stop.wait(self._cleanup_interval_seconds):
            try:
                self.cleanup_expired_evidence()
            except (OSError, RuntimeError, SQLAlchemyError):
                LOGGER.exception("Periodic incident evidence cleanup failed.")

    def _write_queued_clips(self) -> None:
        while True:
            task = self._clip_queue.get()
            try:
                if task is None:
                    return
                incident_id, end_time, frames = task
                self._save_clip(incident_id, end_time, frames)
            except (OSError, RuntimeError, cv2.error, SQLAlchemyError):
                LOGGER.exception("Could not write evidence clip for incident.")
            finally:
                self._clip_queue.task_done()

    def _save_clip(
        self,
        incident_id: int,
        end_time: str,
        frames: list[EvidenceFrame],
    ) -> None:
        unique_frames: list[EvidenceFrame] = []
        seen_timestamps: set[float] = set()
        for timestamp, image in sorted(frames, key=lambda item: item[0]):
            if timestamp in seen_timestamps or image.size == 0:
                continue
            unique_frames.append((timestamp, image))
            seen_timestamps.add(timestamp)
        if not unique_frames:
            raise RuntimeError(f"Incident {incident_id} has no frames for an evidence clip.")

        height, width = unique_frames[0][1].shape[:2]
        filename = f"incident-{incident_id}-{uuid.uuid4().hex}.mp4"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="mob-alert-incident-",
            suffix=".mp4",
            dir=self._evidence_dir,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        final_path = self._evidence_dir / filename
        writer = cv2.VideoWriter(
            str(temporary_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self._evidence_fps,
            (width, height),
        )
        if not writer.isOpened():
            writer.release()
            temporary_path.unlink(missing_ok=True)
            raise RuntimeError("Could not open MP4 evidence writer.")
        try:
            for _timestamp, image in unique_frames:
                frame_height, frame_width = image.shape[:2]
                if (frame_width, frame_height) != (width, height):
                    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(image)
        finally:
            writer.release()
        if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            temporary_path.unlink(missing_ok=True)
            raise RuntimeError("MP4 evidence writer produced an empty clip.")
        try:
            os.replace(temporary_path, final_path)
        except OSError:
            temporary_path.unlink(missing_ok=True)
            raise

        try:
            with self._lock, Session(self._engine) as session:
                row = session.get(IncidentRow, incident_id)
                if row is None:
                    raise RuntimeError(f"Incident {incident_id} was not found.")
                if row.end_time != end_time:
                    return
                row.clip_filename = filename
                session.add(row)
                session.commit()
        except (OSError, RuntimeError, SQLAlchemyError):
            self._remove_evidence(filename)
            raise

    def _save_snapshot(self, frame: np.ndarray) -> str:
        try:
            encoded, buffer = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 85],
            )
        except cv2.error as exc:
            raise RuntimeError("Could not encode incident snapshot.") from exc
        if not encoded:
            raise RuntimeError("Could not encode incident snapshot.")
        filename = f"incident-{uuid.uuid4().hex}.jpg"
        self._write_atomic(filename, buffer.tobytes())
        return filename

    def _write_atomic(self, filename: str, content: bytes) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="mob-alert-incident-",
            suffix=".tmp",
            dir=self._evidence_dir,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, self._evidence_dir / filename)
        except OSError:
            temporary_path.unlink(missing_ok=True)
            raise

    def _get_evidence_path(
        self,
        incident_id: int,
        attribute: Literal["snapshot_filename", "clip_filename"],
    ) -> Path | None:
        with self._lock, Session(self._engine) as session:
            row = session.get(IncidentRow, incident_id)
            filename = getattr(row, attribute) if row is not None else None
            if filename is None:
                return None
            path = self._safe_evidence_path(filename)
            return path if path.is_file() else None

    def _safe_evidence_path(self, filename: str) -> Path:
        path = (self._evidence_dir / filename).resolve()
        if path.parent != self._evidence_dir:
            raise RuntimeError("Incident evidence path is invalid.")
        return path

    def _remove_evidence(self, filename: str) -> None:
        try:
            self._safe_evidence_path(filename).unlink(missing_ok=True)
        except OSError:
            LOGGER.exception("Could not remove unreferenced incident evidence %s.", filename)

    def _update_status(
        self,
        incident_id: int,
        status: IncidentStatus,
        *,
        allowed: set[str],
    ) -> IncidentResponse | None:
        with self._lock, Session(self._engine) as session:
            row = session.get(IncidentRow, incident_id)
            if row is None:
                return None
            if row.status not in allowed:
                raise ValueError(f"Incident cannot transition from {row.status} to {status}.")
            row.status = status
            try:
                session.add(row)
                session.commit()
                session.refresh(row)
            except SQLAlchemyError as exc:
                LOGGER.exception("Could not update incident %s status.", incident_id)
                raise RuntimeError(f"Could not update incident status: {exc}") from exc
            return self._to_response(row)

    @staticmethod
    def _normalize_timestamp(timestamp: str) -> tuple[str, float]:
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("Incident timestamp must be a valid ISO-8601 datetime.") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        normalized = parsed.astimezone(timezone.utc)
        return normalized.isoformat(), normalized.timestamp()

    @staticmethod
    def _filters(
        *,
        camera_id: str | None,
        status: IncidentStatus | None,
        start_time: datetime | None,
        end_time: datetime | None,
        search: str | None,
    ) -> list[ColumnElement[bool]]:
        filters: list[ColumnElement[bool]] = []
        if camera_id is not None:
            filters.append(IncidentRow.camera_id == camera_id)
        if status is not None:
            filters.append(IncidentRow.status == status)
        if start_time is not None:
            filters.append(
                IncidentRow.start_epoch >= IncidentStore._timestamp_epoch(start_time)
            )
        if end_time is not None:
            filters.append(
                IncidentRow.start_epoch <= IncidentStore._timestamp_epoch(end_time)
            )
        if search:
            filters.append(IncidentRow.camera_id.contains(search, autoescape=True))
        return filters

    @staticmethod
    def _timestamp_epoch(timestamp: datetime) -> float:
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc).timestamp()

    @staticmethod
    def _to_response(row: IncidentRow) -> IncidentResponse:
        incident_id = row.id or 0
        return IncidentResponse(
            id=incident_id,
            camera_id=row.camera_id,
            start_time=row.start_time,
            end_time=row.end_time,
            peak_confidence=row.peak_confidence,
            snapshot_url=(
                f"/incidents/{incident_id}/snapshot"
                if row.snapshot_filename is not None
                else None
            ),
            clip_url=(
                f"/incidents/{incident_id}/clip"
                if row.clip_filename is not None
                else None
            ),
            status=row.status,
        )
