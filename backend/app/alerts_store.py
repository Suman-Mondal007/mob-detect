import csv
import logging
import os
import threading
import tempfile
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

from backend.app.schemas import AlertListResponse, AlertStats, LiveAlert


LOGGER = logging.getLogger("mob_alert.backend")
AlertSource = Literal["live", "upload"]


class AlertRow(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    timestamp: str = Field(index=True)
    timestamp_epoch: float = Field(index=True)
    source: str = Field(index=True)
    source_name: str
    label: str
    confidence: float
    smoothed_confidence: float
    threshold: float
    snapshot_filename: str | None = None
    video_time_seconds: float | None = None


class AlertStore:
    def __init__(self, database_path: Path, snapshots_dir: Path) -> None:
        self._database_path = database_path
        self._snapshots_dir = snapshots_dir
        self._lock = threading.RLock()
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)
        database_url = URL.create(
            "sqlite",
            database=str(self._database_path),
        )
        self._engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False, "timeout": 30},
        )
        SQLModel.metadata.create_all(self._engine)

    def record(
        self,
        *,
        timestamp: str,
        source: AlertSource,
        source_name: str,
        label: str,
        confidence: float,
        smoothed_confidence: float,
        threshold: float,
        snapshot_frame: np.ndarray | None = None,
        video_time_seconds: float | None = None,
    ) -> LiveAlert:
        normalized_timestamp, timestamp_epoch = self._normalize_timestamp(timestamp)
        snapshot_filename = (
            self._save_snapshot(snapshot_frame) if snapshot_frame is not None else None
        )
        row = AlertRow(
            timestamp=normalized_timestamp,
            timestamp_epoch=timestamp_epoch,
            source=source,
            source_name=source_name,
            label=label,
            confidence=confidence,
            smoothed_confidence=smoothed_confidence,
            threshold=threshold,
            snapshot_filename=snapshot_filename,
            video_time_seconds=video_time_seconds,
        )

        try:
            with self._lock, Session(self._engine) as session:
                session.add(row)
                session.commit()
                session.refresh(row)
                return self._to_response(row)
        except (OSError, RuntimeError, SQLAlchemyError) as exc:
            if snapshot_filename is not None:
                self._remove_snapshot(snapshot_filename)
            LOGGER.exception("Could not persist alert to %s.", self._database_path)
            raise RuntimeError(f"Could not persist alert: {exc}") from exc

    def list_alerts(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        source: AlertSource | None = None,
        search: str | None = None,
    ) -> AlertListResponse:
        filters = self._filters(source=source, search=search)
        with self._lock, Session(self._engine) as session:
            count_query = select(func.count()).select_from(AlertRow)
            filtered_count_query = count_query.where(*filters)
            total = int(session.exec(count_query).one())
            filtered_total = int(session.exec(filtered_count_query).one())

            alerts_query = (
                select(AlertRow)
                .where(*filters)
                .order_by(AlertRow.timestamp_epoch.desc(), AlertRow.id.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            )
            alerts = [
                self._to_response(row)
                for row in session.exec(alerts_query).all()
            ]

            cutoff_epoch = (datetime.now(timezone.utc) - timedelta(days=1)).timestamp()
            last_24_hours = int(
                session.exec(
                    select(func.count())
                    .select_from(AlertRow)
                    .where(AlertRow.timestamp_epoch >= cutoff_epoch)
                ).one()
            )
            average_confidence = session.exec(
                select(func.avg(AlertRow.confidence))
            ).one()

        return AlertListResponse(
            alerts=alerts,
            total=filtered_total,
            page=page,
            page_size=page_size,
            total_pages=(filtered_total + page_size - 1) // page_size,
            stats=AlertStats(
                total_alerts=total,
                last_24h=last_24_hours,
                average_confidence=float(average_confidence or 0),
            ),
        )

    def get_snapshot_path(self, alert_id: int) -> Path | None:
        with self._lock, Session(self._engine) as session:
            row = session.get(AlertRow, alert_id)
            if row is None or row.snapshot_filename is None:
                return None
            snapshot_path = (self._snapshots_dir / row.snapshot_filename).resolve()
            if snapshot_path.parent != self._snapshots_dir.resolve():
                LOGGER.error("Alert %s has an invalid snapshot path.", alert_id)
                return None
            return snapshot_path if snapshot_path.is_file() else None

    def csv_content(
        self,
        *,
        source: AlertSource | None = None,
        search: str | None = None,
    ) -> str:
        filters = self._filters(source=source, search=search)
        output = StringIO(newline="")
        fields = (
            "id",
            "timestamp",
            "source",
            "source_name",
            "label",
            "confidence",
            "smoothed_confidence",
            "threshold",
            "snapshot_url",
            "video_time_seconds",
        )
        with self._lock, Session(self._engine) as session:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            query = (
                select(AlertRow)
                .where(*filters)
                .order_by(AlertRow.timestamp_epoch.desc(), AlertRow.id.desc())
            )
            for row in session.exec(query):
                alert = self._to_response(row)
                writer.writerow(alert.model_dump())
        return output.getvalue()

    def close(self) -> None:
        self._engine.dispose()

    @staticmethod
    def _normalize_timestamp(timestamp: str) -> tuple[str, float]:
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("Alert timestamp must be a valid ISO-8601 datetime.") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        normalized = parsed.astimezone(timezone.utc)
        return normalized.isoformat(), normalized.timestamp()

    @staticmethod
    def _filters(
        *,
        source: AlertSource | None,
        search: str | None,
    ) -> list[ColumnElement[bool]]:
        filters: list[ColumnElement[bool]] = []
        if source is not None:
            filters.append(AlertRow.source == source)
        if search:
            filters.append(
                AlertRow.label.contains(search, autoescape=True)
                | AlertRow.source_name.contains(search, autoescape=True)
            )
        return filters

    def _save_snapshot(self, frame: np.ndarray) -> str:
        encoded, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 78],
        )
        if not encoded:
            raise RuntimeError("Could not encode the alert snapshot.")

        filename = f"{uuid.uuid4().hex}.jpg"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="mob-alert-snapshot-",
            suffix=".tmp",
            dir=self._snapshots_dir,
        )
        temporary_path = Path(temporary_name)
        snapshot_path = self._snapshots_dir / filename
        try:
            with os.fdopen(descriptor, "wb") as snapshot_file:
                snapshot_file.write(buffer.tobytes())
                snapshot_file.flush()
                os.fsync(snapshot_file.fileno())
            os.replace(temporary_path, snapshot_path)
        except OSError:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                LOGGER.exception("Could not clean up temporary alert snapshot.")
            raise
        return filename

    def _remove_snapshot(self, filename: str) -> None:
        try:
            (self._snapshots_dir / filename).unlink(missing_ok=True)
        except OSError:
            LOGGER.exception("Could not remove unreferenced alert snapshot %s.", filename)

    @staticmethod
    def _to_response(row: AlertRow) -> LiveAlert:
        return LiveAlert(
            id=row.id or 0,
            timestamp=row.timestamp,
            source=row.source,
            source_name=row.source_name,
            label=row.label,
            confidence=row.confidence,
            smoothed_confidence=row.smoothed_confidence,
            threshold=row.threshold,
            snapshot_url=(
                f"/alerts/{row.id}/snapshot"
                if row.snapshot_filename is not None
                else None
            ),
            video_time_seconds=row.video_time_seconds,
        )
