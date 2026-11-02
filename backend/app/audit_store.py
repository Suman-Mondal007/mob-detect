import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Field, Session, SQLModel, create_engine, select

LOGGER = logging.getLogger("mob_alert.backend")


class AuditRow(SQLModel, table=True):
    __tablename__ = "audit_log"

    id: int | None = Field(default=None, primary_key=True)
    timestamp: str = Field(index=True)
    username: str = Field(max_length=64, index=True)
    role: str = Field(max_length=32)
    action: str = Field(max_length=128, index=True)
    target: str | None = Field(default=None, max_length=256)
    detail: str | None = Field(default=None)


class AuditStore:
    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        self._lock = threading.RLock()
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(
            f"sqlite:///{database_path}",
            connect_args={"check_same_thread": False, "timeout": 30},
        )
        SQLModel.metadata.create_all(self._engine)

    def record(self, *, username: str, role: str, action: str, target: str | None = None, detail: str | None = None) -> None:
        try:
            with self._lock, Session(self._engine) as session:
                session.add(
                    AuditRow(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        username=username,
                        role=role,
                        action=action,
                        target=target,
                        detail=detail,
                    )
                )
                session.commit()
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Could not write audit log: %s", exc)

    def list_entries(self, limit: int = 100) -> list[dict[str, object]]:
        with self._lock, Session(self._engine) as session:
            rows = session.exec(
                select(AuditRow).order_by(AuditRow.id.desc()).limit(limit)  # type: ignore[attr-defined]
            ).all()
            return [
                {
                    "id": row.id,
                    "timestamp": row.timestamp,
                    "username": row.username,
                    "role": row.role,
                    "action": row.action,
                    "target": row.target,
                    "detail": row.detail,
                }
                for row in rows
            ]

    def close(self) -> None:
        self._engine.dispose()
