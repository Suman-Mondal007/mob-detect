import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Field, Session, SQLModel, create_engine, select

Role = Literal["admin", "operator", "viewer"]

JWT_ALGORITHM = "HS256"


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def hash_password(password: str, salt: str | None = None) -> str:
    if salt is None:
        salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000)
    return f"pbkdf2_sha256${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _scheme, salt, digest = stored.split("$", 2)
    except ValueError:
        return False
    candidate = hash_password(password, salt)
    return hmac.compare_digest(candidate, stored)


@dataclass
class AuthUser:
    username: str
    role: Role


class UserRow(SQLModel, table=True):
    __tablename__ = "auth_user"

    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True, max_length=64)
    password_hash: str = Field(max_length=256)
    role: str = Field(default="viewer", max_length=32)


class UserStore:
    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        self._lock = threading.RLock()
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(
            f"sqlite:///{database_path}",
            connect_args={"check_same_thread": False, "timeout": 30},
        )
        SQLModel.metadata.create_all(self._engine)

    def create_user(self, username: str, password: str, role: Role) -> AuthUser:
        with self._lock, Session(self._engine) as session:
            existing = session.exec(select(UserRow).where(UserRow.username == username)).first()
            if existing is not None:
                existing.password_hash = hash_password(password)
                existing.role = role
                session.add(existing)
                session.commit()
                return AuthUser(username=username, role=role)  # type: ignore[arg-type]
            row = UserRow(username=username, password_hash=hash_password(password), role=role)
            session.add(row)
            session.commit()
            return AuthUser(username=username, role=role)  # type: ignore[arg-type]

    def authenticate(self, username: str, password: str) -> AuthUser | None:
        with self._lock, Session(self._engine) as session:
            row = session.exec(select(UserRow).where(UserRow.username == username)).first()
            if row is None or not verify_password(password, row.password_hash):
                return None
            return AuthUser(username=row.username, role=row.role)  # type: ignore[arg-type]

    def get_user(self, username: str) -> AuthUser | None:
        with self._lock, Session(self._engine) as session:
            row = session.exec(select(UserRow).where(UserRow.username == username)).first()
            if row is None:
                return None
            return AuthUser(username=row.username, role=row.role)  # type: ignore[arg-type]

    def list_users(self) -> list[AuthUser]:
        with self._lock, Session(self._engine) as session:
            rows = session.exec(select(UserRow)).all()
            return [AuthUser(username=row.username, role=row.role) for row in rows]  # type: ignore[arg-type]

    def close(self) -> None:
        self._engine.dispose()


def create_access_token(user: AuthUser, secret: str, expires_minutes: int = 60) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    now = int(time.time())
    payload = {
        "sub": user.username,
        "role": user.role,
        "iat": now,
        "exp": now + expires_minutes * 60,
    }
    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(secret.encode("utf-8"), f"{header_b64}.{payload_b64}".encode("ascii"), hashlib.sha256).digest()
    return f"{header_b64}.{payload_b64}.{_b64url_encode(signature)}"


def decode_access_token(token: str, secret: str) -> AuthUser:
    try:
        header_b64, payload_b64, signature_b64 = token.split(".", 2)
    except ValueError as exc:
        raise ValueError("Invalid token format.") from exc
    expected = hmac.new(secret.encode("utf-8"), f"{header_b64}.{payload_b64}".encode("ascii"), hashlib.sha256).digest()
    if not hmac.compare_digest(_b64url_decode(signature_b64), expected):
        raise ValueError("Invalid token signature.")
    payload = json.loads(_b64url_decode(payload_b64))
    if int(payload.get("exp", 0)) < int(time.time()):
        raise ValueError("Token has expired.")
    username = str(payload.get("sub", ""))
    role = str(payload.get("role", "viewer"))
    if role not in {"admin", "operator", "viewer"}:
        raise ValueError("Invalid role.")
    return AuthUser(username=username, role=role)  # type: ignore[arg-type]
