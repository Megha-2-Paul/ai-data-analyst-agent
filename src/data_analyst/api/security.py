"""Password hashing and signed access tokens."""

from datetime import datetime, timedelta, timezone

import jwt
from pwdlib import PasswordHash

_password_hash = PasswordHash.recommended()
_ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    return _password_hash.hash(password)


def verify_password(password: str, encoded: str) -> bool:
    try:
        return _password_hash.verify(password, encoded)
    except Exception:
        return False


def create_access_token(user_id: int, secret_key: str, expires_minutes: int = 60) -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {"sub": str(user_id), "iat": now, "exp": now + timedelta(minutes=expires_minutes)},
        secret_key, algorithm=_ALGORITHM,
    )


def decode_access_token(token: str, secret_key: str) -> int:
    payload = jwt.decode(token, secret_key, algorithms=[_ALGORITHM])
    subject = payload.get("sub")
    if not isinstance(subject, str) or not subject.isdigit():
        raise jwt.InvalidTokenError("Invalid subject")
    return int(subject)
