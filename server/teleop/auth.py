"""Pairing (6-digit one-time codes) and JWT session management.

Flow:
 1. The web app calls ``POST /api/pair/request``. A pairing request with a random 6-digit code is created.
    The code is *only* shown inside Isaac Sim (pairing window); the HTTP response contains just the request id.
 2. The user reads the code in Isaac Sim and types it into the web app, which calls ``POST /api/pair/confirm``.
 3. On success the server issues a signed JWT (HS256). Every other API call and the WebRTC offer require
    ``Authorization: Bearer <jwt>``. Sessions can be revoked from the Isaac Sim window.

All methods are thread-safe: they are used from the HTTP thread and from the Kit main thread (UI).
"""

from __future__ import annotations

import hmac
import secrets
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

import jwt

ISSUER = "isaacsimvr"
AUDIENCE = "isaacsimvr-webapp"


class AuthError(Exception):
    def __init__(self, message: str, status: int = 401, code: str = "unauthorized"):
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass
class PairingRequest:
    request_id: str
    code: str
    device_name: str
    remote_ip: str
    user_agent: str
    created_at: float
    expires_at: float
    attempts: int = 0


@dataclass
class Session:
    session_id: str
    device_name: str
    remote_ip: str
    issued_at: float
    expires_at: float
    last_seen: float = field(default_factory=time.time)
    connected_peers: int = 0


class AuthManager:
    def __init__(
        self,
        *,
        jwt_ttl_seconds: int = 8 * 3600,
        pairing_ttl_seconds: int = 120,
        max_attempts: int = 5,
        max_pending: int = 8,
        rate_limit_per_minute: int = 6,
        secret: bytes | None = None,
    ) -> None:
        # A fresh secret on every start: restarting Isaac Sim invalidates all previously issued tokens.
        self._secret = secret or secrets.token_bytes(32)
        self._jwt_ttl = jwt_ttl_seconds
        self._pairing_ttl = pairing_ttl_seconds
        self._max_attempts = max_attempts
        self._max_pending = max_pending
        self._rate_limit = rate_limit_per_minute
        self._lock = threading.RLock()
        self._pending: dict[str, PairingRequest] = {}
        self._sessions: dict[str, Session] = {}
        self._rate: dict[str, deque[float]] = {}
        self._version = 0  # bumped on every change so the UI knows when to redraw
        self.on_new_request = None  # optional callback(PairingRequest)

    # -- helpers ----------------------------------------------------------------------------------
    @property
    def version(self) -> int:
        return self._version

    def _bump(self) -> None:
        self._version += 1

    def _purge(self, now: float) -> None:
        expired = [rid for rid, r in self._pending.items() if r.expires_at <= now]
        for rid in expired:
            del self._pending[rid]
        dead = [sid for sid, s in self._sessions.items() if s.expires_at <= now]
        for sid in dead:
            del self._sessions[sid]
        if expired or dead:
            self._bump()

    def _check_rate(self, ip: str, now: float) -> None:
        q = self._rate.setdefault(ip, deque())
        while q and q[0] < now - 60:
            q.popleft()
        if len(q) >= self._rate_limit:
            raise AuthError("Too many pairing requests. Try again later.", 429, "rate_limited")
        q.append(now)

    # -- pairing ----------------------------------------------------------------------------------
    def create_pairing_request(self, device_name: str, remote_ip: str, user_agent: str) -> PairingRequest:
        now = time.time()
        with self._lock:
            self._purge(now)
            self._check_rate(remote_ip, now)
            # One pending request per device: a new request from the same client (IP + browser) replaces the
            # previous one, so the Isaac Sim window never shows stale codes for that device.
            for old in [r for r in self._pending.values() if r.remote_ip == remote_ip and r.user_agent == (user_agent or "")[:200]]:
                del self._pending[old.request_id]
            if len(self._pending) >= self._max_pending:
                oldest = min(self._pending.values(), key=lambda r: r.created_at)
                del self._pending[oldest.request_id]
            req = PairingRequest(
                request_id=uuid.uuid4().hex,
                code=f"{secrets.randbelow(1_000_000):06d}",
                device_name=(device_name or "Unknown device").strip()[:64],
                remote_ip=remote_ip,
                user_agent=(user_agent or "")[:200],
                created_at=now,
                expires_at=now + self._pairing_ttl,
            )
            self._pending[req.request_id] = req
            self._bump()
        if self.on_new_request:
            self.on_new_request(req)
        return req

    def confirm_pairing(self, request_id: str, code: str, remote_ip: str) -> tuple[str, Session]:
        now = time.time()
        with self._lock:
            self._purge(now)
            req = self._pending.get(request_id or "")
            if req is None:
                raise AuthError("Pairing request expired or not found. Request a new code.", 404, "not_found")
            if req.remote_ip != remote_ip:
                raise AuthError("Pairing request belongs to another client.", 403, "forbidden")
            req.attempts += 1
            ok = hmac.compare_digest(req.code.encode(), (code or "").strip().encode())
            if not ok:
                left = self._max_attempts - req.attempts
                if left <= 0:
                    del self._pending[request_id]
                    self._bump()
                    raise AuthError("Too many wrong codes. Request a new code.", 403, "locked")
                self._bump()
                raise AuthError(f"Wrong code. {left} attempt(s) left.", 401, "wrong_code")
            del self._pending[request_id]
            session = Session(
                session_id=uuid.uuid4().hex,
                device_name=req.device_name,
                remote_ip=remote_ip,
                issued_at=now,
                expires_at=now + self._jwt_ttl,
            )
            self._sessions[session.session_id] = session
            self._bump()
        token = jwt.encode(
            {
                "iss": ISSUER,
                "aud": AUDIENCE,
                "sub": session.session_id,
                "name": session.device_name,
                "iat": int(session.issued_at),
                "nbf": int(session.issued_at) - 5,
                "exp": int(session.expires_at),
                "jti": uuid.uuid4().hex,
            },
            self._secret,
            algorithm="HS256",
        )
        return token, session

    def cancel_pairing(self, request_id: str) -> None:
        with self._lock:
            if self._pending.pop(request_id, None) is not None:
                self._bump()

    # -- tokens -----------------------------------------------------------------------------------
    def verify_token(self, token: str) -> Session:
        if not token:
            raise AuthError("Missing bearer token.")
        try:
            claims = jwt.decode(
                token,
                self._secret,
                algorithms=["HS256"],
                audience=AUDIENCE,
                issuer=ISSUER,
                options={"require": ["exp", "iat", "sub", "aud", "iss"]},
            )
        except jwt.ExpiredSignatureError as exc:
            raise AuthError("Token expired.", 401, "token_expired") from exc
        except jwt.InvalidTokenError as exc:
            raise AuthError("Invalid token.", 401, "invalid_token") from exc
        with self._lock:
            self._purge(time.time())
            session = self._sessions.get(claims["sub"])
            if session is None:
                raise AuthError("Session revoked.", 401, "revoked")
            session.last_seen = time.time()
            return session

    def revoke(self, session_id: str) -> None:
        with self._lock:
            if self._sessions.pop(session_id, None) is not None:
                self._bump()

    def set_peer_count(self, session_id: str, delta: int) -> None:
        with self._lock:
            s = self._sessions.get(session_id)
            if s:
                s.connected_peers = max(0, s.connected_peers + delta)
                self._bump()

    # -- snapshots for the UI ---------------------------------------------------------------------
    def snapshot(self) -> tuple[list[PairingRequest], list[Session]]:
        with self._lock:
            self._purge(time.time())
            pend = sorted(self._pending.values(), key=lambda r: r.created_at, reverse=True)  # newest first
            sess = sorted(self._sessions.values(), key=lambda s: s.issued_at)
            return [PairingRequest(**vars(r)) for r in pend], [Session(**vars(s)) for s in sess]
