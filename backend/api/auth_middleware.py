"""
api/auth_middleware.py
FastAPI dependency that verifies Firebase ID tokens on every protected endpoint.
Supports Email/Password, Google Sign-In, and Anonymous auth.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from firebase_admin import auth as firebase_auth
from loguru import logger

bearer_scheme = HTTPBearer()


class AuthenticatedUser:
    """Represents a verified Firebase user extracted from the ID token."""

    def __init__(self, decoded_token: dict):
        self.uid: str = decoded_token["uid"]
        self.email: str | None = decoded_token.get("email")
        self.name: str | None = decoded_token.get("name")
        self.is_anonymous: bool = decoded_token.get("firebase", {}).get("sign_in_provider") == "anonymous"
        self.provider: str = decoded_token.get("firebase", {}).get("sign_in_provider", "unknown")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> AuthenticatedUser:
    """
    FastAPI dependency — verifies the Bearer token in Authorization header.
    Usage: add `user: AuthenticatedUser = Depends(get_current_user)` to any route.

    Raises 401 if token is missing, expired, or invalid.
    """
    token = credentials.credentials
    try:
        decoded = firebase_auth.verify_id_token(token, check_revoked=True)
    except firebase_auth.RevokedIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked. Please sign in again.",
        )
    except firebase_auth.ExpiredIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please refresh your session.",
        )
    except firebase_auth.InvalidIdTokenError as e:
        logger.warning(f"Invalid token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
        )
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not verify authentication.",
        )

    return AuthenticatedUser(decoded)
