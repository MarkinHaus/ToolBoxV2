"""
Challenge Service - Stateless Challenge Management
Version: 2.0.0

Verwaltet WebAuthn Challenges mit TTL.
Löst das Problem der Race-Conditions und Challenge-Speicherung im User-Objekt.
"""

import base64
import os
from datetime import datetime, timedelta
from typing import Optional, Dict
from ..models import ChallengeData


class ChallengeService:
    """
    Service für temporäre Challenge-Speicherung.

    Implementierung:
    - In-Memory Dict mit TTL (für Entwicklung)
    - TODO: ToolBoxBD für Production (bessere Performance & Skalierung)
    """

    def __init__(self):
        """Initialize challenge storage"""
        self._challenges: Dict[str, ChallengeData] = {}
        self._cleanup_interval = 60  # Cleanup alle 60 Sekunden
        self._last_cleanup = datetime.utcnow()

    def _cleanup_expired(self):
        """Remove expired challenges"""
        now = datetime.utcnow()

        # Nur cleanup wenn genug Zeit vergangen ist
        if (now - self._last_cleanup).total_seconds() < self._cleanup_interval:
            return

        expired_keys = [
            key for key, challenge in self._challenges.items()
            if challenge.is_expired()
        ]

        for key in expired_keys:
            del self._challenges[key]

        self._last_cleanup = now

    def generate_challenge(self) -> str:
        """
        Generate a cryptographically secure challenge.

        Returns:
            Base64URL-encoded challenge string (32 bytes)
        """
        random_bytes = os.urandom(32)
        return base64.urlsafe_b64encode(random_bytes).rstrip(b'=').decode('utf-8')

    def store_challenge(
        self,
        session_id: str,
        username: str,
        challenge_type: str,
        ttl_minutes: int = 5
    ) -> str:
        """
        Store a challenge with TTL.

        Args:
            session_id: Unique session identifier
            username: Username for this challenge
            challenge_type: 'register' or 'login'
            ttl_minutes: Time-to-live in minutes (default: 5)

        Returns:
            Generated challenge string
        """
        self._cleanup_expired()

        challenge = self.generate_challenge()
        expires_at = datetime.utcnow() + timedelta(minutes=ttl_minutes)

        challenge_data = ChallengeData(
            challenge=challenge,
            username=username,
            type=challenge_type,
            expires_at=expires_at
        )

        self._challenges[session_id] = challenge_data
        return challenge

    def get_challenge(self, session_id: str) -> Optional[ChallengeData]:
        """
        Retrieve and validate a challenge.

        Args:
            session_id: Session identifier

        Returns:
            ChallengeData if valid, None if expired or not found
        """
        self._cleanup_expired()

        challenge_data = self._challenges.get(session_id)

        if not challenge_data:
            return None

        if challenge_data.is_expired():
            del self._challenges[session_id]
            return None

        return challenge_data

    def consume_challenge(self, session_id: str) -> Optional[ChallengeData]:
        """
        Retrieve and remove a challenge (one-time use).

        Args:
            session_id: Session identifier

        Returns:
            ChallengeData if valid, None if expired or not found
        """
        challenge_data = self.get_challenge(session_id)

        if challenge_data:
            # Remove after retrieval (one-time use)
            del self._challenges[session_id]

        return challenge_data

    def invalidate_challenge(self, session_id: str) -> bool:
        """
        Manually invalidate a challenge.

        Args:
            session_id: Session identifier

        Returns:
            True if challenge was found and removed, False otherwise
        """
        if session_id in self._challenges:
            del self._challenges[session_id]
            return True
        return False

    def get_stats(self) -> Dict[str, int]:
        """
        Get challenge storage statistics.

        Returns:
            Dict with active and expired challenge counts
        """
        self._cleanup_expired()

        active = len(self._challenges)
        expired = sum(1 for c in self._challenges.values() if c.is_expired())

        return {
            'active_challenges': active,
            'expired_challenges': expired,
            'total': active + expired
        }


# Global singleton instance
_challenge_service_instance: Optional[ChallengeService] = None


def get_challenge_service() -> ChallengeService:
    """Get or create the global ChallengeService instance"""
    global _challenge_service_instance

    if _challenge_service_instance is None:
        _challenge_service_instance = ChallengeService()

    return _challenge_service_instance

