"""Verification service for publisher verification."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import httpx

from registry.db.repositories.user_repo import UserRepository
from registry.models.user import VerificationStatus

logger = logging.getLogger(__name__)


class VerificationMethod(str, Enum):
    """Methods for publisher verification."""

    GITHUB = "github"
    DOMAIN = "domain"
    MANUAL = "manual"


@dataclass
class VerificationRequest:
    """Verification request data.

    Attributes:
        id: Request ID.
        publisher_id: Publisher ID.
        method: Verification method.
        data: Method-specific data.
        status: Request status.
        submitted_at: Submission timestamp.
        reviewed_at: Review timestamp.
        reviewed_by: Admin who reviewed.
        notes: Review notes.
    """

    id: int
    publisher_id: str
    method: VerificationMethod
    data: dict = field(default_factory=dict)
    status: str = "pending"
    submitted_at: datetime = field(default_factory=datetime.utcnow)
    reviewed_at: Optional[datetime] = None
    reviewed_by: Optional[str] = None
    notes: Optional[str] = None


class VerificationService:
    """Service for publisher verification.

    Handles verification requests and validation.

    Attributes:
        user_repo: User repository.
    """

    def __init__(self, user_repo: UserRepository) -> None:
        """Initialize the service.

        Args:
            user_repo: User repository.
        """
        self.user_repo = user_repo

    async def submit_request(
        self,
        cloudm_user_id: str,
        publisher_id: str,
        method: VerificationMethod,
        data: dict,
    ) -> bool:
        """Submit a verification request.

        Args:
            cloudm_user_id: Clerk user ID.
            publisher_id: Publisher ID.
            method: Verification method.
            data: Method-specific data.

        Returns:
            True if request submitted.
        """
        # Update publisher status to pending
        await self.user_repo.update_publisher_status(
            publisher_id,
            VerificationStatus.PENDING,
        )

        logger.info(f"Verification request submitted for publisher {publisher_id}")
        return True

    async def verify_github(
        self,
        github_username: str,
        expected_user_id: str,
    ) -> bool:
        """Verify publisher via GitHub repository.

        User must create a repo named 'tb-registry-verify' with a
        VERIFY.txt file containing their user ID.

        Args:
            github_username: GitHub username.
            expected_user_id: Expected user ID in verification file.

        Returns:
            True if verification successful.
        """
        try:
            url = f"https://raw.githubusercontent.com/{github_username}/tb-registry-verify/main/VERIFY.txt"

            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0)

                if response.status_code != 200:
                    logger.warning(f"GitHub verification failed: {response.status_code}")
                    return False

                content = response.text.strip()
                if content == expected_user_id:
                    logger.info(f"GitHub verification successful for {github_username}")
                    return True

                logger.warning(f"GitHub verification mismatch: expected {expected_user_id}, got {content}")
                return False

        except Exception as e:
            logger.error(f"GitHub verification error: {e}")
            return False

    async def verify_domain(
        self,
        domain: str,
        expected_user_id: str,
    ) -> bool:
        """Verify publisher via domain.

        User must add a TXT record or serve a file at
        /.well-known/tb-registry-verify containing their user ID.

        Args:
            domain: Domain to verify.
            expected_user_id: Expected user ID.

        Returns:
            True if verification successful.
        """
        # Try well-known file first
        try:
            url = f"https://{domain}/.well-known/tb-registry-verify"

            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0)

                if response.status_code == 200:
                    content = response.text.strip()
                    if content == expected_user_id:
                        logger.info(f"Domain verification successful for {domain}")
                        return True

        except Exception as e:
            logger.debug(f"Well-known file check failed: {e}")

        # TODO: Add DNS TXT record verification
        logger.warning(f"Domain verification failed for {domain}")
        return False

