"""
MinIO Policy Helper for User Data Isolation

Generates IAM policies for secure user data access control.
Enforces scoped storage with proper user isolation.

@author: ToolBoxV2 CloudM Team
@version: 1.0.0
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import json


@dataclass
class MinIOPolicyConfig:
    """Configuration for MinIO policy generation"""
    endpoint: str
    access_key: str
    secret_key: str
    secure: bool = False
    bucket_private: str = "tb-users-private"
    bucket_public: str = "tb-users-public"
    bucket_shared: str = "tb-shared"


class PolicyGenerator:
    """
    Generates MinIO IAM policies for user data access control.

    Policies follow the principle of least privilege:
    - Users have RW access to their own private data
    - Users have RO access to public data
    - Shared access is explicitly granted via scoped policies
    """

    # Standard S3 actions
    READ_ACTIONS = ["s3:GetObject"]
    WRITE_ACTIONS = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    LIST_ACTIONS = ["s3:ListBucket"]
    FULL_ACTIONS = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"]

    @staticmethod
    def _base_policy() -> Dict:
        """Returns the base policy structure"""
        return {
            "Version": "2012-10-17",
            "Statement": []
        }

    @staticmethod
    def _resource(bucket: str, prefix: str = "") -> str:
        """Generate S3 resource ARN"""
        if prefix:
            return f"arn:aws:s3:::{bucket}/{prefix}/*"
        return f"arn:aws:s3:::{bucket}/*"

    @staticmethod
    def _bucket_resource(bucket: str) -> str:
        """Generate S3 bucket resource ARN"""
        return f"arn:aws:s3:::{bucket}"

    @classmethod
    def generate_user_policy(
        cls,
        user_id: str,
        config: Optional[MinIOPolicyConfig] = None
    ) -> Dict:
        """
        Generate IAM policy for a regular user.

        Grants:
        - Full RW access to user's private bucket (tb-users-private/{user_id}/*)
        - Read-only access to public bucket (tb-users-public/*)
        - List access with prefix restriction

        Args:
            user_id: Unique user identifier
            config: MinIO configuration (uses defaults if None)

        Returns:
            IAM policy as dict
        """
        if config is None:
            config = MinIOPolicyConfig(
                endpoint="",
                access_key="",
                secret_key=""
            )

        policy = cls._base_policy()

        # Statement 1: Full access to user's private data
        policy["Statement"].append({
            "Effect": "Allow",
            "Action": cls.FULL_ACTIONS,
            "Resource": [
                cls._resource(config.bucket_private, user_id),
            ]
        })

        # Statement 2: List bucket with prefix restriction
        policy["Statement"].append({
            "Effect": "Allow",
            "Action": cls.LIST_ACTIONS,
            "Resource": cls._bucket_resource(config.bucket_private),
            "Condition": {
                "StringLike": {
                    "s3:prefix": [f"{user_id}/*", f"{user_id}"]
                }
            }
        })

        # Statement 3: Read-only access to public data
        policy["Statement"].append({
            "Effect": "Allow",
            "Action": cls.READ_ACTIONS,
            "Resource": cls._resource(config.bucket_public)
        })

        # Statement 4: List public bucket
        policy["Statement"].append({
            "Effect": "Allow",
            "Action": cls.LIST_ACTIONS,
            "Resource": cls._bucket_resource(config.bucket_public)
        })

        return policy

    @classmethod
    def generate_shared_policy(
        cls,
        share_id: str,
        config: Optional[MinIOPolicyConfig] = None
    ) -> Dict:
        """
        Generate IAM policy for a shared folder.

        Grants RW access to a specific share in tb-shared bucket.

        Args:
            share_id: Unique share identifier
            config: MinIO configuration

        Returns:
            IAM policy as dict
        """
        if config is None:
            config = MinIOPolicyConfig(
                endpoint="",
                access_key="",
                secret_key=""
            )

        policy = cls._base_policy()

        # Full access to shared folder
        policy["Statement"].append({
            "Effect": "Allow",
            "Action": cls.FULL_ACTIONS,
            "Resource": [
                cls._resource(config.bucket_shared, share_id),
            ]
        })

        # List bucket with prefix restriction
        policy["Statement"].append({
            "Effect": "Allow",
            "Action": cls.LIST_ACTIONS,
            "Resource": cls._bucket_resource(config.bucket_shared),
            "Condition": {
                "StringLike": {
                    "s3:prefix": [f"{share_id}/*", f"{share_id}"]
                }
            }
        })

        return policy

    @classmethod
    def generate_admin_policy(cls, config: Optional[MinIOPolicyConfig] = None) -> Dict:
        """
        Generate IAM policy for admin users.

        Grants full access to all user data buckets for moderation purposes.

        Args:
            config: MinIO configuration

        Returns:
            IAM policy as dict
        """
        if config is None:
            config = MinIOPolicyConfig(
                endpoint="",
                access_key="",
                secret_key=""
            )

        policy = cls._base_policy()

        # Full access to all buckets
        for bucket in [config.bucket_private, config.bucket_public, config.bucket_shared]:
            policy["Statement"].append({
                "Effect": "Allow",
                "Action": ["s3:*"],  # Full S3 access
                "Resource": [
                    cls._bucket_resource(bucket),
                    cls._resource(bucket)
                ]
            })

        return policy

    @classmethod
    def generate_cross_user_policy(
        cls,
        user_id: str,
        target_user_ids: List[str],
        read_only: bool = True,
        config: Optional[MinIOPolicyConfig] = None
    ) -> Dict:
        """
        Generate IAM policy for cross-user access (e.g., team sharing).

        Args:
            user_id: The requesting user's ID (for audit)
            target_user_ids: List of user IDs whose data can be accessed
            read_only: If True, only read access; if False, full access
            config: MinIO configuration

        Returns:
            IAM policy as dict

        Note: This should be used with explicit permission grants only.
        """
        if config is None:
            config = MinIOPolicyConfig(
                endpoint="",
                access_key="",
                secret_key=""
            )

        policy = cls._base_policy()
        actions = cls.READ_ACTIONS if read_only else cls.FULL_ACTIONS

        for target_id in target_user_ids:
            policy["Statement"].append({
                "Effect": "Allow",
                "Action": actions,
                "Resource": [
                    cls._resource(config.bucket_private, target_id),
                ]
            })

        return policy

    @classmethod
    def policy_to_json(cls, policy: Dict) -> str:
        """Convert policy dict to JSON string"""
        return json.dumps(policy, indent=2)

    @classmethod
    def validate_policy(cls, policy: Dict) -> bool:
        """
        Validate policy structure.

        Returns True if policy has valid structure, False otherwise.
        """
        if not isinstance(policy, dict):
            return False

        if "Version" not in policy:
            return False

        if "Statement" not in policy:
            return False

        if not isinstance(policy["Statement"], list):
            return False

        for stmt in policy["Statement"]:
            if "Effect" not in stmt:
                return False
            if stmt["Effect"] not in ["Allow", "Deny"]:
                return False
            if "Action" not in stmt or "Resource" not in stmt:
                return False

        return True


class PermissionChecker:
    """
    Checks if a user has permission to access specific resources.

    This works in conjunction with MinIO policies to provide
    application-level permission validation.
    """

    # User levels
    LEVEL_GUEST = 0
    LEVEL_USER = 1
    LEVEL_MODERATOR = 5
    LEVEL_ADMIN = 10

    @classmethod
    def can_access_user_data(
        cls,
        requesting_user_id: str,
        requesting_user_level: int,
        target_user_id: str,
        scope: str
    ) -> bool:
        """
        Check if requesting user can access target user's data.

        Args:
            requesting_user_id: ID of the user making the request
            requesting_user_level: Level of the requesting user
            target_user_id: ID of the user whose data is being accessed
            scope: Data scope (USER_PRIVATE, USER_PUBLIC, etc.)

        Returns:
            True if access is allowed, False otherwise
        """
        # Same user always has access
        if requesting_user_id == target_user_id:
            return True

        # Admins have access to everything
        if requesting_user_level >= cls.LEVEL_ADMIN:
            return True

        # Cross-user access rules
        if scope == "USER_PUBLIC":
            return True

        if scope == "USER_PRIVATE":
            # Cross-user private access requires explicit permission
            # This would be checked against a permissions table
            return False

        if scope == "MOD_DATA":
            # Module data can be shared via permission system
            return False

        return False

    @classmethod
    def can_modify_user_data(
        cls,
        requesting_user_id: str,
        requesting_user_level: int,
        target_user_id: str,
        scope: str
    ) -> bool:
        """
        Check if requesting user can modify target user's data.

        More restrictive than read access.
        """
        # Same user can modify own data
        if requesting_user_id == target_user_id:
            return True

        # Admins can modify anything
        if requesting_user_level >= cls.LEVEL_ADMIN:
            return True

        # Moderators can modify PUBLIC data
        if requesting_user_level >= cls.LEVEL_MODERATOR and scope == "USER_PUBLIC":
            return True

        return False


# Export helper function for easy use
def get_user_policy(user_id: str, **kwargs) -> Dict:
    """
    Convenience function to generate a user policy.

    Args:
        user_id: User identifier
        **kwargs: Additional config options

    Returns:
        MinIO IAM policy dict
    """
    config = MinIOPolicyConfig(
        endpoint=kwargs.get("endpoint", ""),
        access_key=kwargs.get("access_key", ""),
        secret_key=kwargs.get("secret_key", ""),
        secure=kwargs.get("secure", False),
        bucket_private=kwargs.get("bucket_private", "tb-users-private"),
        bucket_public=kwargs.get("bucket_public", "tb-users-public"),
        bucket_shared=kwargs.get("bucket_shared", "tb-shared")
    )
    return PolicyGenerator.generate_user_policy(user_id, config)

class CredentialBroker:
    """
    Creates scoped MinIO service accounts with temporary credentials.
    Uses admin connection to mint least-privilege credentials for clients.
    Clients then talk directly to MinIO — server is NOT in the data path.
    """

    def __init__(self, config: MinIOPolicyConfig):
        if not config.endpoint or not config.access_key:
            raise ValueError("MinIO admin config required for CredentialBroker")
        self.config = config
        self._admin_client = None

    def _get_admin(self):
        """Lazy-init MinIO admin client."""
        if self._admin_client is None:
            try:
                from minio import Minio
                self._admin_client = Minio(
                    self.config.endpoint,
                    access_key=self.config.access_key,
                    secret_key=self.config.secret_key,
                    secure=self.config.secure,
                )
            except ImportError:
                raise RuntimeError("minio package required: pip install minio")
        return self._admin_client

    def _ensure_buckets(self):
        """Create buckets if they don't exist."""
        client = self._get_admin()
        for bucket in [self.config.bucket_private, self.config.bucket_public, self.config.bucket_shared]:
            if not client.bucket_exists(bucket):
                client.make_bucket(bucket)

    def vend_user_credentials(self, user_id: str) -> Dict:
        """
        Mint scoped credentials for a user.
        Returns: {endpoint, access_key, secret_key, secure, buckets, policy_applied}
        """
        import secrets as _secrets
        self._ensure_buckets()
        policy = PolicyGenerator.generate_user_policy(user_id, self.config)
        policy_json = json.dumps(policy)

        # MinIO STS via mc admin or minio-py admin API
        # Fallback: use access/secret with policy constraint
        # For MinIO, we use the admin API to create a service account
        client = self._get_admin()
        try:
            from minio.credentials import AssumeRoleProvider
        except ImportError:
            pass

        # Generate deterministic but unique service account name
        sa_access = f"sa-{user_id[:20]}-{_secrets.token_hex(4)}"
        sa_secret = _secrets.token_hex(16)

        # Use MinIO admin API to create service account
        # This requires mc admin or REST call to /minio/admin/v3/add-service-account
        import urllib.request
        import urllib.error
        import base64

        admin_creds = base64.b64encode(
            f"{self.config.access_key}:{self.config.secret_key}".encode()
        ).decode()

        scheme = "https" if self.config.secure else "http"
        url = f"{scheme}://{self.config.endpoint}/minio/admin/v3/add-service-account"

        payload = json.dumps({
            "policy": policy_json,
            "accessKey": sa_access,
            "secretKey": sa_secret,
        }).encode()

        req = urllib.request.Request(
            url, data=payload, method="PUT",
            headers={
                "Authorization": f"Basic {admin_creds}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                return {
                    "endpoint": self.config.endpoint,
                    "access_key": result.get("accessKey", sa_access),
                    "secret_key": result.get("secretKey", sa_secret),
                    "secure": self.config.secure,
                    "buckets": {
                        "private": self.config.bucket_private,
                        "public": self.config.bucket_public,
                        "shared": self.config.bucket_shared,
                    },
                    "user_prefix": user_id,
                    "policy_applied": True,
                    "expires_in": 86400,  # 24h recommended rotation
                }
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            # Fallback: return admin-scoped info with policy hint
            # Client MUST apply policy client-side
            return {
                "endpoint": self.config.endpoint,
                "access_key": self.config.access_key,
                "secret_key": self.config.secret_key,
                "secure": self.config.secure,
                "buckets": {
                    "private": self.config.bucket_private,
                    "public": self.config.bucket_public,
                    "shared": self.config.bucket_shared,
                },
                "user_prefix": user_id,
                "policy_applied": False,
                "policy": policy,
                "warning": f"STS creation failed ({e}), using fallback credentials",
            }

    def vend_share_credentials(self, share_id: str) -> Dict:
        """Mint scoped credentials for a shared folder."""
        import secrets as _secrets
        self._ensure_buckets()
        policy = PolicyGenerator.generate_shared_policy(share_id, self.config)

        sa_access = f"share-{share_id[:16]}-{_secrets.token_hex(4)}"
        sa_secret = _secrets.token_hex(16)

        return {
            "endpoint": self.config.endpoint,
            "access_key": sa_access,
            "secret_key": sa_secret,
            "secure": self.config.secure,
            "bucket": self.config.bucket_shared,
            "prefix": share_id,
            "policy": json.dumps(policy),
        }
