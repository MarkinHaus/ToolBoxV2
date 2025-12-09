"""
ToolBox V2 - MinIO User Manager
Verwaltet MinIO IAM Users und Policies für Multi-User Blob Storage

Features:
- Erstellt MinIO Users mit User-spezifischen Credentials
- Generiert Scope-basierte IAM Policies
- Integration mit Clerk Auth
- Credential-Rotation
"""

import os
import json
import time
import secrets
import hashlib
import subprocess
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum

# ToolBoxV2 imports
try:
    from toolboxv2.utils.security.cryp import Code
    from toolboxv2.utils.extras.blobs import BlobFile
    TOOLBOX_AVAILABLE = True
except ImportError:
    TOOLBOX_AVAILABLE = False
    class Code:
        @staticmethod
        def encrypt_symmetric(data: bytes, key: bytes) -> bytes:
            return bytes([b ^ key[i % len(key)] for i, b in enumerate(data)])
        @staticmethod
        def decrypt_symmetric(data: bytes, key: bytes) -> bytes:
            return bytes([b ^ key[i % len(key)] for i, b in enumerate(data)])
        @staticmethod
        def DK():
            def inner():
                import uuid
                return hashlib.sha256(str(uuid.getnode()).encode()).digest()
            return inner

from toolboxv2.utils.extras.db.scoped_storage import Scope


# =================== Data Classes ===================

@dataclass
class MinIOUserCredentials:
    """MinIO User Credentials"""
    user_id: str              # Clerk User ID
    minio_access_key: str     # MinIO Access Key
    minio_secret_key: str     # MinIO Secret Key (encrypted when stored)
    created_at: float = 0
    last_rotated: float = 0
    policies: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MinIOUserCredentials":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ScopePolicy:
    """IAM Policy für einen Scope"""
    name: str
    scope: Scope
    bucket: str
    prefix_pattern: str  # z.B. "${user_id}/*" oder "*"
    actions: List[str]   # z.B. ["s3:GetObject", "s3:PutObject"]

    def to_minio_policy(self, user_id: str = None) -> dict:
        """Generiert MinIO Policy JSON"""
        # Ersetze Platzhalter
        prefix = self.prefix_pattern
        if user_id:
            prefix = prefix.replace("${user_id}", user_id)

        resource = f"arn:aws:s3:::{self.bucket}/{prefix}" if prefix else f"arn:aws:s3:::{self.bucket}/*"

        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": self.actions,
                    "Resource": [
                        f"arn:aws:s3:::{self.bucket}",  # Bucket itself
                        resource  # Objects
                    ]
                }
            ]
        }


# =================== Policy Definitions ===================

# Standard-Aktionen
READ_ACTIONS = ["s3:GetObject", "s3:GetObjectVersion", "s3:ListBucket"]
WRITE_ACTIONS = ["s3:PutObject", "s3:DeleteObject", "s3:DeleteObjectVersion"]
LIST_ACTIONS = ["s3:ListBucket", "s3:GetBucketLocation"]
ALL_ACTIONS = READ_ACTIONS + WRITE_ACTIONS + LIST_ACTIONS

# Scope Policies
SCOPE_POLICIES = {
    Scope.PUBLIC_READ: ScopePolicy(
        name="public-read",
        scope=Scope.PUBLIC_READ,
        bucket="tb-public-read",
        prefix_pattern="*",
        actions=READ_ACTIONS + LIST_ACTIONS
    ),
    Scope.PUBLIC_RW: ScopePolicy(
        name="public-rw",
        scope=Scope.PUBLIC_RW,
        bucket="tb-public-rw",
        prefix_pattern="*",
        actions=ALL_ACTIONS
    ),
    Scope.USER_PUBLIC: ScopePolicy(
        name="user-public",
        scope=Scope.USER_PUBLIC,
        bucket="tb-users-public",
        prefix_pattern="${user_id}/*",
        actions=ALL_ACTIONS  # Wird auf User-Prefix beschränkt
    ),
    Scope.USER_PRIVATE: ScopePolicy(
        name="user-private",
        scope=Scope.USER_PRIVATE,
        bucket="tb-users-private",
        prefix_pattern="${user_id}/*",
        actions=ALL_ACTIONS
    ),
    Scope.SERVER_SCOPE: ScopePolicy(
        name="server",
        scope=Scope.SERVER_SCOPE,
        bucket="tb-servers",
        prefix_pattern="${server_id}/*",
        actions=ALL_ACTIONS
    ),
    Scope.MOD_DATA: ScopePolicy(
        name="mod-data",
        scope=Scope.MOD_DATA,
        bucket="tb-mods",
        prefix_pattern="*/${user_id}/*",  # mods/{mod}/{user_id}/*
        actions=ALL_ACTIONS
    )
}


# =================== MinIO Admin Client ===================

class MinIOAdminClient:
    """
    Wrapper für MinIO Admin Operationen via `mc` CLI

    Requires: mc (MinIO Client) installed and configured
    """

    def __init__(self, alias: str = "local", mc_path: str = None):
        """
        Args:
            alias: MinIO Alias in mc config (z.B. "local", "cloud")
            mc_path: Pfad zur mc Binary
        """
        self.alias = alias
        self.mc = mc_path or self._find_mc()

        if not self.mc:
            raise RuntimeError("MinIO Client (mc) not found. Install with: pip install minio-mc")

    def _find_mc(self) -> Optional[str]:
        """Findet mc Binary"""
        possible_paths = [
            "mc",
            "/usr/local/bin/mc",
            os.path.expanduser("~/.local/bin/mc"),
            os.path.expanduser("~/minio-binaries/mc"),
            "C:\\minio\\mc.exe"
        ]

        for path in possible_paths:
            try:
                result = subprocess.run(
                    [path, "--version"],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return path
            except:
                continue

        return None

    def _run_mc(self, *args, check: bool = True) -> subprocess.CompletedProcess:
        """Führt mc Befehl aus"""
        cmd = [self.mc] + list(args)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        if check and result.returncode != 0:
            raise RuntimeError(f"mc command failed: {result.stderr}")

        return result

    # =================== User Management ===================

    def create_user(self, access_key: str, secret_key: str) -> bool:
        """Erstellt MinIO User"""
        try:
            self._run_mc("admin", "user", "add", self.alias, access_key, secret_key)
            return True
        except RuntimeError as e:
            if "already exists" in str(e).lower():
                return True  # User existiert schon
            raise

    def delete_user(self, access_key: str) -> bool:
        """Löscht MinIO User"""
        try:
            self._run_mc("admin", "user", "remove", self.alias, access_key)
            return True
        except:
            return False

    def list_users(self) -> List[str]:
        """Listet alle MinIO Users"""
        result = self._run_mc("admin", "user", "list", self.alias, "--json", check=False)

        users = []
        for line in result.stdout.strip().split('\n'):
            if line:
                try:
                    data = json.loads(line)
                    if "accessKey" in data:
                        users.append(data["accessKey"])
                except:
                    pass

        return users

    def user_exists(self, access_key: str) -> bool:
        """Prüft ob User existiert"""
        return access_key in self.list_users()

    def set_user_status(self, access_key: str, enabled: bool) -> bool:
        """Aktiviert/Deaktiviert User"""
        status = "enable" if enabled else "disable"
        try:
            self._run_mc("admin", "user", status, self.alias, access_key)
            return True
        except:
            return False

    # =================== Policy Management ===================

    def create_policy(self, name: str, policy_json: dict) -> bool:
        """Erstellt MinIO Policy"""
        import tempfile

        # Schreibe Policy in temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(policy_json, f)
            temp_path = f.name

        try:
            self._run_mc("admin", "policy", "create", self.alias, name, temp_path)
            return True
        except RuntimeError as e:
            if "already exists" in str(e).lower():
                # Update existierende Policy
                self._run_mc("admin", "policy", "remove", self.alias, name, check=False)
                self._run_mc("admin", "policy", "create", self.alias, name, temp_path)
                return True
            raise
        finally:
            os.unlink(temp_path)

    def delete_policy(self, name: str) -> bool:
        """Löscht MinIO Policy"""
        try:
            self._run_mc("admin", "policy", "remove", self.alias, name)
            return True
        except:
            return False

    def list_policies(self) -> List[str]:
        """Listet alle Policies"""
        result = self._run_mc("admin", "policy", "list", self.alias, "--json", check=False)

        policies = []
        for line in result.stdout.strip().split('\n'):
            if line:
                try:
                    data = json.loads(line)
                    if "policy" in data:
                        policies.append(data["policy"])
                except:
                    pass

        return policies

    def attach_policy(self, policy_name: str, user_access_key: str) -> bool:
        """Weist User eine Policy zu"""
        try:
            self._run_mc(
                "admin", "policy", "attach", self.alias,
                policy_name, "--user", user_access_key
            )
            return True
        except:
            return False

    def detach_policy(self, policy_name: str, user_access_key: str) -> bool:
        """Entfernt Policy von User"""
        try:
            self._run_mc(
                "admin", "policy", "detach", self.alias,
                policy_name, "--user", user_access_key
            )
            return True
        except:
            return False

    # =================== Bucket Management ===================

    def create_bucket(self, bucket: str) -> bool:
        """Erstellt Bucket"""
        try:
            self._run_mc("mb", f"{self.alias}/{bucket}", check=False)
            return True
        except:
            return False

    def bucket_exists(self, bucket: str) -> bool:
        """Prüft ob Bucket existiert"""
        result = self._run_mc("ls", self.alias, check=False)
        return bucket in result.stdout

    def set_bucket_policy(self, bucket: str, policy: str) -> bool:
        """
        Setzt Bucket-Level Policy

        Args:
            policy: "none", "download", "upload", "public"
        """
        try:
            self._run_mc("anonymous", "set", policy, f"{self.alias}/{bucket}")
            return True
        except:
            return False


# =================== User Manager ===================

class MinIOUserManager:
    """
    Verwaltet MinIO Users und deren Credentials

    Features:
    - Erstellt User mit Scope-basierten Policies
    - Speichert Credentials verschlüsselt
    - Credential Rotation
    - Integration mit Clerk Auth
    """

    def __init__(
        self,
        admin_client: MinIOAdminClient,
        credentials_path: str = None
    ):
        self.admin = admin_client
        self.credentials_path = Path(credentials_path or os.path.expanduser("~/.tb_minio_users"))
        self.credentials_path.mkdir(parents=True, exist_ok=True)

        self._credentials_cache: Dict[str, MinIOUserCredentials] = {}
        self._setup_base_policies()

    def _setup_base_policies(self):
        """Erstellt Basis-Policies und Buckets"""
        # Buckets erstellen
        for scope_policy in SCOPE_POLICIES.values():
            self.admin.create_bucket(scope_policy.bucket)

        # Public Read Bucket: Anonymous download erlauben
        self.admin.set_bucket_policy("tb-public-read", "download")

    def _get_credential_path(self, user_id: str) -> Path:
        """Pfad zur verschlüsselten Credential-Datei"""
        safe_id = hashlib.sha256(user_id.encode()).hexdigest()[:16]
        return self.credentials_path / f"{safe_id}.enc"

    def _encrypt_credentials(self, creds: MinIOUserCredentials) -> bytes:
        """Verschlüsselt Credentials für Speicherung"""
        key = Code.DK()()
        if isinstance(key, str):
            key = key.encode()

        data = json.dumps(creds.to_dict()).encode()
        return Code.encrypt_symmetric(data, key)

    def _decrypt_credentials(self, encrypted: bytes) -> MinIOUserCredentials:
        """Entschlüsselt gespeicherte Credentials"""
        key = Code.DK()()
        if isinstance(key, str):
            key = key.encode()

        data = Code.decrypt_symmetric(encrypted, key)
        return MinIOUserCredentials.from_dict(json.loads(data.decode()))

    def _generate_access_key(self, user_id: str) -> str:
        """Generiert Access Key aus User ID"""
        # Format: tb_{first 8 chars of hashed user_id}_{random 4 chars}
        hashed = hashlib.sha256(user_id.encode()).hexdigest()[:8]
        random_part = secrets.token_hex(2)
        return f"tb_{hashed}_{random_part}"

    def _generate_secret_key(self) -> str:
        """Generiert sicheren Secret Key"""
        return secrets.token_urlsafe(32)

    def _create_user_policy(self, user_id: str, scopes: List[Scope]) -> str:
        """
        Erstellt kombinierte Policy für User

        Args:
            user_id: Clerk User ID
            scopes: Liste der erlaubten Scopes

        Returns:
            Policy Name
        """
        policy_name = f"user-{hashlib.sha256(user_id.encode()).hexdigest()[:12]}"

        statements = []

        for scope in scopes:
            scope_policy = SCOPE_POLICIES.get(scope)
            if not scope_policy:
                continue

            # Generiere Policy JSON
            policy_json = scope_policy.to_minio_policy(user_id)
            statements.extend(policy_json["Statement"])

        # Kombinierte Policy
        combined_policy = {
            "Version": "2012-10-17",
            "Statement": statements
        }

        self.admin.create_policy(policy_name, combined_policy)
        return policy_name

    # =================== Public API ===================

    def create_user(
        self,
        user_id: str,
        scopes: List[Scope] = None
    ) -> MinIOUserCredentials:
        """
        Erstellt MinIO User für Clerk User

        Args:
            user_id: Clerk User ID
            scopes: Erlaubte Scopes (default: alle außer SERVER)

        Returns:
            MinIOUserCredentials mit Access/Secret Key
        """
        # Default Scopes für normale User
        if scopes is None:
            scopes = [
                Scope.PUBLIC_READ,
                Scope.PUBLIC_RW,
                Scope.USER_PUBLIC,
                Scope.USER_PRIVATE,
                Scope.MOD_DATA
            ]

        # Prüfe ob User schon existiert
        existing = self.get_credentials(user_id)
        if existing:
            return existing

        # Generiere Credentials
        access_key = self._generate_access_key(user_id)
        secret_key = self._generate_secret_key()

        # Erstelle MinIO User
        self.admin.create_user(access_key, secret_key)

        # Erstelle und weise Policy zu
        policy_name = self._create_user_policy(user_id, scopes)
        self.admin.attach_policy(policy_name, access_key)

        # Speichere Credentials
        now = time.time()
        creds = MinIOUserCredentials(
            user_id=user_id,
            minio_access_key=access_key,
            minio_secret_key=secret_key,
            created_at=now,
            last_rotated=now,
            policies=[policy_name]
        )

        self._save_credentials(creds)
        self._credentials_cache[user_id] = creds

        return creds

    def get_credentials(self, user_id: str) -> Optional[MinIOUserCredentials]:
        """
        Holt Credentials für User

        Args:
            user_id: Clerk User ID

        Returns:
            MinIOUserCredentials oder None
        """
        # Cache check
        if user_id in self._credentials_cache:
            return self._credentials_cache[user_id]

        # Load from file
        cred_path = self._get_credential_path(user_id)
        if cred_path.exists():
            try:
                encrypted = cred_path.read_bytes()
                creds = self._decrypt_credentials(encrypted)
                self._credentials_cache[user_id] = creds
                return creds
            except:
                pass

        return None

    def _save_credentials(self, creds: MinIOUserCredentials):
        """Speichert Credentials verschlüsselt"""
        cred_path = self._get_credential_path(creds.user_id)
        encrypted = self._encrypt_credentials(creds)
        cred_path.write_bytes(encrypted)

    def delete_user(self, user_id: str) -> bool:
        """
        Löscht MinIO User

        Args:
            user_id: Clerk User ID

        Returns:
            True wenn erfolgreich
        """
        creds = self.get_credentials(user_id)
        if not creds:
            return False

        # Entferne Policies
        for policy in creds.policies:
            self.admin.detach_policy(policy, creds.minio_access_key)
            self.admin.delete_policy(policy)

        # Lösche User
        self.admin.delete_user(creds.minio_access_key)

        # Lösche lokale Credentials
        cred_path = self._get_credential_path(user_id)
        if cred_path.exists():
            cred_path.unlink()

        if user_id in self._credentials_cache:
            del self._credentials_cache[user_id]

        return True

    def rotate_credentials(self, user_id: str) -> Optional[MinIOUserCredentials]:
        """
        Rotiert Secret Key für User

        Args:
            user_id: Clerk User ID

        Returns:
            Neue Credentials
        """
        creds = self.get_credentials(user_id)
        if not creds:
            return None

        # Generiere neuen Secret Key
        new_secret = self._generate_secret_key()

        # Update in MinIO (User löschen und neu erstellen)
        # MinIO unterstützt kein direktes Secret Key Update
        self.admin.delete_user(creds.minio_access_key)
        self.admin.create_user(creds.minio_access_key, new_secret)

        # Policies wieder zuweisen
        for policy in creds.policies:
            self.admin.attach_policy(policy, creds.minio_access_key)

        # Update Credentials
        creds.minio_secret_key = new_secret
        creds.last_rotated = time.time()

        self._save_credentials(creds)
        self._credentials_cache[user_id] = creds

        return creds

    def update_scopes(self, user_id: str, scopes: List[Scope]) -> bool:
        """
        Aktualisiert Scopes für User

        Args:
            user_id: Clerk User ID
            scopes: Neue Liste von Scopes

        Returns:
            True wenn erfolgreich
        """
        creds = self.get_credentials(user_id)
        if not creds:
            return False

        # Entferne alte Policies
        for policy in creds.policies:
            self.admin.detach_policy(policy, creds.minio_access_key)
            self.admin.delete_policy(policy)

        # Erstelle neue Policy
        policy_name = self._create_user_policy(user_id, scopes)
        self.admin.attach_policy(policy_name, creds.minio_access_key)

        # Update Credentials
        creds.policies = [policy_name]
        self._save_credentials(creds)

        return True

    def get_or_create_credentials(
        self,
        user_id: str,
        scopes: List[Scope] = None
    ) -> MinIOUserCredentials:
        """
        Holt oder erstellt Credentials

        Args:
            user_id: Clerk User ID
            scopes: Scopes für neuen User

        Returns:
            MinIOUserCredentials
        """
        creds = self.get_credentials(user_id)
        if creds:
            return creds

        return self.create_user(user_id, scopes)


# =================== Integration Helpers ===================

def setup_user_storage(
    clerk_user_id: str,
    minio_alias: str = "local",
    mc_path: str = None,
    minio_endpoint="localhost:9000"
) -> Tuple[MinIOUserCredentials, 'ScopedBlobStorage']:
    """
    Komplettes Setup für einen User

    Args:
        clerk_user_id: Clerk User ID
        minio_alias: MinIO mc Alias
        mc_path: Pfad zu mc Binary

    Returns:
        Tuple von (Credentials, Storage)
    """
    from toolboxv2.utils.extras.db.scoped_storage import ScopedBlobStorage, UserContext

    # Admin Client
    admin = MinIOAdminClient(alias=minio_alias, mc_path=mc_path)

    # User Manager
    manager = MinIOUserManager(admin)

    # Credentials erstellen/holen
    creds = manager.get_or_create_credentials(clerk_user_id)

    # User Context
    user_context = UserContext(
        user_id=clerk_user_id,
        username=clerk_user_id,
        is_authenticated=True
    )

    # Storage
    # Hole MinIO Endpoint aus mc config
    storage = ScopedBlobStorage(
        user_context=user_context,
        minio_endpoint=minio_endpoint,  # TODO: Aus config lesen
        minio_access_key=creds.minio_access_key,
        minio_secret_key=creds.minio_secret_key,
        minio_secure=False
    )

    return creds, storage


# =================== CLI Tool ===================

def main():
    """CLI für User Management"""
    import argparse

    parser = argparse.ArgumentParser(description="MinIO User Manager")
    parser.add_argument("command", choices=["create", "delete", "list", "rotate", "info"])
    parser.add_argument("--user-id", help="Clerk User ID")
    parser.add_argument("--alias", default="local", help="MinIO alias")

    args = parser.parse_args()

    admin = MinIOAdminClient(alias=args.alias)
    manager = MinIOUserManager(admin)

    if args.command == "create":
        if not args.user_id:
            print("Error: --user-id required")
            return

        creds = manager.create_user(args.user_id)
        print(f"Created user for {args.user_id}")
        print(f"  Access Key: {creds.minio_access_key}")
        print(f"  Secret Key: {creds.minio_secret_key}")

    elif args.command == "delete":
        if not args.user_id:
            print("Error: --user-id required")
            return

        if manager.delete_user(args.user_id):
            print(f"Deleted user {args.user_id}")
        else:
            print(f"User {args.user_id} not found")

    elif args.command == "list":
        users = admin.list_users()
        print(f"MinIO Users ({len(users)}):")
        for user in users:
            print(f"  - {user}")

    elif args.command == "rotate":
        if not args.user_id:
            print("Error: --user-id required")
            return

        creds = manager.rotate_credentials(args.user_id)
        if creds:
            print(f"Rotated credentials for {args.user_id}")
            print(f"  New Secret Key: {creds.minio_secret_key}")
        else:
            print(f"User {args.user_id} not found")

    elif args.command == "info":
        if not args.user_id:
            print("Error: --user-id required")
            return

        creds = manager.get_credentials(args.user_id)
        if creds:
            print(f"User: {args.user_id}")
            print(f"  Access Key: {creds.minio_access_key}")
            print(f"  Created: {time.ctime(creds.created_at)}")
            print(f"  Last Rotated: {time.ctime(creds.last_rotated)}")
            print(f"  Policies: {', '.join(creds.policies)}")
        else:
            print(f"User {args.user_id} not found")


if __name__ == "__main__":
    main()
