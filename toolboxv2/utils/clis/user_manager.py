"""
ToolBox V2 - Unified User Manager
Verwaltet CloudM Users mit integrierten MinIO Credentials

Features:
- Unified User Management (CloudM Auth + MinIO)
- MinIO Credentials werden in UserData gespeichert
- User Level Management (Admin, Moderator, User)
- CLI Interface: tb user list, info, set-level, rotate-minio, revoke-minio, delete
"""

import argparse
import json
import os
import secrets
import hashlib
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add toolboxv2 to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from toolboxv2.utils.clis.db_cli_manager import DEFAULT_BASE_DIR
from toolboxv2.utils.extras.db.scoped_storage import Scope

# ToolBoxV2 imports
try:
    from toolboxv2.utils.security.cryp import Code
    from toolboxv2 import App, get_app, TBEF
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


# =================== Constants ===================

# User Levels
LEVEL_GUEST = 0
LEVEL_USER = 1
LEVEL_MODERATOR = 5
LEVEL_ADMIN = 10
LEVEL_ROOT = -1

# Level Names
LEVEL_NAMES = {
    -1: "ROOT",
    0: "GUEST",
    1: "USER",
    5: "MODERATOR",
    10: "ADMIN"
}


# =================== MinIO Admin Client ===================

class MinIOAdminClient:
    """Wrapper für MinIO Admin Operationen via `mc` CLI"""

    def __init__(self, alias: str = "local", mc_path: str = None):
        self.alias = alias
        self.mc = mc_path or self._find_mc()
        if not self.mc:
            print("⚠ Warning: MinIO Client (mc) not found. MinIO features will be limited.")
            self.mc = "mc"  # Placeholder for error messages

    def _find_mc(self) -> Optional[str]:
        """Findet mc Binary"""
        possible_paths = [
            "mc",
            "/usr/local/bin/mc",
            os.path.expanduser("~/.local/bin/mc"),
            os.path.expanduser("~/minio-binaries/mc"),
            "C:\\minio\\mc",
            os.path.join(os.environ.get("LOCALAPPDATA", "."), "minio"),
            os.path.expanduser("~/.local/bin"),
            DEFAULT_BASE_DIR+"/bin/mc"
        ]

        file_name = ".exe" if os.name == "nt" else ""

        for path in possible_paths:
            try:
                full_path = path + file_name
                if os.path.exists(full_path):
                    result = subprocess.run([full_path, "--help"], capture_output=True, text=True, timeout=2)
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

    # User Management
    def create_user(self, access_key: str, secret_key: str) -> bool:
        try:
            self._run_mc("admin", "user", "add", self.alias, access_key, secret_key)
            return True
        except RuntimeError as e:
            if "already exists" in str(e).lower():
                return True
            raise
        except Exception:
            return False

    def delete_user(self, access_key: str) -> bool:
        try:
            self._run_mc("admin", "user", "remove", self.alias, access_key)
            return True
        except Exception:
            return False

    def set_user_status(self, access_key: str, enabled: bool) -> bool:
        status = "enable" if enabled else "disable"
        try:
            self._run_mc("admin", "user", status, self.alias, access_key)
            return True
        except Exception:
            return False

    def list_users(self) -> List[str]:
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

    # Policy Management
    def create_policy(self, name: str, policy_json: dict) -> bool:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(policy_json, f)
            temp_path = f.name

        try:
            self._run_mc("admin", "policy", "create", self.alias, name, temp_path)
            return True
        except RuntimeError as e:
            if "already exists" in str(e).lower():
                self._run_mc("admin", "policy", "remove", self.alias, name, check=False)
                self._run_mc("admin", "policy", "create", self.alias, name, temp_path)
                return True
            raise
        except Exception:
            return False
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass

    def delete_policy(self, name: str) -> bool:
        try:
            self._run_mc("admin", "policy", "remove", self.alias, name)
            return True
        except Exception:
            return False

    def attach_policy(self, policy_name: str, user_access_key: str) -> bool:
        try:
            self._run_mc(
                "admin", "policy", "attach", self.alias,
                policy_name, "--user", user_access_key
            )
            return True
        except Exception:
            return False

    def detach_policy(self, policy_name: str, user_access_key: str) -> bool:
        try:
            self._run_mc(
                "admin", "policy", "detach", self.alias,
                policy_name, "--user", user_access_key
            )
            return True
        except Exception:
            return False


# =================== Unified User Manager ===================

class UnifiedUserManager:
    """
    Verwaltet CloudM Users mit integrierten MinIO Credentials

    - Credentials werden in UserData.minio_credentials gespeichert
    - MinIO Users werden automatisch erstellt/gelöscht
    - User Level Management
    """

    def __init__(self, app: App = None, minio_alias: str = "local"):
        self.app = app
        self.minio_alias = minio_alias

        # Initialize MinIO Admin Client
        self.minio_admin = MinIOAdminClient(alias=minio_alias)

        # Check if app is available
        self.auth_available = bool(app)

    def _encrypt_secret_key(self, secret_key: str) -> str:
        """Verschlüsselt Secret Key für Speicherung in UserData"""
        key = Code.DK()()
        if isinstance(key, str):
            key = key.encode()
        encrypted = Code.encrypt_symmetric(secret_key.encode(), key)
        return encrypted.hex()

    def _decrypt_secret_key(self, encrypted_hex: str) -> str:
        """Entschlüsselt Secret Key aus UserData"""
        key = Code.DK()()
        if isinstance(key, str):
            key = key.encode()
        encrypted = bytes.fromhex(encrypted_hex)
        decrypted = Code.decrypt_symmetric(encrypted, key)
        return decrypted.decode()

    def _generate_access_key(self, user_id: str) -> str:
        """Generiert Access Key aus User ID"""
        hashed = hashlib.sha256(user_id.encode()).hexdigest()[:8]
        random_part = secrets.token_hex(2)
        return f"tb_{hashed}_{random_part}"

    def _generate_secret_key(self) -> str:
        """Generiert sicheren Secret Key"""
        return secrets.token_urlsafe(32)

    def _create_user_policy(self, user_id: str) -> str:
        """Erstellt User-spezifische MinIO Policy"""
        policy_name = f"user-{hashlib.sha256(user_id.encode()).hexdigest()[:12]}"

        # Standard buckets
        buckets = [
            "tb-public-read",
            "tb-public-rw",
            "tb-users-public",
            "tb-users-private",
            "tb-mods"
        ]

        statements = []
        for bucket in buckets:
            if bucket == "tb-users-public":
                resource_prefix = f"{user_id}/*"
            elif bucket == "tb-users-private":
                resource_prefix = f"{user_id}/*"
            elif bucket == "tb-mods":
                resource_prefix = f"*/{user_id}/*"
            else:
                resource_prefix = "*"

            statements.append({
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"],
                "Resource": [
                    f"arn:aws:s3:::{bucket}",
                    f"arn:aws:s3:::{bucket}/{resource_prefix}"
                ]
            })

        policy = {
            "Version": "2012-10-17",
            "Statement": statements
        }

        self.minio_admin.create_policy(policy_name, policy)
        return policy_name

    async def _get_user(self, user_id: str):
        """Lädt User aus Datenbank"""
        if not self.app:
            raise RuntimeError("App not available")

        from toolboxv2.mods.CloudM.auth.user_store import _load_user
        return await _load_user(self.app, user_id)

    async def _resolve_user(self, identifier: str):
        """Resolve user_id or username to UserData"""
        # Try direct user_id lookup first
        user = await self._get_user(identifier)
        if user:
            return user

        # Fallback: search all users by username
        users = await self._list_all_users()
        for u in users:
            if u.get("username") == identifier:
                from toolboxv2.mods.CloudM.auth.user_store import _load_user
                return await _load_user(self.app, u["user_id"])
        return None

    async def _update_user(self, user):
        """Speichert User in Datenbank"""
        if not self.app:
            raise RuntimeError("App not available")

        from toolboxv2.mods.CloudM.auth.user_store import _save_user
        await _save_user(self.app, user)

    async def _list_all_users(self):
        """Listet alle Users aus Datenbank"""
        if not self.app:
            raise RuntimeError("App not available")

        users = []
        try:
            # Hole alle AUTH_USER::* Keys
            from toolboxv2.mods.CloudM.auth.db_helpers import _db_get_raw
            result = await self.app.a_run_any(TBEF.DB.GET, query="AUTH_USER::*", get_results=True)
            if not result.is_error() and result.get():
                raw_data = result.get()
                if isinstance(raw_data, list):
                    for item in raw_data:
                        try:
                            if isinstance(item, bytes):
                                item = item.decode()
                            if isinstance(item, str):
                                data = json.loads(item)
                                users.append(data)
                        except:
                            pass
                elif isinstance(raw_data, dict):
                    for _key, item in raw_data.items():
                        try:
                            if isinstance(item, bytes):
                                item = item.decode()
                            if isinstance(item, str):
                                data = json.loads(item)
                                users.append(data)
                            elif isinstance(item, dict):
                                users.append(item)
                        except:
                            pass
                elif isinstance(raw_data, (bytes, str)):
                    try:
                        if isinstance(raw_data, bytes):
                            raw_data = raw_data.decode()
                        data = json.loads(raw_data)
                        users.append(data)
                    except:
                        pass
        except Exception as e:
            print(f"⚠ Error listing users: {e}")

        return users

    # =================== Public API ===================

    async def ensure_minio_credentials(self, user_id: str) -> dict:
        """
        Stellt sicher dass ein User MinIO Credentials hat

        Returns:
            Dict mit access_key und secret_key (entschlüsselt)
        """
        # Hole User (resolve username or user_id)
        user = await self._resolve_user(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")

        # Prüfe ob Credentials existieren
        creds = user.minio_credentials
        if creds and creds.access_key and creds.secret_key:
            # Credentials existieren
            return {
                "access_key": creds.access_key,
                "secret_key": self._decrypt_secret_key(creds.secret_key),
                "created_at": creds.created_at,
                "last_rotated": creds.last_rotated,
                "enabled": creds.enabled
            }

        # Neue Credentials erstellen
        access_key = self._generate_access_key(user_id)
        secret_key = self._generate_secret_key()

        # MinIO User erstellen
        self.minio_admin.create_user(access_key, secret_key)

        # Policy erstellen und zuweisen
        policy_name = self._create_user_policy(user_id)
        self.minio_admin.attach_policy(policy_name, access_key)

        # In UserData speichern
        from toolboxv2.mods.CloudM.auth.models import MinIOCredentials
        user.minio_credentials = MinIOCredentials(
            access_key=access_key,
            secret_key=self._encrypt_secret_key(secret_key),
            created_at=time.time(),
            last_rotated=time.time(),
            policies=[policy_name],
            enabled=True
        )

        await self._update_user(user)

        return {
            "access_key": access_key,
            "secret_key": secret_key,
            "created_at": user.minio_credentials.created_at,
            "last_rotated": user.minio_credentials.last_rotated,
            "enabled": True
        }

    async def rotate_minio_credentials(self, user_id: str) -> dict:
        """
        Rotiert MinIO Credentials für einen User

        Returns:
            Dict mit neuen access_key und secret_key
        """
        user = await self._resolve_user(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")

        creds = user.minio_credentials
        if not creds or not creds.access_key:
            # Noch keine Credentials, neu erstellen
            return await self.ensure_minio_credentials(user_id)

        # Neuen Secret Key generieren
        new_secret = self._generate_secret_key()
        old_access_key = creds.access_key

        # MinIO User aktualisieren (löschen und neu erstellen)
        self.minio_admin.delete_user(old_access_key)
        self.minio_admin.create_user(old_access_key, new_secret)

        # Policies wieder zuweisen
        for policy in creds.policies:
            self.minio_admin.attach_policy(policy, old_access_key)

        # In UserData speichern
        creds.secret_key = self._encrypt_secret_key(new_secret)
        creds.last_rotated = time.time()

        await self._update_user(user)

        return {
            "access_key": old_access_key,
            "secret_key": new_secret,
            "created_at": creds.created_at,
            "last_rotated": creds.last_rotated
        }

    async def revoke_minio_credentials(self, user_id: str) -> bool:
        """
        Entzieht MinIO Credentials von einem User
        """
        user = await self._resolve_user(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")

        creds = user.minio_credentials
        if not creds or not creds.access_key:
            return True  # Keine Credentials zu entziehen

        # Policies entfernen
        for policy in creds.policies:
            self.minio_admin.detach_policy(policy, creds.access_key)
            self.minio_admin.delete_policy(policy)

        # MinIO User löschen
        self.minio_admin.delete_user(creds.access_key)

        # In UserData löschen
        from toolboxv2.mods.CloudM.auth.models import MinIOCredentials
        user.minio_credentials = MinIOCredentials(enabled=False)
        await self._update_user(user)

        return True

    async def set_user_level(self, user_id: str, level: int) -> bool:
        """Setzt User Level"""
        user = await self._resolve_user(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")

        user.level = level
        await self._update_user(user)
        return True

    async def get_user_info(self, user_id: str) -> dict:
        """Gibt vollständige User Info zurück"""
        user = await self._resolve_user(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")

        # MinIO Status
        has_minio = bool(user.minio_credentials and user.minio_credentials.access_key)

        return {
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "level": user.level,
            "level_name": LEVEL_NAMES.get(user.level, "UNKNOWN"),
            "created_at": user.created_at,
            "last_login": user.last_login,
            "minio_enabled": has_minio,
            "minio_access_key": user.minio_credentials.access_key if has_minio else None,
            "minio_created_at": user.minio_credentials.created_at if has_minio else None,
            "minio_last_rotated": user.minio_credentials.last_rotated if has_minio else None,
            "oauth_providers": list(user.oauth_providers.keys()) if user.oauth_providers else [],
            "passkeys_count": len(user.passkeys) if user.passkeys else 0
        }

    async def list_all_users_info(self) -> List[dict]:
        """Listet alle Users mit Info"""
        users = await self._list_all_users()
        result = []
        for user_dict in users:
            user_id = user_dict.get("user_id")
            if user_id:
                try:
                    from toolboxv2.mods.CloudM.auth.models import UserData
                    user_data = UserData.from_dict(user_dict)
                    has_minio = bool(user_data.minio_credentials and user_data.minio_credentials.access_key)

                    result.append({
                        "user_id": user_data.user_id,
                        "username": user_data.username,
                        "email": user_data.email,
                        "level": user_data.level,
                        "level_name": LEVEL_NAMES.get(user_data.level, "UNKNOWN"),
                        "created_at": user_data.created_at,
                        "last_login": user_data.last_login,
                        "minio_enabled": has_minio,
                        "minio_access_key": user_data.minio_credentials.access_key if has_minio else None,
                        "oauth_providers": list(user_data.oauth_providers.keys()) if user_data.oauth_providers else [],
                        "passkeys_count": len(user_data.passkeys) if user_data.passkeys else 0
                    })
                except Exception as e:
                    print(f"⚠ Error parsing user: {e}")
        return result


# =================== CLI Commands ===================

async def cmd_list(args):
    """Listet alle Users"""
    if TOOLBOX_AVAILABLE:
        app = get_app("CloudM.Export")
    else:
        print("❌ ToolBoxV2 not available")
        return 1

    manager = UnifiedUserManager(app=app, minio_alias=args.alias)

    try:
        users = await manager.list_all_users_info()
        print(f"\n📋 Users ({len(users)}):\n")
        print(f"{'Username':<20} {'Email':<30} {'Level':<12} {'MinIO':<8}")
        print("-" * 70)
        for user in users:
            minio_status = "✓" if user["minio_enabled"] else "✗"
            print(f"{user['username']:<20} {user['email']:<30} {user['level_name']:<12} {minio_status:<8}")
        print("-" * 70)
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


async def cmd_info(args):
    """Zeigt User Info"""
    if TOOLBOX_AVAILABLE:
        app = get_app("CloudM.Export")
    else:
        print("❌ ToolBoxV2 not available")
        return 1

    manager = UnifiedUserManager(app=app, minio_alias=args.alias)

    try:
        user_id = args.user_id or args.username
        info = await manager.get_user_info(user_id)

        print(f"\n👤 User Information:\n")
        print(f"  Username:    {info['username']}")
        print(f"  Email:       {info['email']}")
        print(f"  User ID:     {info['user_id']}")
        print(f"  Level:       {info['level_name']} ({info['level']})")
        print(f"  Created:     {time.ctime(info['created_at'])}")
        print(f"  Last Login:  {time.ctime(info['last_login']) if info['last_login'] else 'Never'}")
        print(f"\n🔐 Security:\n")
        print(f"  MinIO:       {'Enabled' if info['minio_enabled'] else 'Disabled'}")
        if info['minio_enabled']:
            print(f"  Access Key:  {info['minio_access_key']}")
            print(f"  Created:     {time.ctime(info['minio_created_at'])}")
            print(f"  Rotated:     {time.ctime(info['minio_last_rotated'])}")
        print(f"  OAuth:       {', '.join(info['oauth_providers']) if info['oauth_providers'] else 'None'}")
        print(f"  Passkeys:    {info['passkeys_count']}")
        print()
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


async def cmd_set_level(args):
    """Setzt User Level"""
    if TOOLBOX_AVAILABLE:
        app = get_app("CloudM.Export")
    else:
        print("❌ ToolBoxV2 not available")
        return 1

    manager = UnifiedUserManager(app=app, minio_alias=args.alias)

    # Level string zu int konvertieren
    level_map = {
        "guest": LEVEL_GUEST,
        "user": LEVEL_USER,
        "moderator": LEVEL_MODERATOR,
        "admin": LEVEL_ADMIN,
        "root": LEVEL_ROOT
    }
    if args.level.lower() in level_map:
        level = level_map[args.level.lower()]
    else:
        try:
            level = int(args.level)
        except:
            print(f"❌ Invalid level: {args.level}")
            return 1

    try:
        user_id = args.user_id or args.username
        await manager.set_user_level(user_id, level)
        print(f"✅ Set {user_id} level to {LEVEL_NAMES.get(level, level)}")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


async def cmd_rotate_minio(args):
    """Rotiert MinIO Credentials"""
    if TOOLBOX_AVAILABLE:
        app = get_app("CloudM.Export")
    else:
        print("❌ ToolBoxV2 not available")
        return 1

    manager = UnifiedUserManager(app=app, minio_alias=args.alias)

    try:
        user_id = args.user_id or args.username
        creds = await manager.rotate_minio_credentials(user_id)
        print(f"✅ Rotated MinIO credentials for {user_id}")
        print(f"   Access Key:  {creds['access_key']}")
        print(f"   Secret Key:  {creds['secret_key']}")
        print(f"   Rotated:     {time.ctime(creds['last_rotated'])}")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


async def cmd_revoke_minio(args):
    """Entzieht MinIO Credentials"""
    if TOOLBOX_AVAILABLE:
        app = get_app("CloudM.Export")
    else:
        print("❌ ToolBoxV2 not available")
        return 1

    manager = UnifiedUserManager(app=app, minio_alias=args.alias)

    try:
        user_id = args.user_id or args.username
        if await manager.revoke_minio_credentials(user_id):
            print(f"✅ Revoked MinIO credentials for {user_id}")
            return 0
        else:
            print(f"❌ Failed to revoke credentials")
            return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


async def cmd_delete(args):
    """Löscht User"""
    if TOOLBOX_AVAILABLE:
        app = get_app("CloudM.Export")
    else:
        print("❌ ToolBoxV2 not available")
        return 1

    manager = UnifiedUserManager(app=app, minio_alias=args.alias)

    # Bestätigung
    if not args.force:
        user_id = args.user_id or args.username
        confirm = input(f"Delete user {user_id}? This cannot be undone! (yes/no): ").strip().lower()
        if confirm != "yes":
            print("Cancelled")
            return 0

    try:
        user_id = args.user_id or args.username

        # Erst MinIO Credentials entziehen
        try:
            await manager.revoke_minio_credentials(user_id)
        except:
            pass

        # Load user to delete indexes
        from toolboxv2.mods.CloudM.auth.user_store import _load_user
        from toolboxv2.mods.CloudM.auth.db_helpers import _db_delete

        # Check via DB
        result = await app.a_run_any(TBEF.DB.GET, query=f"AUTH_USER::{user_id}", get_results=True)

        if result.is_error() or not result.get():
            print(f"❌ User {user_id} not found")
            return 1

        # Load user for index cleanup
        user = await _load_user(app, user_id)
        if user:
            # Delete email index
            if user.email:
                await _db_delete(app, f"AUTH_USER_EMAIL::{user.email}")
            # Delete provider indexes
            for prov, pdata in user.oauth_providers.items():
                pid = pdata.get("provider_id", "")
                if pid:
                    await _db_delete(app, f"AUTH_USER_PROVIDER::{prov}::{pid}")
            # Delete passkey indexes
            for pk in user.passkeys:
                cid = pk.get("credential_id", "")
                if cid:
                    await _db_delete(app, f"AUTH_USER_PROVIDER::passkey::{cid}")

        # Delete user
        await app.a_run_any(TBEF.DB.DELETE, query=f"AUTH_USER::{user_id}")

        print(f"✅ Deleted user {user_id}")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


async def main():
    """CLI Main (async)"""
    parser = argparse.ArgumentParser(
        description="ToolBoxV2 Unified User Manager",
        prog="tb user",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tb user list                           # List all users
  tb user info --username john            # Show user info
  tb user set-level john admin            # Set user level
  tb user rotate-minio john               # Rotate MinIO credentials
  tb user revoke-minio john               # Revoke MinIO access
  tb user delete --username john          # Delete user

Levels:
  guest, user, moderator, admin, root
        """
    )

    parser.add_argument("--alias", default="local", help="MinIO alias (default: local)")

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # list
    subparsers.add_parser("list", help="List all users")

    # info
    info_parser = subparsers.add_parser("info", help="Show user info")
    info_parser.add_argument("--user-id", help="User ID")
    info_parser.add_argument("--username", help="Username")

    # set-level
    level_parser = subparsers.add_parser("set-level", help="Set user level")
    level_parser.add_argument("--user-id", help="User ID")
    level_parser.add_argument("--username", help="Username")
    level_parser.add_argument("level", help="Level: guest, user, moderator, admin, root")

    # rotate-minio
    rotate_parser = subparsers.add_parser("rotate-minio", help="Rotate MinIO credentials")
    rotate_parser.add_argument("--user-id", help="User ID")
    rotate_parser.add_argument("--username", help="Username")

    # revoke-minio
    revoke_parser = subparsers.add_parser("revoke-minio", help="Revoke MinIO credentials")
    revoke_parser.add_argument("--user-id", help="User ID")
    revoke_parser.add_argument("--username", help="Username")

    # delete
    delete_parser = subparsers.add_parser("delete", help="Delete user")
    delete_parser.add_argument("--user-id", help="User ID")
    delete_parser.add_argument("--username", help="Username")
    delete_parser.add_argument("--force", action="store_true", help="Skip confirmation")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Dispatch to async command
    if args.command == "list":
        return await cmd_list(args)
    elif args.command == "info":
        return await cmd_info(args)
    elif args.command == "set-level":
        return await cmd_set_level(args)
    elif args.command == "rotate-minio":
        return await cmd_rotate_minio(args)
    elif args.command == "revoke-minio":
        return await cmd_revoke_minio(args)
    elif args.command == "delete":
        return await cmd_delete(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))
