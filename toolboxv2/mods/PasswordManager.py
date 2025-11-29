#!/usr/bin/env python3
"""
ToolBox Password Manager Module
Advanced password management with blob storage, device key encryption, and 2FA support
api available at http://localhost:8080/api/PasswordManager/{function_name}
"""

import json
import csv
import io
import base64
import secrets
import hashlib
import time
import re
import urllib.parse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
from pathlib import Path

from toolboxv2 import App, Result, MainTool, get_app
from toolboxv2.utils.security.cryp import Code, DEVICE_KEY
from toolboxv2.utils.extras.blobs import BlobStorage, BlobFile
from toolboxv2.mods.DB.blob_instance import BlobDB

Name = "PasswordManager"
export = get_app(from_="PasswordManager.EXPORT").tb


_pm_instance: Optional['PasswordManagerCore'] = None


def get_pm_core(app: App) -> 'PasswordManagerCore':
    """
    Initialisiert und gibt eine Singleton-Instanz des PasswordManagerCore zurück.
    Dies verhindert das wiederholte Laden der Datenbank bei jeder Anfrage.
    """
    global _pm_instance
    if _pm_instance is None:
        try:
            _pm_instance = PasswordManagerCore(app)
        except Exception as e:
            app.logger.critical(f"FATAL: PasswordManagerCore konnte nicht initialisiert werden: {e}")
            # In einem realen Szenario könnte hier ein Fallback oder ein Neustart-Mechanismus ausgelöst werden.
            raise
    return _pm_instance


@dataclass
class PasswordEntry:
    """Secure password entry data structure"""
    id: str
    url: str
    username: str
    password: str
    title: str = ""
    notes: str = ""
    totp_secret: str = ""
    totp_issuer: str = ""
    totp_account: str = ""
    folder: str = "Default"
    tags: List[str] = None
    favorite: bool = False
    created_at: float = None
    updated_at: float = None
    last_used: float = None
    password_history: List[Dict] = None
    custom_fields: Dict[str, str] = None
    breach_detected: bool = False
    auto_fill_enabled: bool = True

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.password_history is None:
            self.password_history = []
        if self.custom_fields is None:
            self.custom_fields = {}
        if self.created_at is None:
            self.created_at = time.time()
        if self.updated_at is None:
            self.updated_at = self.created_at
        if self.id is None or self.id == "":
            self.id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate unique ID for password entry"""
        data = f"{self.url}{self.username}{self.created_at}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'PasswordEntry':
        """Create from dictionary"""
        return cls(**data)

    def update_password(self, new_password: str):
        """Update password and maintain history"""
        if self.password != new_password:
            # Add old password to history
            self.password_history.append({
                'password': self.password,
                'changed_at': time.time()
            })
            # Keep only last 5 passwords
            self.password_history = self.password_history[-5:]

            self.password = new_password
            self.updated_at = time.time()

    def get_domain(self) -> str:
        """Extract domain from URL"""
        try:
            parsed = urllib.parse.urlparse(self.url)
            return parsed.netloc.lower()
        except:
            return self.url.lower()


@dataclass
class ImportResult:
    """Result of password import operation"""
    success: bool
    imported_count: int = 0
    skipped_count: int = 0
    error_count: int = 0
    errors: List[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class PasswordManagerCore:
    """Core password management functionality"""

    def __init__(self, app: App):
        self.app = app
        self.device_key = DEVICE_KEY()
        self.storage_client = None
        self.password_db = None
        self.blob_path = "password_manager/vault.json"
        self._initialize_storage()

    def _initialize_storage(self):
        """Initialize blob storage for passwords"""
        try:
            # Get blob storage servers from config
            self.storage_client = self.app.root_blob_storage

            # Initialize encrypted password database
            self.password_db = BlobDB()
            result = self.password_db.initialize(
                db_path=self.blob_path,
                key=self.device_key,
                storage_client=self.storage_client
            )

            if result.is_error():
                raise Exception(f"Failed to initialize password storage: {result.info}")

        except Exception as e:
            self.app.logger.error(f"Password storage initialization failed: {e}")
            raise

    def add_password(self, entry: PasswordEntry) -> Result:
        """Add new password entry"""
        try:
            # Validate entry
            if not entry.url or not entry.username:
                return Result.default_user_error("URL and username are required")

            # Check for duplicates
            existing = self.get_password_by_url_username(entry.url, entry.username)
            if existing.is_data():
                return Result.default_user_error("Password entry already exists")

            # Store in database
            self.password_db.set(entry.id, entry.to_dict())
            self.password_db.exit()  # Save to blob storage

            return Result.ok(data=entry.to_dict(), info="Password added successfully")

        except Exception as e:
            return Result.default_internal_error(f"Failed to add password: {e}")

    def get_password(self, entry_id: str) -> Result:
        """Get password entry by ID"""
        try:
            if not self.password_db.if_exist(entry_id):
                return Result.default_user_error("Password entry not found")

            entry_data = self.password_db.get(entry_id)
            entry = PasswordEntry.from_dict(entry_data)

            # Update last used timestamp
            entry.last_used = time.time()
            self.password_db.set(entry.id, entry.to_dict())
            self.password_db.exit()

            return Result.ok(data=entry.to_dict())

        except Exception as e:
            return Result.default_internal_error(f"Failed to get password: {e}")

    def get_password_by_url_username(self, url: str, username: str) -> Result:
        """Get password entry by URL and username"""
        try:
            domain = self._extract_domain(url)

            for entry_data in self.password_db.get('all'):
                entry = PasswordEntry.from_dict(entry_data)
                if (entry.get_domain() == domain and
                    entry.username.lower() == username.lower()):
                    return Result.ok(data=entry.to_dict())

            return Result.default_user_error("Password entry not found")

        except Exception as e:
            return Result.default_internal_error(f"Failed to find password: {e}")

    def search_passwords(self, query: str, limit: int = 50) -> Result:
        """Search password entries"""
        try:
            query = query.lower()
            results = []

            for entry_data in self.password_db.get('all'):
                entry = PasswordEntry.from_dict(entry_data)

                # Search in multiple fields
                searchable_text = f"{entry.title} {entry.url} {entry.username} {entry.notes}".lower()
                if query in searchable_text or any(query in tag.lower() for tag in entry.tags):
                    results.append(entry.to_dict())

                if len(results) >= limit:
                    break

            # Sort by relevance (title matches first, then URL, etc.)
            results.sort(key=lambda x: (
                query not in x['title'].lower(),
                query not in x['url'].lower(),
                query not in x['username'].lower()
            ))

            return Result.ok(data=results)

        except Exception as e:
            return Result.default_internal_error(f"Search failed: {e}")

    def update_password(self, entry_id: str, updates: Dict) -> Result:
        """Update password entry"""
        try:
            if not self.password_db.if_exist(entry_id):
                return Result.default_user_error("Password entry not found")

            entry_data = self.password_db.get(entry_id)
            entry = PasswordEntry.from_dict(entry_data)

            # Update fields
            for key, value in updates.items():
                if hasattr(entry, key):
                    if key == 'password':
                        entry.update_password(value)
                    else:
                        setattr(entry, key, value)

            entry.updated_at = time.time()
            self.password_db.set(entry.id, entry.to_dict())
            self.password_db.exit()

            return Result.ok(data=entry.to_dict(), info="Password updated successfully")

        except Exception as e:
            return Result.default_internal_error(f"Failed to update password: {e}")

    def delete_password(self, entry_id: str) -> Result:
        """Delete password entry"""
        try:
            if not self.password_db.if_exist(entry_id):
                return Result.default_user_error("Password entry not found")

            self.password_db.delete(entry_id)
            self.password_db.exit()

            return Result.ok(info="Password deleted successfully")

        except Exception as e:
            return Result.default_internal_error(f"Failed to delete password: {e}")

    def list_passwords(self, folder: str = None, limit: int = 100) -> Result:
        """List password entries"""
        try:
            results = []

            for entry_data in self.password_db.get('all'):
                entry = PasswordEntry.from_dict(entry_data)

                if folder and entry.folder != folder:
                    continue

                # Return safe data (no actual passwords)
                safe_data = entry.to_dict()
                safe_data['password'] = '***'  # Hide password in list
                results.append(safe_data)

                if len(results) >= limit:
                    break

            # Sort by title
            results.sort(key=lambda x: x['title'].lower())

            return Result.ok(data=results)

        except Exception as e:
            return Result.default_internal_error(f"Failed to list passwords: {e}")

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            parsed = urllib.parse.urlparse(url)
            return parsed.netloc.lower()
        except:
            return url.lower()


# Export functions for ToolBox integration
@export(mod_name=Name, api=True)
def add_password(app: App, url: str, username: str, password: str,
                title: str = "", notes: str = "", folder: str = "Default") -> Result:
    """Add new password entry"""
    try:
        pm = get_pm_core(app)
        entry = PasswordEntry(
            id="",  # Will be auto-generated
            url=url,
            username=username,
            password=password,
            title=title or url,
            notes=notes,
            folder=folder
        )
        return pm.add_password(entry)
    except Exception as e:
        return Result.default_internal_error(f"Add password failed: {e}")


@export(mod_name=Name, api=True)
def get_password(app: App, entry_id: str) -> Result:
    """Get password entry by ID"""
    try:
        pm = get_pm_core(app)
        return pm.get_password(entry_id)
    except Exception as e:
        return Result.default_internal_error(f"Get password failed: {e}")


@export(mod_name=Name, api=True)
def search_passwords(app: App, query: str, limit: int = 50) -> Result:
    """Search password entries"""
    try:
        pm = get_pm_core(app)
        return pm.search_passwords(query, limit)
    except Exception as e:
        return Result.default_internal_error(f"Search passwords failed: {e}")


@export(mod_name=Name, api=True)
def list_passwords(app: App, folder: str = None, limit: int = 100) -> Result:
    """List password entries"""
    try:
        pm = get_pm_core(app)
        return pm.list_passwords(folder, limit)
    except Exception as e:
        return Result.default_internal_error(f"List passwords failed: {e}")


@export(mod_name=Name, api=True)
def generate_password(app: App, length: int = 16, include_symbols: bool = True,
                      include_numbers: bool = True, include_uppercase: bool = True,
                      include_lowercase: bool = True, exclude_ambiguous: bool = True) -> Result:
    """Generate secure password"""
    try:
        if not 4 <= length <= 128:
            return Result.default_user_error("Password length must be between 4 and 128")

        # Definiere Zeichensätze
        LOWERCASE = "abcdefghijkmnopqrstuvwxyz"
        UPPERCASE = "ABCDEFGHJKLMNPQRSTUVWXYZ"
        NUMBERS = "23456789"
        SYMBOLS = "!@#$%^&*()_+-=[]{}|;:,.<>?"

        AMBIGUOUS_CHARS = "0OIl"

        chars = ""
        if include_lowercase:
            chars += LOWERCASE
        if include_uppercase:
            chars += UPPERCASE
        if include_numbers:
            chars += NUMBERS
        if include_symbols:
            chars += SYMBOLS

        if not exclude_ambiguous:
            # Füge mehrdeutige Zeichen nur hinzu, wenn explizit gewünscht
            if include_lowercase or include_uppercase:
                chars += "il"
            if include_uppercase:
                chars += "O"
            if include_numbers:
                chars += "0"
            if include_uppercase:
                chars += "I"

        if not chars:
            return Result.default_user_error("No character types selected for password generation")

        password = ''.join(secrets.choice(chars) for _ in range(length))

        # Stelle sicher, dass jeder ausgewählte Zeichentyp mindestens einmal vorkommt
        # (Erhöht die Komplexität und verhindert einfache Passwörter wie 'aaaaaa')
        # Diese Logik kann bei Bedarf hinzugefügt werden, ist aber für den Moment optional.

        return Result.ok(data={'password': password}, info="Password generated successfully")

    except Exception as e:
        return Result.default_internal_error(f"Password generation failed: {e}")

class PasswordImporter:
    """Universal password manager import parser"""

    def __init__(self, app: App):
        self.app = app
        self.pm_core = PasswordManagerCore(app)

    def import_from_file(self, file_content: str, file_format: str,
                        folder: str = "Imported") -> ImportResult:
        """Import passwords from various formats"""
        try:
            if file_format.lower() == 'csv':
                return self._import_csv(file_content, folder)
            elif file_format.lower() == 'json':
                return self._import_json(file_content, folder)
            elif file_format.lower() == 'chrome':
                return self._import_chrome_csv(file_content, folder)
            elif file_format.lower() == 'firefox':
                return self._import_firefox_csv(file_content, folder)
            elif file_format.lower() == 'lastpass':
                return self._import_lastpass_csv(file_content, folder)
            elif file_format.lower() == 'bitwarden':
                return self._import_bitwarden_json(file_content, folder)
            elif file_format.lower() == '1password':
                return self._import_1password_csv(file_content, folder)
            else:
                return ImportResult(
                    success=False,
                    errors=[f"Unsupported format: {file_format}"]
                )
        except Exception as e:
            return ImportResult(
                success=False,
                errors=[f"Import failed: {str(e)}"]
            )

    def _import_csv(self, content: str, folder: str) -> ImportResult:
        """Import generic CSV format"""
        result = ImportResult(success=True)

        try:
            reader = csv.DictReader(io.StringIO(content))

            for row in reader:
                try:
                    # Map common field names
                    url = row.get('url', row.get('URL', row.get('website', '')))
                    username = row.get('username', row.get('Username', row.get('login', '')))
                    password = row.get('password', row.get('Password', ''))
                    title = row.get('title', row.get('Title', row.get('name', url)))
                    notes = row.get('notes', row.get('Notes', row.get('note', '')))

                    if not url or not username or not password:
                        result.skipped_count += 1
                        result.warnings.append(f"Skipped entry: missing required fields")
                        continue

                    entry = PasswordEntry(
                        id="",
                        url=url,
                        username=username,
                        password=password,
                        title=title,
                        notes=notes,
                        folder=folder
                    )

                    add_result = self.pm_core.add_password(entry)
                    if add_result.is_ok():
                        result.imported_count += 1
                    else:
                        result.error_count += 1
                        result.errors.append(f"Failed to add {url}: {add_result.info}")

                except Exception as e:
                    result.error_count += 1
                    result.errors.append(f"Error processing row: {str(e)}")

        except Exception as e:
            result.success = False
            result.errors.append(f"CSV parsing failed: {str(e)}")

        return result

    def _import_chrome_csv(self, content: str, folder: str) -> ImportResult:
        """Import Chrome password export CSV"""
        result = ImportResult(success=True)

        try:
            reader = csv.DictReader(io.StringIO(content))

            for row in reader:
                try:
                    url = row.get('url', '')
                    username = row.get('username', '')
                    password = row.get('password', '')

                    if not url or not username or not password:
                        result.skipped_count += 1
                        continue

                    # Logik zum Aktualisieren oder Hinzufügen
                    existing_entry_result = self.pm_core.get_password_by_url_username(url, username)
                    if existing_entry_result.is_data():
                        # Eintrag existiert -> aktualisieren
                        entry_id = existing_entry_result.get()['id']
                        updates = {'password': password}
                        update_result = self.pm_core.update_password(entry_id, updates)
                        if update_result.is_ok():
                            result.imported_count += 1
                            result.warnings.append(f"Updated existing entry for {url}")
                        else:
                            result.error_count += 1
                            result.errors.append(f"Failed to update {url}: {update_result.info}")
                    else:
                        # Eintrag existiert nicht -> neu hinzufügen
                        entry = PasswordEntry(
                            id=row.get('name', ''),
                            url=url,
                            username=username,
                            password=password,
                            title=self._extract_site_name(url),
                            folder=folder
                        )
                        add_result = self.pm_core.add_password(entry)
                        if add_result.is_ok():
                            result.imported_count += 1
                        else:
                            result.error_count += 1
                            result.errors.append(f"Failed to add {url}: {add_result.info}")

                except Exception as e:
                    result.error_count += 1
                    result.errors.append(f"Error processing Chrome entry: {str(e)}")

        except Exception as e:
            result.success = False
            result.errors.append(f"Chrome CSV parsing failed: {str(e)}")

        return result

    def _import_firefox_csv(self, content: str, folder: str) -> ImportResult:
        """Import Firefox password export CSV"""
        result = ImportResult(success=True)

        try:
            reader = csv.DictReader(io.StringIO(content))

            for row in reader:
                try:
                    url = row.get('url', '')
                    username = row.get('username', '')
                    password = row.get('password', '')

                    if not url or not username or not password:
                        result.skipped_count += 1
                        continue

                    entry = PasswordEntry(
                        id="",
                        url=url,
                        username=username,
                        password=password,
                        title=self._extract_site_name(url),
                        folder=folder,
                        created_at=self._parse_firefox_date(row.get('timeCreated', '')),
                        updated_at=self._parse_firefox_date(row.get('timePasswordChanged', ''))
                    )

                    add_result = self.pm_core.add_password(entry)
                    if add_result.is_ok():
                        result.imported_count += 1
                    else:
                        result.error_count += 1
                        result.errors.append(f"Failed to add {url}: {add_result.info}")

                except Exception as e:
                    result.error_count += 1
                    result.errors.append(f"Error processing Firefox entry: {str(e)}")

        except Exception as e:
            result.success = False
            result.errors.append(f"Firefox CSV parsing failed: {str(e)}")

        return result

    def _extract_site_name(self, url: str) -> str:
        """Extract readable site name from URL"""
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain.split('.')[0].title()
        except:
            return url

    def _parse_firefox_date(self, date_str: str) -> float:
        """Parse Firefox timestamp"""
        try:
            if date_str:
                # Firefox uses microseconds since epoch
                return float(date_str) / 1000000
        except:
            pass
        return time.time()


@export(mod_name=Name, api=True)
def import_passwords(app: App, file_content: str, file_format: str,
                    folder: str = "Imported") -> Result:
    """Import passwords from file"""
    try:
        importer = PasswordImporter(app)
        result = importer.import_from_file(file_content, file_format, folder)

        return Result.ok(
            data=asdict(result),
            info=f"Import completed: {result.imported_count} imported, "
                 f"{result.skipped_count} skipped, {result.error_count} errors"
        )
    except Exception as e:
        return Result.default_internal_error(f"Import failed: {e}")


class TOTPManager:
    """Time-based One-Time Password (2FA) manager"""

    @staticmethod
    def generate_totp_code(secret: str, time_step: int = 30) -> str:
        """Generate TOTP code from secret"""
        try:
            import hmac
            import struct

            # Decode base32 secret
            secret = secret.upper().replace(' ', '')
            # Add padding if needed
            missing_padding = len(secret) % 8
            if missing_padding:
                secret += '=' * (8 - missing_padding)

            secret_bytes = base64.b32decode(secret)

            # Get current time step
            current_time = int(time.time() // time_step)

            # Generate HMAC
            time_bytes = struct.pack('>Q', current_time)
            hmac_hash = hmac.new(secret_bytes, time_bytes, hashlib.sha1).digest()

            # Extract dynamic binary code
            offset = hmac_hash[-1] & 0xf
            code = struct.unpack('>I', hmac_hash[offset:offset + 4])[0]
            code &= 0x7fffffff
            code %= 1000000

            return f"{code:06d}"

        except Exception as e:
            raise Exception(f"TOTP generation failed: {e}")

    @staticmethod
    def parse_totp_uri(uri: str) -> Dict[str, str]:
        """Parse TOTP URI (otpauth://totp/...)"""
        try:
            if not uri.startswith('otpauth://totp/'):
                raise ValueError("Invalid TOTP URI format")

            parsed = urllib.parse.urlparse(uri)
            params = urllib.parse.parse_qs(parsed.query)

            # Extract account name from path
            account = parsed.path.lstrip('/')
            if ':' in account:
                issuer, account = account.split(':', 1)
            else:
                issuer = params.get('issuer', [''])[0]

            return {
                'secret': params.get('secret', [''])[0],
                'issuer': issuer,
                'account': account,
                'algorithm': params.get('algorithm', ['SHA1'])[0],
                'digits': params.get('digits', ['6'])[0],
                'period': params.get('period', ['30'])[0]
            }

        except Exception as e:
            raise Exception(f"TOTP URI parsing failed: {e}")

    @staticmethod
    def generate_qr_code_uri(secret: str, account: str, issuer: str = "") -> str:
        """Generate TOTP QR code URI"""
        try:
            account_name = f"{issuer}:{account}" if issuer else account
            params = {
                'secret': secret,
                'issuer': issuer,
                'algorithm': 'SHA1',
                'digits': '6',
                'period': '30'
            }

            query_string = urllib.parse.urlencode(params)
            return f"otpauth://totp/{urllib.parse.quote(account_name)}?{query_string}"

        except Exception as e:
            raise Exception(f"QR code URI generation failed: {e}")


@export(mod_name=Name, api=True)
def generate_totp_code(app: App, entry_id: str) -> Result:
    """Generate TOTP code for password entry"""
    try:
        pm = get_pm_core(app)
        entry_result = pm.get_password(entry_id)

        if entry_result.is_error():
            return entry_result

        entry_data = entry_result.get()
        totp_secret = entry_data.get('totp_secret', '')

        if not totp_secret:
            return Result.default_user_error("No TOTP secret configured for this entry")

        code = TOTPManager.generate_totp_code(totp_secret)

        # Calculate time remaining
        time_remaining = 30 - (int(time.time()) % 30)

        return Result.ok(data={
            'code': code,
            'time_remaining': time_remaining,
            'issuer': entry_data.get('totp_issuer', ''),
            'account': entry_data.get('totp_account', entry_data.get('username', ''))
        })

    except Exception as e:
        return Result.default_internal_error(f"TOTP generation failed: {e}")
# PasswordManager.py

@export(mod_name=Name, api=True)
def delete_password(app: App, entry_id: str) -> Result:
    """Deletes a password entry by its ID."""
    try:
        pm = get_pm_core(app)
        return pm.delete_password(entry_id)
    except Exception as e:
        return Result.default_internal_error(f"Failed to delete password: {e}")

@export(mod_name=Name, api=True)
def add_totp_secret(app: App, entry_id: str, secret: str,
                   issuer: str = "", account: str = "") -> Result:
    """Add TOTP secret to password entry"""
    try:
        pm = get_pm_core(app)

        # Validate TOTP secret by generating a code
        try:
            TOTPManager.generate_totp_code(secret)
        except Exception as e:
            return Result.default_user_error(f"Invalid TOTP secret: {e}")

        updates = {
            'totp_secret': secret,
            'totp_issuer': issuer,
            'totp_account': account
        }

        return pm.update_password(entry_id, updates)

    except Exception as e:
        return Result.default_internal_error(f"Failed to add TOTP secret: {e}")


@export(mod_name=Name, api=True)
def parse_totp_qr_code(app: App, qr_data: str) -> Result:
    """Parse TOTP QR code data"""
    try:
        totp_data = TOTPManager.parse_totp_uri(qr_data)
        return Result.ok(data=totp_data)
    except Exception as e:
        return Result.default_internal_error(f"QR code parsing failed: {e}")


@export(mod_name=Name, api=True)
def get_password_for_autofill(app: App, url: str) -> Result:
    """Get password entry for browser autofill with improved matching."""
    try:
        pm = get_pm_core(app)
        domain = pm._extract_domain(url)

        if not domain:
            return Result.default_user_error("Invalid URL provided")

        potential_matches = []
        for entry_data in pm.password_db.get('all'):
            entry_domain = pm._extract_domain(entry_data['url'])
            if domain.endswith(entry_domain):
                # Berechne einen Score basierend auf der Übereinstimmungslänge
                score = len(entry_domain)
                potential_matches.append((score, entry_data))

        if not potential_matches:
            return Result.default_user_error("No matching password entries found")

        # Sortiere nach bestem Match (längste Domain-Übereinstimmung zuerst)
        potential_matches.sort(key=lambda x: x[0], reverse=True)

        # Nimm den besten Match
        best_match_data = potential_matches[0][1]

        # Bereite die finale Antwort vor
        autofill_data = {
            'id': best_match_data['id'],
            'url': best_match_data['url'],
            'username': best_match_data['username'],
            'password': best_match_data['password'],
            'title': best_match_data['title'],
            'totp_code': None,
            'time_remaining': None
        }

        # Generiere TOTP-Code, falls ein Geheimnis vorhanden ist
        if best_match_data.get('totp_secret'):
            try:
                secret = best_match_data['totp_secret']
                autofill_data['totp_code'] = TOTPManager.generate_totp_code(secret)
                autofill_data['time_remaining'] = 30 - (int(time.time()) % 30)
            except Exception as totp_error:
                app.logger.warning(f"Konnte TOTP für {best_match_data['id']} nicht generieren: {totp_error}")

        return Result.ok(data=autofill_data)

    except Exception as e:
        return Result.default_internal_error(f"Autofill lookup failed: {e}")
