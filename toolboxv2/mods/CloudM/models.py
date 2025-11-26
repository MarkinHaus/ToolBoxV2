"""
CloudM Authentication Models - Pydantic Models for Type Safety
Version: 2.0.0

Diese Datei enthält die neuen Pydantic Models für das modernisierte Auth-System.
Ersetzt die alten dataclass-basierten User-Modelle mit sauberer Typisierung.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, EmailStr, field_validator
import uuid


# =================== WebAuthn Credential Models ===================

class WebAuthnCredential(BaseModel):
    """
    Speichert ein WebAuthn/Passkey Credential.
    Basiert auf FIDO2 Standard.
    """
    credential_id: str = Field(..., description="Base64-encoded credential ID")
    public_key: bytes = Field(..., description="COSE-encoded public key")
    sign_count: int = Field(default=0, description="Signature counter (anti-cloning)")
    transports: List[str] = Field(default_factory=list, description="Authenticator transports (usb, nfc, ble, internal)")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    last_used: Optional[datetime] = Field(default=None, description="Last usage timestamp")
    label: str = Field(default="Unnamed Device", description="User-friendly device name")
    aaguid: Optional[str] = Field(default=None, description="Authenticator AAGUID")
    
    class Config:
        json_encoders = {
            bytes: lambda v: v.hex() if v else None,
            datetime: lambda v: v.isoformat() if v else None
        }


# =================== User Models ===================

class UserBase(BaseModel):
    """Base User Model - Nur WebAuthn, keine Custom Crypto"""
    uid: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique user ID")
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: EmailStr = Field(..., description="User email address")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Account creation time")
    level: int = Field(default=0, ge=0, le=100, description="User permission level")
    log_level: str = Field(default="INFO", description="Logging level")
    settings: Dict[str, Any] = Field(default_factory=dict, description="User settings")
    
    @field_validator('username')
    @classmethod
    def validate_username(cls, v: str) -> str:
        """Validate username format"""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username must be alphanumeric (with _ or - allowed)')
        return v.lower()


class User(UserBase):
    """
    Vollständiges User Model mit WebAuthn Credentials.
    KEINE user_pass_pri, user_pass_pub, user_pass_sync mehr!
    """
    credentials: List[WebAuthnCredential] = Field(default_factory=list, description="WebAuthn credentials")
    is_active: bool = Field(default=True, description="Account active status")
    last_login: Optional[datetime] = Field(default=None, description="Last login timestamp")
    
    def get_credential_by_id(self, credential_id: str) -> Optional[WebAuthnCredential]:
        """Find credential by ID"""
        for cred in self.credentials:
            if cred.credential_id == credential_id:
                return cred
        return None
    
    def update_credential_sign_count(self, credential_id: str, new_count: int) -> bool:
        """Update sign count for credential (anti-cloning protection)"""
        cred = self.get_credential_by_id(credential_id)
        if cred:
            cred.sign_count = new_count
            cred.last_used = datetime.utcnow()
            return True
        return False
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


# =================== Session Token Models ===================

class TokenType:
    """Token type constants"""
    ACCESS = "access"           # Short-lived API access token (15 min)
    REFRESH = "refresh"         # Long-lived refresh token (7 days)
    DEVICE_INVITE = "device_invite"  # Magic link for device registration (15 min)
    CLI_SESSION = "cli_session"      # CLI login session token (1 hour)


class SessionToken(BaseModel):
    """JWT Token Claims"""
    sub: str = Field(..., description="Subject (username)")
    uid: str = Field(..., description="User ID")
    type: str = Field(..., description="Token type (access/refresh/device_invite/cli_session)")
    exp: int = Field(..., description="Expiration timestamp (Unix)")
    iat: int = Field(..., description="Issued at timestamp (Unix)")
    jti: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), description="JWT ID (unique token ID)")
    device_label: Optional[str] = Field(default=None, description="Device label for tracking")
    
    @field_validator('type')
    @classmethod
    def validate_token_type(cls, v: str) -> str:
        """Validate token type"""
        valid_types = [TokenType.ACCESS, TokenType.REFRESH, TokenType.DEVICE_INVITE, TokenType.CLI_SESSION]
        if v not in valid_types:
            raise ValueError(f'Token type must be one of: {valid_types}')
        return v


# =================== Challenge Models ===================

class ChallengeData(BaseModel):
    """Temporary challenge data for WebAuthn flows"""
    challenge: str = Field(..., description="Base64URL-encoded challenge")
    username: str = Field(..., description="Username")
    type: str = Field(..., description="Challenge type (register/login)")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Challenge creation time")
    expires_at: datetime = Field(..., description="Challenge expiration time")
    
    @field_validator('type')
    @classmethod
    def validate_challenge_type(cls, v: str) -> str:
        """Validate challenge type"""
        if v not in ['register', 'login']:
            raise ValueError('Challenge type must be "register" or "login"')
        return v
    
    def is_expired(self) -> bool:
        """Check if challenge is expired"""
        return datetime.utcnow() > self.expires_at


# =================== Migration Helper ===================

class LegacyUser(BaseModel):
    """
    Legacy User Model für Migration.
    Enthält alte Felder für Kompatibilität.
    """
    uid: str
    name: str
    email: str
    pub_key: str = ""
    user_pass_pub: str = ""
    user_pass_pri: str = ""
    user_pass_sync: str = ""
    user_pass_pub_devices: List[str] = Field(default_factory=list)
    user_pass_pub_persona: Dict[str, Any] = Field(default_factory=dict)
    challenge: str = ""
    is_persona: bool = False
    level: int = 0
    creation_time: str = ""
    log_level: str = "INFO"
    settings: Dict[str, Any] = Field(default_factory=dict)

