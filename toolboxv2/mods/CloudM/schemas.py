"""
CloudM Authentication API Schemas - Request/Response DTOs
Version: 2.0.0

Diese Datei enthält alle Request/Response Models für die Auth-API.
Eliminiert dict-Zugriffe und sorgt für saubere Typisierung.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, EmailStr


# =================== Registration Schemas ===================

class RegistrationStartRequest(BaseModel):
    """Request to start WebAuthn registration"""
    username: str = Field(..., min_length=3, max_length=50, description="Desired username")
    email: EmailStr = Field(..., description="User email address")
    invite_code: Optional[str] = Field(default=None, description="Invitation code (if required)")
    device_label: Optional[str] = Field(default="My Device", description="Friendly device name")


class RegistrationStartResponse(BaseModel):
    """Response with WebAuthn registration options"""
    options: Dict[str, Any] = Field(..., description="WebAuthn PublicKeyCredentialCreationOptions")
    session_id: str = Field(..., description="Session ID for this registration flow")


class RegistrationFinishRequest(BaseModel):
    """Request to complete WebAuthn registration"""
    username: str = Field(..., description="Username")
    session_id: str = Field(..., description="Session ID from start phase")
    credential: Dict[str, Any] = Field(..., description="WebAuthn credential response from navigator.credentials.create()")
    device_label: Optional[str] = Field(default="My Device", description="Friendly device name")


class RegistrationFinishResponse(BaseModel):
    """Response after successful registration"""
    success: bool = Field(..., description="Registration success status")
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    user: Dict[str, Any] = Field(..., description="User data (uid, username, email, level)")


# =================== Authentication Schemas ===================

class AuthStartRequest(BaseModel):
    """Request to start WebAuthn authentication"""
    username: str = Field(..., description="Username")


class AuthStartResponse(BaseModel):
    """Response with WebAuthn authentication options"""
    options: Dict[str, Any] = Field(..., description="WebAuthn PublicKeyCredentialRequestOptions")
    session_id: str = Field(..., description="Session ID for this auth flow")


class AuthFinishRequest(BaseModel):
    """Request to complete WebAuthn authentication"""
    username: str = Field(..., description="Username")
    session_id: str = Field(..., description="Session ID from start phase")
    credential: Dict[str, Any] = Field(..., description="WebAuthn assertion response from navigator.credentials.get()")


class AuthFinishResponse(BaseModel):
    """Response after successful authentication"""
    success: bool = Field(..., description="Authentication success status")
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    user: Dict[str, Any] = Field(..., description="User data (uid, username, email, level)")


# =================== Token Refresh Schemas ===================

class TokenRefreshRequest(BaseModel):
    """Request to refresh access token"""
    refresh_token: str = Field(..., description="Valid refresh token")


class TokenRefreshResponse(BaseModel):
    """Response with new access token"""
    success: bool = Field(..., description="Refresh success status")
    access_token: str = Field(..., description="New JWT access token")
    refresh_token: Optional[str] = Field(default=None, description="New refresh token (if rotated)")


# =================== Device Management Schemas ===================

class DeviceInviteRequest(BaseModel):
    """Request to create device invitation link"""
    device_label: Optional[str] = Field(default="New Device", description="Label for the new device")
    ttl_minutes: int = Field(default=15, ge=5, le=60, description="Invitation validity in minutes")


class DeviceInviteResponse(BaseModel):
    """Response with magic link"""
    success: bool = Field(..., description="Invite creation success")
    invite_url: str = Field(..., description="Magic link URL")
    invite_token: str = Field(..., description="Invitation token")
    expires_at: str = Field(..., description="Expiration timestamp (ISO format)")


class DeviceListResponse(BaseModel):
    """Response with list of user's devices"""
    devices: List[Dict[str, Any]] = Field(..., description="List of registered devices")


class DeviceRemoveRequest(BaseModel):
    """Request to remove a device"""
    credential_id: str = Field(..., description="Credential ID to remove")


# =================== CLI Login Schemas ===================

class CLILoginStartRequest(BaseModel):
    """Request to start CLI login flow"""
    session_id: str = Field(..., description="CLI-generated session UUID")


class CLILoginStartResponse(BaseModel):
    """Response with approval URL"""
    approval_url: str = Field(..., description="URL for user to approve in browser")
    session_id: str = Field(..., description="Session ID for polling")


class CLILoginStatusRequest(BaseModel):
    """Request to check CLI login status"""
    session_id: str = Field(..., description="CLI session ID")


class CLILoginStatusResponse(BaseModel):
    """Response with login status"""
    status: str = Field(..., description="Status: pending/approved/expired")
    access_token: Optional[str] = Field(default=None, description="JWT token if approved")
    refresh_token: Optional[str] = Field(default=None, description="Refresh token if approved")


class CLILoginApproveRequest(BaseModel):
    """Request to approve CLI login (from browser)"""
    session_id: str = Field(..., description="CLI session ID to approve")
    device_label: Optional[str] = Field(default="CLI", description="Device label")


# =================== Magic Link Schemas ===================

class MagicLinkRequest(BaseModel):
    """Request to send magic link email"""
    username: str = Field(..., description="Username")
    email: EmailStr = Field(..., description="User email")


class MagicLinkResponse(BaseModel):
    """Response after magic link sent"""
    success: bool = Field(..., description="Email sent status")
    message: str = Field(..., description="Status message")
    invite_url: Optional[str] = Field(default="", description="Magic link URL (empty if sent via email)")
    invite_token: Optional[str] = Field(default="", description="Magic link token (empty if sent via email)")
    expires_at: Optional[str] = Field(default="", description="Expiration timestamp (ISO format)")


class MagicLinkConsumeRequest(BaseModel):
    """Request to consume magic link token"""
    token: str = Field(..., description="Magic link token from URL")


# =================== Error Response Schema ===================

class ErrorResponse(BaseModel):
    """Standard error response"""
    success: bool = Field(default=False, description="Always false for errors")
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")

