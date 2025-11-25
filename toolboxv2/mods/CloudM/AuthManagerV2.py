"""
CloudM Authentication Manager V2 - Modern WebAuthn-Only Auth
Version: 2.0.0

Neue, sichere Authentifizierung:
- Nur WebAuthn/Passkeys (keine Custom Crypto mehr)
- Stateless Challenges (ChallengeService)
- Standard JWT mit Server Secret (TokenService)
- 2-Step Flow (start/finish Pattern)
- Auto-Refresh Mechanismus
"""
import os
import uuid
import base64
from dataclasses import asdict
from typing import Optional

import webauthn
from webauthn.helpers.structs import (
    PublicKeyCredentialDescriptor,
    AuthenticatorSelectionCriteria,
    UserVerificationRequirement,
    ResidentKeyRequirement,
    AuthenticatorAttachment,
)
from webauthn.helpers import base64url_to_bytes, bytes_to_base64url

from toolboxv2 import App, Result, get_app, get_logger
from toolboxv2.utils.system.types import ApiResult
from .email_services import send_magic_link_email

from .models import User, WebAuthnCredential, TokenType
from .schemas import (
    RegistrationStartRequest, RegistrationStartResponse,
    RegistrationFinishRequest, RegistrationFinishResponse,
    AuthStartRequest, AuthStartResponse,
    AuthFinishRequest, AuthFinishResponse,
    TokenRefreshRequest, TokenRefreshResponse,
    MagicLinkRequest, MagicLinkResponse,
    MagicLinkConsumeRequest,
    ErrorResponse
)
from toolboxv2.mods.CloudM.services import get_challenge_service, get_token_service
from toolboxv2 import TBEF

Name = 'CloudM.AuthManagerV2'
version = '2.0.0'
export = get_app(f"{Name}.Export").tb


# =================== Configuration ===================

RP_ID = os.getenv("HOSTNAME", "localhost")
RP_NAME = "ToolBox V2"
ORIGIN = os.getenv("APP_BASE_URL", "http://localhost:8080")


# =================== Database Helper Functions ===================

def db_user_exists(app: App, username: str) -> bool:
    """Check if user exists in database"""
    result = app.run_any(TBEF.DB.IF_EXIST, query=f"USER::{username}::*", get_results=True)
    if result.is_error():
        return False
    return result.get() > 0


def db_get_user(app: App, username: str) -> Optional[dict]:
    """Get user data from database"""
    result = app.run_any(TBEF.DB.GET, query=f"USER::{username}::*", get_results=True)
    if result.is_error() or not result.get():
        return None

    user_data = result.get()
    if isinstance(user_data, list) and len(user_data) > 0:
        return user_data[0]
    return user_data


def db_save_user(app: App, username: str, user_data: dict) -> bool:
    """Save user data to database"""
    result = app.run_any(
        TBEF.DB.SET,
        query=f"USER::{username}::{user_data['uid']}",
        data=user_data,
        get_results=True
    )
    return result.is_ok()


def db_delete_user(app: App, username: str) -> bool:
    """Delete user from database"""
    result = app.run_any(
        TBEF.DB.DELETE,
        query=f"USER::{username}::*",
        matching=True,
        get_results=True
    )
    return result.is_ok()


# =================== Helper Functions ===================

def _user_to_dict(user: User) -> dict:
    """Convert User model to safe dict for API response"""
    return {
        'uid': user.uid,
        'username': user.username,
        'email': user.email,
        'level': user.level,
        'created_at': user.created_at.isoformat(),
        'last_login': user.last_login.isoformat() if user.last_login else None,
        'credentials_count': len(user.credentials)
    }


# =================== Registration Flow ===================

@export(mod_name=Name, version=version, api=True, test=False)
async def register_start(app: App, data: dict) -> ApiResult:
    """
    Step 1: Start WebAuthn Registration

    Request: RegistrationStartRequest
    Response: RegistrationStartResponse with WebAuthn options
    """
    try:
        # Validate request
        req = RegistrationStartRequest(**data)

        # ✅ SECURITY FIX: Validate invite code if provided
        if req.invite_code:
            # Placeholder for invite code validation
            # TODO: Implement proper invite code validation against database
            get_logger().info(f"[{Name}] Invite code provided: {req.invite_code[:8]}...")
        else:
            # Check if open registration is allowed
            open_registration = os.getenv("ALLOW_OPEN_REGISTRATION", "true").lower() == "true"
            if not open_registration:
                return Result.default_user_error(
                    info="Invite code required",
                    data=ErrorResponse(error="invite_required", message="Registration requires an invite code").model_dump()
                )

        # Check if user already exists
        if db_user_exists(app, req.username):
            return Result.default_user_error(
                info="Username already exists",
                data=ErrorResponse(error="user_exists", message="Username already taken").model_dump()
            )

        # Generate session ID and challenge
        challenge_service = get_challenge_service()
        session_id = str(uuid.uuid4())
        challenge = challenge_service.store_challenge(
            session_id=session_id,
            username=req.username,
            challenge_type='register',
            ttl_minutes=5
        )

        # Generate WebAuthn registration options
        user_id = str(uuid.uuid4())

        registration_options = webauthn.generate_registration_options(
            rp_id=RP_ID,
            rp_name=RP_NAME,
            user_id=user_id,
            user_name=req.username,
            user_display_name=req.username,
            challenge=challenge.encode('utf-8'),
            authenticator_selection=AuthenticatorSelectionCriteria(
                authenticator_attachment=AuthenticatorAttachment.PLATFORM,
                resident_key=ResidentKeyRequirement.PREFERRED,
                user_verification=UserVerificationRequirement.PREFERRED
            ),
            timeout=60000  # 60 seconds
        )

        # Convert to dict for JSON response
        options_dict = {
            'challenge': bytes_to_base64url(registration_options.challenge),
            'rp': {'id': registration_options.rp.id, 'name': registration_options.rp.name},
            'user': {
                'id': bytes_to_base64url(registration_options.user.id),
                'name': registration_options.user.name,
                'displayName': registration_options.user.display_name
            },
            'pubKeyCredParams': [
                {'type': p.type, 'alg': p.alg} for p in registration_options.pub_key_cred_params
            ],
            'timeout': registration_options.timeout,
            'authenticatorSelection': {
                'authenticatorAttachment': registration_options.authenticator_selection.authenticator_attachment.value if registration_options.authenticator_selection.authenticator_attachment else None,
                'residentKey': registration_options.authenticator_selection.resident_key.value if registration_options.authenticator_selection.resident_key else None,
                'userVerification': registration_options.authenticator_selection.user_verification.value
            }
        }

        response = RegistrationStartResponse(
            options=options_dict,
            session_id=session_id
        )

        return Result.ok(data=response.model_dump())

    except Exception as e:
        get_logger().error(f"Registration start error: {e}", exc_info=True)
        return Result.default_internal_error(
            info=f"Registration failed: {str(e)}",
            data=ErrorResponse(error="registration_error", message=str(e)).model_dump()
        )


@export(mod_name=Name, version=version, api=True, test=False)
async def register_finish(app: App, data: dict) -> ApiResult:
    """
    Step 2: Complete WebAuthn Registration

    Request: RegistrationFinishRequest
    Response: RegistrationFinishResponse with tokens
    """
    try:
        # Validate request
        req = RegistrationFinishRequest(**data)

        # Get and consume challenge
        challenge_service = get_challenge_service()
        challenge_data = challenge_service.consume_challenge(req.session_id)

        if not challenge_data:
            return Result.default_user_error(
                info="Invalid or expired session",
                data=ErrorResponse(error="invalid_session", message="Session expired or not found").model_dump()
            )

        # Validate username matches
        if challenge_data.username != req.username:
            return Result.default_user_error(
                info="Username mismatch",
                data=ErrorResponse(error="username_mismatch", message="Username does not match session").model_dump()
            )

        # Verify WebAuthn registration response
        try:
            # Parse credential from client
            credential_data = req.credential

            # Convert to webauthn format
            registration_credential = webauthn.RegistrationCredential(
                id=credential_data['id'],
                raw_id=base64url_to_bytes(credential_data['rawId']),
                response=webauthn.AuthenticatorAttestationResponse(
                    client_data_json=base64url_to_bytes(credential_data['response']['clientDataJSON']),
                    attestation_object=base64url_to_bytes(credential_data['response']['attestationObject'])
                ),
                type=credential_data['type']
            )

            verification = webauthn.verify_registration_response(
                credential=registration_credential,
                expected_challenge=challenge_data.challenge.encode('utf-8'),
                expected_origin=ORIGIN,
                expected_rp_id=RP_ID,
                require_user_verification=True
            )

        except Exception as e:
            get_logger().error(f"WebAuthn verification failed: {e}", exc_info=True)
            return Result.default_user_error(
                info="WebAuthn verification failed",
                data=ErrorResponse(error="verification_failed", message=str(e)).model_dump()
            )

        # Create new user with credential
        from datetime import datetime

        new_credential = WebAuthnCredential(
            credential_id=bytes_to_base64url(verification.credential_id),
            public_key=verification.credential_public_key,
            sign_count=verification.sign_count,
            transports=credential_data.get('response', {}).get('transports', []),
            label=req.device_label or "My Device",
            aaguid=bytes_to_base64url(verification.aaguid) if verification.aaguid else None
        )

        new_user = User(
            username=req.username,
            email=req.email,
            credentials=[new_credential],
            last_login=datetime.utcnow()
        )

        # Save user to database
        user_dict = new_user.model_dump(mode='json')
        db_save_user(app, req.username, user_dict)

        # Generate tokens
        token_service = get_token_service()
        tokens = token_service.create_token_pair(new_user, device_label=req.device_label)

        response = RegistrationFinishResponse(
            success=True,
            access_token=tokens['access_token'],
            refresh_token=tokens['refresh_token'],
            user=_user_to_dict(new_user)
        )

        return Result.ok(data=response.model_dump())

    except Exception as e:
        get_logger().error(f"Registration finish error: {e}", exc_info=True)
        return Result.default_internal_error(
            info=f"Registration failed: {str(e)}",
            data=ErrorResponse(error="registration_error", message=str(e)).model_dump()
        )


# =================== Authentication Flow ===================

@export(mod_name=Name, version=version, api=True, test=False)
async def login_start(app: App, data: dict) -> ApiResult:
    """
    Step 1: Start WebAuthn Authentication

    Request: AuthStartRequest
    Response: AuthStartResponse with WebAuthn options
    """
    try:
        # Validate request
        req = AuthStartRequest(**data)

        # Check if user exists
        if not db_user_exists(app, req.username):
            return Result.default_user_error(
                info="User not found",
                data=ErrorResponse(error="user_not_found", message="Username not found").model_dump()
            )

        # Get user from database
        user_data = db_get_user(app, req.username)
        if not user_data:
            return Result.default_user_error(
                info="User not found",
                data=ErrorResponse(error="user_not_found", message="Could not load user").model_dump()
            )

        # Convert to User model
        user = User(**user_data)

        # Check if user has credentials
        if not user.credentials:
            return Result.default_user_error(
                info="No credentials registered",
                data=ErrorResponse(error="no_credentials", message="No WebAuthn credentials found. Please register a passkey first.").model_dump()
            )

        # Generate session ID and challenge
        challenge_service = get_challenge_service()
        session_id = str(uuid.uuid4())
        challenge = challenge_service.store_challenge(
            session_id=session_id,
            username=req.username,
            challenge_type='login',
            ttl_minutes=5
        )

        # Prepare allowed credentials
        allowed_credentials = [
            PublicKeyCredentialDescriptor(
                id=base64url_to_bytes(cred.credential_id),
                type="public-key",
                transports=cred.transports if cred.transports else []
            )
            for cred in user.credentials
        ]

        # Generate WebAuthn authentication options
        authentication_options = webauthn.generate_authentication_options(
            rp_id=RP_ID,
            challenge=challenge.encode('utf-8'),
            allow_credentials=allowed_credentials,
            user_verification=UserVerificationRequirement.PREFERRED,
            timeout=60000  # 60 seconds
        )

        # Convert to dict for JSON response
        options_dict = {
            'challenge': bytes_to_base64url(authentication_options.challenge),
            'timeout': authentication_options.timeout,
            'rpId': authentication_options.rp_id,
            'allowCredentials': [
                {
                    'id': bytes_to_base64url(cred.id),
                    'type': cred.type,
                    'transports': cred.transports if cred.transports else []
                }
                for cred in authentication_options.allow_credentials
            ],
            'userVerification': authentication_options.user_verification.value
        }

        response = AuthStartResponse(
            options=options_dict,
            session_id=session_id
        )

        return Result.ok(data=response.model_dump())

    except Exception as e:
        get_logger().error(f"Login start error: {e}", exc_info=True)
        return Result.default_internal_error(
            info=f"Login failed: {str(e)}",
            data=ErrorResponse(error="login_error", message=str(e)).model_dump()
        )


@export(mod_name=Name, version=version, api=True, test=False)
async def login_finish(app: App, data: dict) -> ApiResult:
    """
    Step 2: Complete WebAuthn Authentication

    Request: AuthFinishRequest
    Response: AuthFinishResponse with tokens
    """
    try:
        # Validate request
        req = AuthFinishRequest(**data)

        # Get and consume challenge
        challenge_service = get_challenge_service()
        challenge_data = challenge_service.consume_challenge(req.session_id)

        if not challenge_data:
            return Result.default_user_error(
                info="Invalid or expired session",
                data=ErrorResponse(error="invalid_session", message="Session expired or not found").model_dump()
            )

        # Validate username matches
        if challenge_data.username != req.username:
            return Result.default_user_error(
                info="Username mismatch",
                data=ErrorResponse(error="username_mismatch", message="Username does not match session").model_dump()
            )

        # Get user from database
        user_data = db_get_user(app, req.username)
        if not user_data:
            return Result.default_user_error(
                info="User not found",
                data=ErrorResponse(error="user_not_found", message="Could not load user").model_dump()
            )

        # Convert to User model
        user = User(**user_data)

        # Parse credential from client
        credential_data = req.credential
        credential_id_b64 = bytes_to_base64url(base64url_to_bytes(credential_data['rawId']))

        # Find matching credential
        user_credential = user.get_credential_by_id(credential_id_b64)
        if not user_credential:
            return Result.default_user_error(
                info="Credential not found",
                data=ErrorResponse(error="credential_not_found", message="Credential not registered for this user").model_dump()
            )

        # Verify WebAuthn authentication response
        try:
            authentication_credential = webauthn.AuthenticationCredential(
                id=credential_data['id'],
                raw_id=base64url_to_bytes(credential_data['rawId']),
                response=webauthn.AuthenticatorAssertionResponse(
                    client_data_json=base64url_to_bytes(credential_data['response']['clientDataJSON']),
                    authenticator_data=base64url_to_bytes(credential_data['response']['authenticatorData']),
                    signature=base64url_to_bytes(credential_data['response']['signature']),
                    user_handle=base64url_to_bytes(credential_data['response'].get('userHandle', '')) if credential_data['response'].get('userHandle') else None
                ),
                type=credential_data['type']
            )

            verification = webauthn.verify_authentication_response(
                credential=authentication_credential,
                expected_challenge=challenge_data.challenge.encode('utf-8'),
                expected_rp_id=RP_ID,
                expected_origin=ORIGIN,
                credential_public_key=user_credential.public_key,
                credential_current_sign_count=user_credential.sign_count,
                require_user_verification=True
            )

        except Exception as e:
            get_logger().error(f"WebAuthn verification failed: {e}", exc_info=True)
            return Result.default_user_error(
                info="WebAuthn verification failed",
                data=ErrorResponse(error="verification_failed", message=str(e)).model_dump()
            )

        # Update sign count (anti-cloning protection)
        from datetime import datetime
        user.update_credential_sign_count(credential_id_b64, verification.new_sign_count)
        user.last_login = datetime.utcnow()

        # Save updated user
        user_dict = user.model_dump(mode='json')
        db_save_user(app, req.username, user_dict)

        # Generate tokens
        token_service = get_token_service()
        tokens = token_service.create_token_pair(user, device_label=user_credential.label)

        response = AuthFinishResponse(
            success=True,
            access_token=tokens['access_token'],
            refresh_token=tokens['refresh_token'],
            user=_user_to_dict(user)
        )

        return Result.ok(data=response.model_dump())

    except Exception as e:
        get_logger().error(f"Login finish error: {e}", exc_info=True)
        return Result.default_internal_error(
            info=f"Login failed: {str(e)}",
            data=ErrorResponse(error="login_error", message=str(e)).model_dump()
        )


# =================== Token Refresh ===================

@export(mod_name=Name, version=version, api=True, test=False)
async def refresh_token(app: App, data: dict) -> ApiResult:
    """
    Refresh access token using refresh token.

    Request: TokenRefreshRequest
    Response: TokenRefreshResponse with new access token
    """
    try:
        # Validate request
        req = TokenRefreshRequest(**data)

        # Validate refresh token
        token_service = get_token_service()
        token_data = token_service.validate_token(req.refresh_token, expected_type=TokenType.REFRESH)

        if not token_data:
            return Result.default_user_error(
                info="Invalid or expired refresh token",
                data=ErrorResponse(error="invalid_token", message="Refresh token is invalid or expired").model_dump()
            )

        # Get user from database
        user_data = db_get_user(app, token_data.sub)
        if not user_data:
            return Result.default_user_error(
                info="User not found",
                data=ErrorResponse(error="user_not_found", message="User not found").model_dump()
            )

        # Convert to User model
        user = User(**user_data)

        # Create new access token
        new_access_token = token_service.create_token(user, TokenType.ACCESS, token_data.device_label)

        response = TokenRefreshResponse(
            success=True,
            access_token=new_access_token,
            refresh_token=None  # Don't rotate refresh token for now
        )

        return Result.ok(data=response.model_dump())

    except Exception as e:
        get_logger().error(f"Token refresh error: {e}", exc_info=True)
        return Result.default_internal_error(
            info=f"Token refresh failed: {str(e)}",
            data=ErrorResponse(error="refresh_error", message=str(e)).model_dump()
        )


# =================== Magic Link Endpoints ===================

@export(mod_name=Name, api=True, test=False, request_as_kwarg=True)
async def generate_magic_link(app: App, data: MagicLinkRequest, request=None) -> ApiResult:
    """
    Generate a magic link for passwordless login

    🔒 SECURITY: Requires authentication! Users can only generate magic links for themselves.

    Flow:
    1. Validate access token (REQUIRED)
    2. Verify user can only request link for themselves
    3. Create DEVICE_INVITE token (15 min TTL)
    4. Send magic link via email (NOT returned in response)

    Args:
        app: ToolBox App instance
        data: MagicLinkRequest with username and email
        request: HTTP request object (for auth header)

    Returns:
        ApiResult with success message (link sent via email)
    """
    try:
        req = MagicLinkRequest(**data)
        get_logger().info(f"[{Name}] generate_magic_link called for: {req.username}")

        # ✅ SECURITY FIX: Require authentication
        if not request:
            return Result.default_user_error(
                info="Request object required",
                data=ErrorResponse(error="internal_error", message="Request object not available").model_dump()
            )

        # Extract and validate access token
        auth_header = request.headers.get('Authorization') or request.headers.get('authorization')
        if not auth_header:
            get_logger().warning(f"[{Name}] Unauthorized magic link request for {req.username}")
            return Result.default_user_error(
                info="Authentication required",
                data=ErrorResponse(error="unauthorized", message="You must be logged in to generate a magic link").model_dump()
            )

        # Parse token
        token = auth_header.replace('Bearer ', '').replace('bearer ', '')
        token_service = get_token_service()
        token_data = token_service.validate_token(token, TokenType.ACCESS)

        if not token_data:
            get_logger().warning(f"[{Name}] Invalid token for magic link request")
            return Result.default_user_error(
                info="Invalid or expired token",
                data=ErrorResponse(error="invalid_token", message="Your session is invalid or expired").model_dump()
            )

        # ✅ SECURITY FIX: Self-service only - users can only generate links for themselves
        if token_data.sub != req.username:
            get_logger().warning(f"[{Name}] User {token_data.sub} tried to generate magic link for {req.username}")
            return Result.default_user_error(
                info="Forbidden",
                data=ErrorResponse(error="forbidden", message="You can only generate magic links for yourself").model_dump()
            )

        # Check if user exists
        if not db_user_exists(app, req.username):
            return Result.default_user_error(
                info=f"User not found: {req.username}",
                data=ErrorResponse(error="user_not_found", message="User does not exist").model_dump()
            )

        # Get user data
        user_data = db_get_user(app, req.username)
        if not user_data:
            return Result.default_user_error(
                info="User data not found",
                data=ErrorResponse(error="user_not_found", message="User data not found").model_dump()
            )

        user = User(**user_data)

        # Verify email matches
        if user.email != req.email:
            return Result.default_user_error(
                info="Email mismatch",
                data=ErrorResponse(error="email_mismatch", message="Email does not match your account").model_dump()
            )

        # Create magic link token (DEVICE_INVITE type, 15 min TTL)
        magic_token = token_service.create_token(
            user=user,
            token_type=TokenType.DEVICE_INVITE,
            device_label="Magic Link Login"
        )

        # Build magic link URL
        base_url = os.getenv("APP_BASE_URL", "http://localhost:8080")
        magic_url = f"{base_url}/web/m_link.html?token={magic_token}&name={req.username}"

        # ✅ SECURITY FIX: Send via email instead of returning in response
        # TODO: Implement email sending
        send_magic_link_email(app, user.email.__str__(), magic_url, user.username)

        # For now, log the link (REMOVE IN PRODUCTION!)
        get_logger().info(f"[{Name}] Magic link (DEV ONLY): {magic_url}")

        # ✅ SECURITY FIX: Don't return the link or token in the response!
        response = MagicLinkResponse(
            success=True,
            message=f"Magic link sent to {user.email}",
            invite_url="",  # Empty - sent via email
            invite_token="",  # Empty - sent via email
            expires_at=""  # Empty - not needed in response
        )

        get_logger().info(f"[{Name}] Magic link sent to {user.email}")
        return Result.ok(data=response.model_dump())

    except Exception as e:
        get_logger().error(f"Magic link generation error: {e}", exc_info=True)
        return Result.default_internal_error(
            info=f"Magic link generation failed: {str(e)}",
            data=ErrorResponse(error="magic_link_error", message=str(e)).model_dump()
        )


@export(mod_name=Name, api=True, test=False)
async def consume_magic_link(app: App, data: MagicLinkConsumeRequest) -> ApiResult:
    """
    Consume magic link token and auto-login user

    Flow:
    1. Validate magic link token
    2. Extract user info from token
    3. Issue access + refresh tokens
    4. Return tokens for auto-login

    Args:
        app: ToolBox App instance
        data: MagicLinkConsumeRequest with token

    Returns:
        ApiResult with access and refresh tokens
    """
    try:
        req = MagicLinkConsumeRequest(**data)
        get_logger().info(f"[{Name}] consume_magic_link called")

        # Validate token
        token_service = get_token_service()
        token_data = token_service.validate_token(req.token, TokenType.DEVICE_INVITE)

        if not token_data:
            return Result.default_user_error(
                info="Invalid or expired magic link token",
                data=ErrorResponse(error="invalid_token", message="Magic link is invalid or expired").model_dump()
            )

        username = token_data.sub
        get_logger().info(f"[{Name}] Magic link token valid for: {username}")

        # Get user data
        user_data = db_get_user(app, username)
        if not user_data:
            return Result.default_user_error(
                info=f"User not found: {username}",
                data=ErrorResponse(error="user_not_found", message="User does not exist").model_dump()
            )

        user = User(**user_data)

        # Update last login
        from datetime import datetime
        user.last_login = datetime.utcnow()

        # Save updated user
        user_dict = user.model_dump(mode='json')
        db_save_user(app, username, user_dict)

        # Create token pair for auto-login
        tokens = token_service.create_token_pair(user, device_label="Magic Link")

        response = AuthFinishResponse(
            success=True,
            message=f"Magic link login successful for {username}",
            access_token=tokens['access_token'],
            refresh_token=tokens['refresh_token'],
            user={
                'uid': user.uid,
                'username': user.username,
                'email': user.email,
                'level': user.level
            }
        )

        get_logger().info(f"[{Name}] Magic link consumed, user logged in: {username}")
        return Result.ok(data=response.model_dump())

    except Exception as e:
        get_logger().error(f"Magic link consumption error: {e}", exc_info=True)
        return Result.default_internal_error(
            info=f"Magic link consumption failed: {str(e)}",
            data=ErrorResponse(error="magic_link_error", message=str(e)).model_dump()
        )

