"""
API: WebAuthn Passkey registration and authentication.
"""

import time
import base64
import json

from toolboxv2 import App, Result, get_app, get_logger
from toolboxv2.utils.system.types import ApiResult

from .config import get_passkey_config
from .models import Passkey
from .state import _store_challenge, _validate_and_consume_challenge
from .user_store import _load_user, _save_user, _find_user_by_provider
from .jwt_tokens import _generate_tokens

Name = "CloudM.Auth"
version = "2.0.0"
export = get_app(f"{Name}.Export").tb


@export(mod_name=Name, version=version, api=True)
async def passkey_register_start(
    app: App = None, user_id: str = None, username: str = None, data: dict = None, **kwargs
) -> ApiResult:
    """Start WebAuthn registration - generate challenge and options."""
    if app is None:
        app = get_app(f"{Name}.passkey_register_start")
    if data:
        user_id = user_id or data.get("user_id")
        username = username or data.get("username")
    if not user_id or not username:
        return Result.default_user_error("User ID and username required")

    try:
        from webauthn import generate_registration_options, options_to_json
        from webauthn.helpers.structs import (
            PublicKeyCredentialDescriptor,
            AuthenticatorSelectionCriteria,
            UserVerificationRequirement,
            ResidentKeyRequirement,
        )
    except ImportError:
        return Result.default_internal_error("py_webauthn not installed")

    config = get_passkey_config()

    # Get existing credentials to exclude
    user = await _load_user(app, user_id)
    exclude_credentials = []
    if user:
        for pk in user.passkeys:
            exclude_credentials.append(PublicKeyCredentialDescriptor(
                id=base64.urlsafe_b64decode(pk["credential_id"] + "=="),
            ))

    options = generate_registration_options(
        rp_id=config["rp_id"],
        rp_name=config["rp_name"],
        user_id=user_id.encode(),
        user_name=username,
        user_display_name=username,
        exclude_credentials=exclude_credentials,
        authenticator_selection=AuthenticatorSelectionCriteria(
            user_verification=UserVerificationRequirement.PREFERRED,
            resident_key=ResidentKeyRequirement.PREFERRED,
        ),
    )

    # Store challenge in DB
    challenge_b64 = base64.urlsafe_b64encode(options.challenge).decode().rstrip("=")
    await _store_challenge(app, challenge_b64, {
        "user_id": user_id,
        "username": username,
        "type": "registration",
        "challenge_bytes": base64.b64encode(options.challenge).decode(),
    })

    return Result.ok(json.loads(options_to_json(options)))


@export(mod_name=Name, version=version, api=True)
async def passkey_register_finish(
    app: App = None, challenge: str = None, credential: dict = None, data: dict = None, username: str = None, displayName: str = None, **kwargs
) -> ApiResult:
    """Complete WebAuthn registration - verify attestation and store credential."""
    if app is None:
        app = get_app(f"{Name}.passkey_register_finish")
    if data:
        challenge = challenge or data.get("challenge")
        credential = credential or data.get("credential")
    if not challenge or not credential:
        return Result.default_user_error("Challenge and credential required")

    challenge_data = await _validate_and_consume_challenge(app, challenge, "registration")
    if not challenge_data:
        return Result.default_user_error("Invalid or expired challenge")

    user_id = challenge_data["user_id"]
    config = get_passkey_config()

    try:
        from webauthn import verify_registration_response
        from webauthn.helpers.structs import RegistrationCredential

        credential_obj = RegistrationCredential.model_validate(credential)
        verification = verify_registration_response(
            credential=credential_obj,
            expected_challenge=base64.b64decode(challenge_data["challenge_bytes"]),
            expected_rp_id=config["rp_id"],
            expected_origin=config["origin"],
        )
    except ImportError:
        return Result.default_internal_error("py_webauthn not installed")
    except Exception as e:
        get_logger().error(f"[{Name}] WebAuthn registration verification failed: {e}")
        return Result.default_user_error(f"Registration verification failed: {e}")

    user = await _load_user(app, user_id)
    if not user:
        return Result.default_user_error("User not found")

    cred_id_b64 = base64.urlsafe_b64encode(verification.credential_id).decode().rstrip("=")
    pub_key_b64 = base64.b64encode(verification.credential_public_key).decode()

    passkey = Passkey(
        credential_id=cred_id_b64,
        public_key=pub_key_b64,
        sign_count=verification.sign_count,
        name=credential.get("name", "Passkey"),
    )
    user.passkeys.append(passkey.to_dict())
    await _save_user(app, user)

    try:
        app.audit_logger.log_action(
            user_id=user_id, action="auth.passkey.register",
            resource="/auth/passkey/register/finish", status="SUCCESS",
            details={"credential_id": cred_id_b64}
        )
    except Exception: pass

    return Result.ok({"success": True, "credential_id": cred_id_b64})


@export(mod_name=Name, version=version, api=True)
async def passkey_login_start(app: App = None, data: dict = None, username: str = None, **kwargs) -> ApiResult:
    """Start WebAuthn authentication - generate challenge."""
    if app is None:
        app = get_app(f"{Name}.passkey_login_start")

    try:
        from webauthn import generate_authentication_options, options_to_json
    except ImportError:
        return Result.default_internal_error("py_webauthn not installed")

    config = get_passkey_config()
    options = generate_authentication_options(rp_id=config["rp_id"])

    challenge_b64 = base64.urlsafe_b64encode(options.challenge).decode().rstrip("=")
    await _store_challenge(app, challenge_b64, {
        "type": "authentication",
        "challenge_bytes": base64.b64encode(options.challenge).decode(),
    })

    return Result.ok(json.loads(options_to_json(options)))


@export(mod_name=Name, version=version, api=True)
async def passkey_login_finish(
    app: App = None, challenge: str = None, credential: dict = None, data: dict = None, username: str = None, **kwargs
) -> ApiResult:
    """Complete WebAuthn authentication - verify assertion."""
    if app is None:
        app = get_app(f"{Name}.passkey_login_finish")
    log = get_logger()
    if data:
        challenge = challenge or data.get("challenge")
        credential = credential or data.get("credential")
    if not challenge or not credential:
        return Result.default_user_error("Challenge and credential required")

    challenge_data = await _validate_and_consume_challenge(app, challenge, "authentication")
    if not challenge_data:
        try:
            app.audit_logger.log_action(
                user_id="unknown", action="auth.login.passkey",
                resource="/auth/passkey/login/finish", status="FAILURE",
                details={"reason": "invalid_challenge"}
            )
        except Exception: pass
        return Result.default_user_error("Invalid or expired challenge")

    # Find user by credential ID
    cred_id = credential.get("id", "")
    user = await _find_user_by_provider(app, "passkey", cred_id)
    if not user:
        return Result.default_user_error("Passkey not registered")

    # Find the matching stored passkey
    stored_pk = None
    for pk in user.passkeys:
        if pk.get("credential_id") == cred_id:
            stored_pk = pk
            break
    if not stored_pk:
        return Result.default_user_error("Passkey credential not found")

    config = get_passkey_config()

    try:
        from webauthn import verify_authentication_response
        from webauthn.helpers.structs import AuthenticationCredential

        credential_obj = AuthenticationCredential.model_validate(credential)
        verification = verify_authentication_response(
            credential=credential_obj,
            expected_challenge=base64.b64decode(challenge_data["challenge_bytes"]),
            expected_rp_id=config["rp_id"],
            expected_origin=config["origin"],
            credential_public_key=base64.b64decode(stored_pk["public_key"]),
            credential_current_sign_count=stored_pk.get("sign_count", 0),
        )
    except ImportError:
        return Result.default_internal_error("py_webauthn not installed")
    except Exception as e:
        log.error(f"[{Name}] WebAuthn auth verification failed: {e}")
        try:
            app.audit_logger.log_action(
                user_id="unknown", action="auth.login.passkey",
                resource="/auth/passkey/login/finish", status="FAILURE",
                details={"reason": "verification_failed"}
            )
        except Exception: pass
        return Result.default_user_error(f"Authentication verification failed: {e}")

    # Update sign count
    stored_pk["sign_count"] = verification.new_sign_count
    user.last_login = time.time()
    await _save_user(app, user)

    jwt_tokens = _generate_tokens(user, "passkey")
    log.info(f"[{Name}] Passkey login: {user.user_id}")

    try:
        app.audit_logger.log_action(
            user_id=user.user_id, action="auth.login.passkey",
            resource="/auth/passkey/login/finish", status="SUCCESS",
            details={"credential_id": cred_id}
        )
    except Exception: pass

    return Result.ok({
        "authenticated": True,
        "user_id": user.user_id,
        "username": user.username,
        "email": user.email,
        "level": user.level,
        "provider": "passkey",
        **jwt_tokens,
    })
