import asyncio
import base64
import datetime
import json
import os
import time
import uuid
from dataclasses import asdict
from typing import Any
from urllib.parse import quote, urlparse

import jwt
import webauthn
from pydantic import BaseModel, field_validator
from webauthn.helpers.exceptions import (
    InvalidAuthenticationResponse,
    InvalidRegistrationResponse,
)
from webauthn.helpers.structs import (
    AuthenticationCredential as WebAuthnAuthenticationCredential,  # Rename to avoid clash if needed
)
from webauthn.helpers.structs import (
    AuthenticatorAssertionResponse as WebAuthnAuthenticatorAssertionResponse,
)
from webauthn.helpers.structs import (
    AuthenticatorAttestationResponse,
    RegistrationCredential,
)

from toolboxv2 import TBEF, App, Result, get_app, get_logger
from toolboxv2.mods.DB.types import DatabaseModes
from toolboxv2.utils.security.cryp import Code
from toolboxv2.utils.system.types import ApiResult, ToolBoxInterfaces

from .email_services import send_magic_link_email
from .types import User, UserCreator

version = "0.0.2"
Name = 'CloudM.AuthManager'
export = get_app(f"{Name}.Export").tb
default_export = export(mod_name=Name, test=False)
test_only = export(mod_name=Name, test_only=True)
instance_bios = str(uuid.uuid4())


def b64decode(s: str) -> bytes: # Used for URL-safe base64
    padding = '=' * (-len(s) % 4)
    return base64.urlsafe_b64decode(s.encode() + padding.encode())
# Helper for standard base64 to bytes, as used by client for response fields
def base64_std_to_bytes(val: str | None) -> bytes | None:
    if val is None:
        return None
    if not isinstance(val, str):
        get_logger().warning(f"base64_std_to_bytes expected string, got {type(val)}")
        # Depending on strictness, either raise an error or return val (which might cause issues later)
        raise ValueError(f"Expected base64 string, got {type(val)}")
    try:
        padding = '=' * (-len(val) % 4)
        return base64.b64decode(val + padding)
    except Exception as e:
        get_logger().error(f"Error decoding standard base64 string '{val[:30]}...': {e}")
        raise # Re-raise to be caught by API handler

class CustomAuthenticationCredential(WebAuthnAuthenticationCredential): # Inherits from webauthn's struct
    @field_validator("raw_id", mode="before")
    @classmethod
    def decode_base64url_to_bytes(cls, v):
        if isinstance(v, str):
            return b64decode(v) # b64decode handles padding and uses urlsafe_b64decode
        return v


class CustomRegistrationCredential(RegistrationCredential):
    @field_validator('raw_id', mode="before") # Changed from after to before
    def convert_raw_id(cls, v: str):
        if isinstance(v, str): # Assuming raw_id comes as standard base64 from client for registration
             return base64_std_to_bytes(v)
        return v # Or raise error if not string

    @field_validator('response', mode="before")
    def convert_response(cls, data: dict):
        if isinstance(data, dict):
            # Assuming client sends these as standard base64 for registration
            return {
                k: base64_std_to_bytes(v) if isinstance(v, str) else v
                for k, v in data.items()
            }
        return data


# app Helper functions interaction with the db

def db_helper_test_exist(app: App, username: str):
    c = app.run_any(TBEF.DB.IF_EXIST, query=f"USER::{username}::*", get_results=True)
    if c.is_error(): return False
    b = c.get() > 0
    get_logger().info(f"TEST IF USER EXIST : {username} {b}")
    return b


def db_delete_invitation(app: App, invitation: str):
    return app.run_any(TBEF.DB.DELETE, query=f"invitation::{invitation}", get_results=True)


def db_valid_invitation(app: App, invitation: str):
    inv_key = app.run_any(TBEF.DB.GET, query=f"invitation::{invitation}", get_results=False)
    if inv_key is None:
        return False
    inv_key = inv_key[0]
    if isinstance(inv_key, bytes):
        inv_key = inv_key.decode()
    return Code.decrypt_symmetric(inv_key, invitation) == invitation


def db_crate_invitation(app: App):
    invitation = Code.generate_symmetric_key()
    inv_key = Code.encrypt_symmetric(invitation, invitation)
    app.run_any(TBEF.DB.SET, query=f"invitation::{invitation}", data=inv_key, get_results=True)
    return invitation


def db_helper_save_user(app: App, user_data: dict) -> Result:
    # db_helper_delete_user(app, user_data['name'], user_data['uid'], matching=True)
    return app.run_any(TBEF.DB.SET, query=f"USER::{user_data['name']}::{user_data['uid']}",
                       data=user_data,
                       get_results=True)


def db_helper_get_user(app: App, username: str, uid: str = '*'):
    return app.run_any(TBEF.DB.GET, query=f"USER::{username}::{uid}",
                       get_results=True)


def db_helper_delete_user(app: App, username: str, uid: str, matching=False):
    return app.run_any(TBEF.DB.DELETE, query=f"USER::{username}::{uid}", matching=matching,
                       get_results=True)


@export(mod_name=Name, state=True, test=False, interface=ToolBoxInterfaces.native)
def delete_user(app: App, username: str):
    """Deletes a user and all their data."""
    if not db_helper_test_exist(app, username):
        return Result.default_user_error(f"User '{username}' not found.")

    # This will delete all entries matching the user
    result = db_helper_delete_user(app, username, '*', matching=True)

    if result.is_ok():
        # Also remove the local private key file if it exists
        app.config_fh.remove_key_file_handler("Pk" + Code.one_way_hash(username, "dvp-k")[:8])
        return Result.ok(f"User '{username}' deleted successfully.")
    else:
        return Result.default_internal_error(f"Failed to delete user '{username}'.", data=result)


@export(mod_name=Name, state=True, test=False, interface=ToolBoxInterfaces.native)
def list_users(app: App):
    """Lists all registered users."""
    keys_result = app.run_any(TBEF.DB.GET, query="all-k", get_results=True)
    if keys_result.is_error():
        return keys_result

    user_keys = keys_result.get()
    if not user_keys:
        return Result.ok("No users found.")

    users = []
    for key in user_keys:
        if isinstance(key, bytes):
            key = key.decode()
        if not key.startswith("USER::"):
            continue
        # Extract username from the key USER::username::uid
        parts = key.split('::')
        if len(parts) > 1 and parts[1] not in [u['username'] for u in users]:
            user_res = get_user_by_name(app, parts[1])
            if user_res.is_ok():
                user_data = user_res.get()
                users.append({"username": user_data.name, "email": user_data.email, "level": user_data.level})

    return Result.ok(data=users)


# jwt helpers


def add_exp(massage: dict, hr_ex=2):
    massage['exp'] = datetime.datetime.now(tz=datetime.UTC) + datetime.timedelta(hours=hr_ex)
    return massage


def crate_jwt(data: dict, private_key: str, sync=False):
    data = add_exp(data)
    algorithm = 'RS256'
    if sync:
        algorithm = 'HS512'
    token = jwt.encode(data, private_key, algorithm=algorithm)
    return token


def validate_jwt(jwt_key: str, public_key: str) -> dict or str:
    if not jwt_key:
        return "No JWT Key provided"

    try:
        token = jwt.decode(jwt_key,
                           public_key,
                           leeway=datetime.timedelta(seconds=10),
                           algorithms=["RS256", "HS512"],
                           # audience=aud,
                           do_time_check=True,
                           verify=True)
        return token
    except jwt.exceptions.InvalidSignatureError:
        return "InvalidSignatureError"
    except jwt.exceptions.ExpiredSignatureError:
        return "ExpiredSignatureError"
    except jwt.exceptions.InvalidAudienceError:
        return "InvalidAudienceError"
    except jwt.exceptions.MissingRequiredClaimError:
        return "MissingRequiredClaimError"
    except Exception as e:
        return str(e)


def reade_jwt(jwt_key: str) -> dict or str:
    if not jwt_key:
        return "No JWT Key provided"

    try:
        token = jwt.decode(jwt_key,
                           leeway=datetime.timedelta(seconds=10),
                           algorithms=["RS256", "HS512"],
                           verify=False)
        return token
    except jwt.exceptions.InvalidSignatureError:
        return "InvalidSignatureError"
    except jwt.exceptions.ExpiredSignatureError:
        return "ExpiredSignatureError"
    except jwt.exceptions.InvalidAudienceError:
        return "InvalidAudienceError"
    except jwt.exceptions.MissingRequiredClaimError:
        return "MissingRequiredClaimError"
    except Exception as e:
        return str(e)


# Export functions

@export(mod_name=Name, state=True, test=False, interface=ToolBoxInterfaces.future)
def get_user_by_name(app: App, username: str, uid: str = '*') -> Result:

    if app is None:
        app = get_app(Name + '.get_user_by_name')

    if not db_helper_test_exist(app, username):
        return Result.default_user_error(info=f"get_user_by_name failed username '{username}' not registered")

    user_data = db_helper_get_user(app, username, uid)
    if isinstance(user_data, str) or user_data.is_error():
        return Result.default_internal_error(info="get_user_by_name failed no User data found is_error")

    user_data = user_data.get()

    if isinstance(user_data, bytes):
        return Result.ok(data=User(**eval(user_data.decode())))
    if isinstance(user_data, str):
        return Result.ok(data=User(**eval(user_data)))
    if isinstance(user_data, dict):
        return Result.ok(data=User(**user_data))
    elif isinstance(user_data, list):
        if len(user_data) == 0:
            return Result.default_internal_error(info="get_user_by_name failed no User data found", exec_code=9283)

        if len(user_data) > 1:
            pass

        if isinstance(user_data[0], bytes):
            user_data[0] = user_data[0].decode()

        return Result.ok(data=User(**eval(user_data[0])))
    else:
        return Result.default_internal_error(info="get_user_by_name failed no User data found", exec_code=2351)


def to_base64(data: str):
    return base64.b64encode(data.encode('utf-8'))


def from_base64(encoded_data: str):
    return base64.b64decode(encoded_data)


def initialize_and_return(app: App, user) -> ApiResult:
    if isinstance(user, User):
        user = UserCreator(**asdict(user))
    db_helper = db_helper_save_user(app, asdict(user))
    return db_helper.lazy_return('intern', data={
        "challenge": user.challenge,
        "userId": to_base64(user.uid),
        "username": user.name,
        "dSync": Code().encrypt_asymmetric(user.user_pass_sync, user.pub_key)})


class CreateUserObject(BaseModel):
    name: str
    email: str
    pub_key: str
    invitation: str
    web_data: bool = True
    as_base64: bool = True


class AddUserDeviceObject(BaseModel):
    name: str
    pub_key: str
    invitation: str
    web_data: bool = True
    as_base64: bool = True


@export(mod_name=Name, state=True, interface=ToolBoxInterfaces.api, api=False, test=False)
def get_new_user_invitation_key(username):
    return Code.one_way_hash(username, "00#", os.getenv("TB_R_KEY", "pepper123"))[:12] + str(uuid.uuid4())[:6]

@export(mod_name=Name, state=True, interface=ToolBoxInterfaces.api, api=True, test=False)
def create_user(app: App, data: CreateUserObject = None, username: str = 'test-user',
                      email: str = 'test@user.com',
                      pub_key: str = '',
                      invitation: str = '', web_data=False, as_base64=False) -> ApiResult:
    if isinstance(data, dict):
        data = CreateUserObject(**data)
    username = data.name if data is not None else username
    email = data.email if data is not None else email
    pub_key = data.pub_key if data is not None else pub_key
    invitation = data.invitation if data is not None else invitation
    web_data = data.web_data if data is not None else web_data
    as_base64 = data.as_base64 if data is not None else as_base64

    if app is None:
        app = get_app(Name + '.crate_user')

    if db_helper_test_exist(app, username):
        return Result.default_user_error(info=f"Username '{username}' already taken",
                                         interface=ToolBoxInterfaces.remote)

    if not invitation.startswith(Code.one_way_hash(username, "00#", os.getenv("TB_R_KEY", "pepper123"))[:12]):  # not db_valid_invitation(app, invitation):
        return Result.default_user_error(info="Invalid invitation", interface=ToolBoxInterfaces.remote)

    test_bub_key = "Invalid"

    if pub_key:
        if as_base64:
            try:
                pub_key = from_base64(pub_key)
                pub_key = str(pub_key)
            except Exception as e:
                return Result.default_internal_error(info=f"Invalid public key not a valid base64 string: {e}")

        test_bub_key = Code().encrypt_asymmetric(username, pub_key)

    if test_bub_key == "Invalid":
        return Result.default_user_error(info="Invalid public key parsed", interface=ToolBoxInterfaces.remote)

    user = User(name=username,
                email=email,
                user_pass_pub_devices=[pub_key],
                pub_key=pub_key)

    db_delete_invitation(app, invitation)

    if web_data:
        return initialize_and_return(app, user)

    db_helper_save_user(app, asdict(user))

    return Result.ok(info=f"User created successfully: {username}",
                     data=Code().encrypt_asymmetric(str(user.name), pub_key)
                     , interface=ToolBoxInterfaces.remote)


@export(mod_name=Name, state=True, interface=ToolBoxInterfaces.api, api=True, test=False)
async def get_magic_link_email(app: App, username=None):
    if app is None:
        app = get_app(Name + '.get_magic_link_email')

    if not db_helper_test_exist(app, username):
        return Result.default_user_error(info=f"Username '{username}' not known", interface=ToolBoxInterfaces.remote)

    user_r: Result = get_user_by_name(app, username=username)
    user: User = user_r.get()

    if user.challenge == '':
        user = UserCreator(**asdict(user))
        db_helper_save_user(app, asdict(user))

    invitation = "01#" + Code.one_way_hash(user.user_pass_sync, "CM", "get_magic_link_email")
    res = send_magic_link_email(app, user.email, os.getenv("APP_BASE_URL", "http://localhost:8080")+f"/web/assets/m_log_in.html?key={quote(invitation)}&name={user.name}", user.name)
    return res

    # if not invitation.endswith(user.challenge[12:]):

@export(mod_name=Name, state=True, interface=ToolBoxInterfaces.api, api=True, test=False)
def add_user_device(app: App, data: AddUserDeviceObject = None, username: str = 'test-user',
                          pub_key: str = '',
                          invitation: str = '', web_data=False, as_base64=False) -> ApiResult:
    if isinstance(data, dict):
        data = AddUserDeviceObject(**data)

    username = data.name if data is not None else username
    pub_key = data.pub_key if data is not None else pub_key
    invitation = data.invitation if data is not None else invitation
    web_data = data.web_data if data is not None else web_data
    as_base64 = data.as_base64 if data is not None else as_base64

    if app is None:
        app = get_app(Name + '.add_user_device')

    if not db_helper_test_exist(app, username):
        return Result.default_user_error(info=f"Username '{username}' not known", interface=ToolBoxInterfaces.remote)

    if not invitation.startswith("01#"):  # not db_valid_invitation(app, invitation):
        return Result.default_user_error(info="Invalid key", interface=ToolBoxInterfaces.remote)
    invitation = invitation.replace("01#", "")
    test_bub_key = "Invalid"

    if pub_key:
        if as_base64:
            try:
                pub_key = from_base64(pub_key)
                pub_key = str(pub_key)
            except Exception as e:
                return Result.default_internal_error(info=f"Invalid public key not a valid base64 string: {e}")

        test_bub_key = Code().encrypt_asymmetric(username, pub_key)

    if test_bub_key == "Invalid":
        return Result.default_user_error(info="Invalid public key parsed", interface=ToolBoxInterfaces.remote)

    user_r: Result = get_user_by_name(app, username=username)
    user: User = user_r.get()
    s_invite = Code.one_way_hash(user.user_pass_sync, "CM", "get_magic_link_email")
    app.print(f"INVATATION : {invitation} and server whants {s_invite} {s_invite == invitation=}")

    if invitation != s_invite:
        return Result.default_user_error(info="Invalid invitation", interface=ToolBoxInterfaces.remote)

    user.user_pass_pub_devices.append(pub_key)
    user.pub_key = pub_key

    db_delete_invitation(app, invitation)

    if web_data:
        return initialize_and_return(app, user)

    db_helper_save_user(app, asdict(user))

    return Result.ok(info=f"User created successfully: {username}",
                     data=Code().encrypt_asymmetric(str(user.name), pub_key)
                     , interface=ToolBoxInterfaces.remote)


class PersonalData(BaseModel):
    userId: str
    username: str
    pk: str  # arrayBufferToBase64
    #pkAlgo: int
    #authenticatorData: str  # arrayBufferToBase64
    client_json: dict  # arrayBufferToBase64
    sing: str
    # rawId: str  # arrayBufferToBase64
    registration_credential: CustomRegistrationCredential


@export(mod_name=Name, api=True, test=False)
async def register_user_personal_key(app: App, data: PersonalData) -> ApiResult:
    if app is None:
        app = get_app(Name + '.register_user_personal_key')
    username = data.get("username")
    userId = data.get("userId")
    client_json = data.get("client_json")
    sing = data.get("sing")
    # app.print(f"Data : {username=}, {userId=}, {client_json=}, {username is None or userId is None or client_json is None or sing is None=}")
    if username is None or userId is None or client_json is None or sing is None:
        return Result.default_user_error(info="Invalid data")

    if 'registration_credential' not in data or 'response' not in data['registration_credential']:
        return Result.default_user_error(info="Invalid data")

    def base64url_to_bytes(val: str) -> bytes:
        # Add padding back (base64 must be a multiple of 4 chars)
        padded = val + '=' * (-len(val) % 4)
        return base64.urlsafe_b64decode(padded)

    def base64_to_bytes(val: str) -> bytes:
        # Ensure proper base64 padding
        padded = val + '=' * (-len(val) % 4)
        return base64.b64decode(padded)

    data['registration_credential']['raw_id'] = base64url_to_bytes(data['registration_credential']['raw_id'])
    data['registration_credential']['response']['client_data_json'] = base64_to_bytes(data['registration_credential']['response']['client_data_json'])
    data['registration_credential']['response']['attestation_object'] = base64_to_bytes(data['registration_credential']['response']['attestation_object'])
    data['registration_credential']['response'] = AuthenticatorAttestationResponse(**data['registration_credential']['response'])
    registration_credential = CustomRegistrationCredential(**data.get("registration_credential"))
    if not db_helper_test_exist(app, username):
        return Result.default_user_error(info=f"Username '{username}' not known")

    user_result = get_user_by_name(app, username, from_base64(userId).decode())

    if user_result.is_error() and not user_result.is_data():
        return Result.default_internal_error(info="No user found", data=user_result)

    client_json = client_json
    challenge = client_json.get("challenge")
    origin = client_json.get("origin")
    # crossOrigin = client_json.get("crossOrigin")

    if challenge is None:
        return Result.default_user_error(info="No challenge found in data invalid date parsed", data=user_result)

    valid_origen = ["https://simplecore.app", "https://simplecorehub.com", os.getenv("APP_BASE_URL", "http://localhost:8080")] + (
        ["http://localhost:5000", "http://localhost:8080", os.getenv("APP_BASE_URL", "http://localhost:8080")] if app.debug else [])

    if origin not in valid_origen:
        return Result.default_user_error(info=f'Invalid origen: {origin} not in {valid_origen}', data=user_result)

    user: User = user_result.get()

    if challenge != user.challenge:
        return Result.default_user_error(info="Invalid challenge returned", data=user)

    if not Code.verify_signature(signature=from_base64(sing), message=user.challenge, public_key_str=user.pub_key,
                                 salt_length=32):
        return Result.default_user_error(info="Verification failed Invalid signature")
    # c = {   "id": rawId,   "rawId": rawId,   "response": {       "attestationObject": attestationObj,
    # "clientDataJSON": clientJSON,       "transports": ["usb", "nfc", "ble", "internal", "cable", "hybrid"],
    # },   "type": "public-key",   "clientExtensionResults": {},   "authenticatorAttachment": "platform",}
    try:
        expected_rp_id = os.environ.get('APP_BASE_URL', 'localhost')
        if 'simplecore' in expected_rp_id:
            expected_rp_id = "simplecore.app"
        else:
            expected_rp_id = "localhost"
        registration_verification = webauthn.verify_registration_response(
            credential=registration_credential,
            expected_challenge=user.challenge.encode(),
            expected_origin=valid_origen,
            expected_rp_id=expected_rp_id,  # simplecore.app
            require_user_verification=True,
        )
    except InvalidRegistrationResponse as e:
        return Result.default_user_error(info=f"Registration failure : {e}")

    if not registration_verification.user_verified:
        return Result.default_user_error(info="Invalid registration not user verified")

    user_persona_pub_key = {
        'public_key_row': base64.b64encode(registration_verification.credential_public_key).decode('utf-8'),
        'public_key': data.get("pk"),
        'sign_count': registration_verification.sign_count,
        'credential_id': base64.b64encode(registration_verification.credential_id).decode('ascii'),
        'rawId': data.get('raw_id'),
        'attestation_object': base64.b64encode(registration_verification.attestation_object).decode('ascii'),
    }

    user.challenge = ""
    user.user_pass_pub_persona = user_persona_pub_key
    user.is_persona = True

    if user.level == 0:
        user.level = 2

    # Speichern des neuen Benutzers in der Datenbank
    save_result = db_helper_save_user(app, asdict(user))
    if save_result.is_error():
        return save_result.to_api_result()

    key = "01#" + Code.one_way_hash(user.user_pass_sync, "CM", "get_magic_link_email")
    url = f"/web/assets/m_log_in.html?key={quote(key)}&name={user.name}"

    return Result.ok(info="User registered successfully", data=url)


@export(mod_name=Name, state=True, interface=ToolBoxInterfaces.cli, test=False)
def crate_local_account(app: App, username: str, email: str = '', invitation: str = '', create=None) -> Result:
    if app is None:
        app = get_app(Name + '.crate_local_account')
    user_pri = app.config_fh.get_file_handler("Pk" + Code.one_way_hash(username, "dvp-k")[:8])
    if user_pri is not None and db_helper_test_exist(app=app, username=username):
        return Result.default_user_error(info="User already registered on this device")
    pub, pri = Code.generate_asymmetric_keys()
    app.config_fh.add_to_save_file_handler("Pk" + Code.one_way_hash(username, "dvp-k")[:8], pri)
    # if ToolBox_over == 'root' and invitation == '':
    #     invitation = Code.one_way_hash(username, "00#", os.getenv("TB_R_KEY", "pepper123"))[:12] # db_crate_invitation(app)
    if invitation == '':
        return Result.default_user_error(info="No Invitation key provided")

    def create_user_(*args):
        return create_user(app, None, *args)
    if create is not None:
        create_user_ = create

    res = create_user_(username, email, pub, invitation)

    if res.info.exec_code != 0:
        return Result.custom_error(data=res, info="user creation failed!", exec_code=res.info.exec_code)

    return Result.ok(info="Success")


@export(mod_name=Name, state=True, interface=ToolBoxInterfaces.cli, test=False)
async def local_login(app: App, username: str) -> Result:
    if app is None:
        app = get_app(Name + '.local_login')

    is_remote_instance = os.getenv("TOOLBOXV2_IS_REMOTE_INSTANCE", "false").lower() == "true"

    user_pri = app.config_fh.get_file_handler("Pk" + Code.one_way_hash(username, "dvp-k")[:8])
    if user_pri is None:
        return Result.ok(info="No User registered on this device")

    if is_remote_instance:
        # Running on the remote server, use original local authentication
        app.print("Performing local authentication on remote instance.")
        s = await get_to_sing_data(app, username=username)
        signature = Code.create_signature(s.as_result().get('challenge'), user_pri, row=True)
        res = await jwt_get_claim(app, username, signature, web=False)
        res = res.as_result()
    else:
        # Running locally, call the remote server for authentication
        app.print("Performing remote authentication from local instance.")
        remote_session = app.session
        s = await remote_session.fetch(f"/api/CloudM.AuthManager/get_to_sing_data?username={username}", method="POST")
        challenge = s.get('challenge')
        signature = Code.create_signature(challenge, user_pri, row=True)
        res = await remote_session.fetch("/api/CloudM.AuthManager/jwt_get_claim", method="POST", data={'username': username, 'signature': signature})

    if res.info.exec_code != 0:
        return Result.custom_error(data=res, info="user login failed!", exec_code=res.info.exec_code)

    return Result.ok(info="Success", data=res.get())


@export(mod_name=Name, api=True, test=False, request_as_kwarg=True)
async def get_to_sing_data(app: App, username=None, personal_key: Any = False, data=None, request=None):  # Use Any for personal_key initially
    t0 = time.time()
    if app is None:
        app = get_app(from_=Name + '.get_to_sing_data')

    is_remote_instance = os.getenv("TOOLBOXV2_IS_REMOTE_INSTANCE", "false").lower() == "true"

    if username is None:
        username = data.get("username")
        personal_key = data.get("personal_key")

    user_result = get_user_by_name(app, username)
    if user_result.is_error() or not user_result.is_data():
        return Result.default_user_error(info=f"User {username} is not a valid user")
    user: User = user_result.get()

    # Generate a new, plain challenge for WebAuthn or device key flow
    new_challenge = base64.urlsafe_b64encode(os.urandom(32)).rstrip(b'=').decode('utf-8')
    user.challenge = new_challenge  # Store the plain challenge

    save_res = db_helper_save_user(app, asdict(user))
    if save_res.is_error():
        app.print(f"Failed to save user {username} with new challenge: {save_res.info.help_text}", level="ERROR")
        return Result.default_internal_error(info="Failed to prepare session challenge.")

    data_to_return = {'challenge': user.challenge}

    # Handle personal_key being passed as string 'true'/'false' or boolean
    if isinstance(personal_key, str):
        is_personal_key_true = personal_key.lower() == 'true'
    elif isinstance(personal_key, bool):
        is_personal_key_true = personal_key
    else:
        is_personal_key_true = False  # Default if type is unexpected

    if is_personal_key_true:
        # rawId for WebAuthn login, stored during registration.
        # Client sends: raw_id: arrayBufferToBase64(credential.rawId)
        # Server stores: user.user_pass_pub_persona['rawId'] = data.get('raw_id')
        # This rawId is standard base64 of the credential's raw ID bytes.
        stored_raw_id = user.user_pass_pub_persona.get("rawId")
        if not stored_raw_id:
            return Result.default_user_error(
                info=f"User {username} has no WebAuthn credential registered (missing rawId).")
        data_to_return['rowId'] = stored_raw_id  # Client expects 'rowId'

    app.print(
        f"END get_to_sing_data for {username}, personal_key={is_personal_key_true}, took {time.perf_counter() - t0:.4f}s", )
    return Result.ok(data=data_to_return, info="Challenge returned")


@export(mod_name=Name, state=True, interface=ToolBoxInterfaces.native, api=False, level=999, test=False)
def get_invitation(app: App, username='') -> Result:
    if app is None:
        app = get_app(Name + '.test_invations')

    invitation = Code.one_way_hash(username, "00#", os.getenv("TB_R_KEY", "pepper123"))[:12]#"00#" + str(Code.generate_seed())  # db_crate_invitation(app)
    return Result.ok(data=invitation)


# a sync contention between server and user

class VdUSER(BaseModel):
    username: str
    signature: str  # Base64 encoded signature for device key validation


class ClientAuthNCredentialPayload(BaseModel):
    id: str  # Base64URL string (credential.id from browser)
    raw_id: str  # Base64URL string (also credential.id from browser, will be decoded to bytes)
    type: str
    authenticator_attachment: str | None = None
    response: dict[str, Any]  # Keep as dict for now, will be processed
    # client_extension_results: Optional[Dict[str, Any]] = None # If you use extensions


class VpUSER(BaseModel):
    username: str
    # signature: str # This top-level signature is for device key, not WebAuthn
    authentication_credential: ClientAuthNCredentialPayload  # Use the defined Pydantic model


@export(mod_name=Name, api=True, test=False)
async def validate_persona(app: App, data: VpUSER) -> ApiResult:
    if app is None:
        app = get_app(f"{Name}.validate_persona")
    if isinstance(data, dict):
        data = VpUSER(**data)
    auth_cred_payload = data.authentication_credential

    # --- 1. Prepare response object for webauthn library ---
    response_payload = auth_cred_payload.response
    if not isinstance(response_payload, dict):
        return Result.default_user_error(info="authentication_credential.response must be an object.")

    try:
        client_data_json_bytes = base64_std_to_bytes(response_payload.get('client_data_json'))
        authenticator_data_bytes = base64_std_to_bytes(response_payload.get('authenticator_data'))
        signature_bytes = base64_std_to_bytes(response_payload.get('signature'))
        user_handle_bytes = base64_std_to_bytes(response_payload.get('user_handle')) if response_payload.get(
            'user_handle') else None

        if not all([client_data_json_bytes, authenticator_data_bytes, signature_bytes]):
            missing = [
                f for f_name, f in [
                    ('client_data_json', client_data_json_bytes),
                    ('authenticator_data', authenticator_data_bytes),
                    ('signature', signature_bytes)
                ] if not f
            ]
            return Result.default_user_error(
                info=f"Missing required fields in authentication_credential.response: {', '.join(missing)}")

    except ValueError as e:  # Catch decoding errors from base64_std_to_bytes
        get_logger().error(f"Base64 decoding error for user {data.username} in validate_persona: {e}")
        return Result.default_user_error(info=f"Invalid base64 encoding in authentication_credential.response: {e}")
    except Exception as e:
        get_logger().error(f"Error processing response for user {data.username} in validate_persona: {e}")
        return Result.default_internal_error(info="Error processing authentication response.")

    # --- 2. Get User and WebAuthn specific data ---
    user_result = get_user_by_name(app, data.username)
    if user_result.is_error() or not user_result.is_data():
        return Result.default_user_error(info=f"Invalid username: {data.username}")
    user: User = user_result.get()

    if not user.is_persona:
        return Result.default_user_error(info=f"No Persona key (WebAuthn) registered for user {data.username}.")

    cose_credential_public_key_b64 = user.user_pass_pub_persona.get("public_key_row")
    if not cose_credential_public_key_b64:
        app.print(f"WebAuthn Error: 'public_key_row' (COSE key) not found for user {data.username}.", level="ERROR")
        return Result.default_user_error(info="WebAuthn credential public key not found. Please re-register passkey.")
    try:
        credential_public_key_bytes = base64.b64decode(cose_credential_public_key_b64.encode('utf-8'))
    except Exception as e:
        app.print(f"Error decoding 'public_key_row' for user {data.username}: {e}", level="ERROR")
        return Result.default_user_error(info="Failed to decode WebAuthn credential public key.")

    # --- 3. Construct webauthn library's Structs ---
    try:
        # raw_id from client is base64url string (assertion.id)
        raw_id_bytes = b64decode(auth_cred_payload.raw_id)  # b64decode for URL-safe

        auth_response_struct = WebAuthnAuthenticatorAssertionResponse(
            client_data_json=client_data_json_bytes,
            authenticator_data=authenticator_data_bytes,
            signature=signature_bytes,
            user_handle=user_handle_bytes
        )
        auth_cred_struct = WebAuthnAuthenticationCredential(
            id=auth_cred_payload.id,  # This is the base64url string ID
            raw_id=raw_id_bytes,  # These are the raw bytes of the ID
            response=auth_response_struct,
            type=auth_cred_payload.type,
            # client_extension_results=auth_cred_payload.client_extension_results, # If you use them
            authenticator_attachment=auth_cred_payload.authenticator_attachment
        )
    except Exception as e:
        get_logger().error(f"Error constructing webauthn library structs for user {data.username}: {e}", exc_info=True)
        return Result.default_internal_error(info="Failed to prepare WebAuthn validation data structure.")

    # --- 4. Determine Expected RP ID and Origin ---
    # Client getRpId(): localhost or actual hostname. For production, eTLD+1.
    # Server's expected_rp_id should match this.
    app_base_url_str = os.getenv("APP_BASE_URL", "http://localhost:8080")  # Default if not set
    parsed_app_url = urlparse(app_base_url_str)
    hostname_from_env = parsed_app_url.hostname

    if hostname_from_env and (hostname_from_env.startswith("localhost") or hostname_from_env == "127.0.0.1"):
        expected_rp_id = "localhost"
    elif hostname_from_env and "simplecore.app" in hostname_from_env:
        expected_rp_id = "simplecore.app"  # eTLD+1
    elif hostname_from_env and "simplecorehub.com" in hostname_from_env:
        expected_rp_id = "simplecorehub.com"  # eTLD+1
    elif hostname_from_env:
        expected_rp_id = hostname_from_env  # May need refinement to eTLD+1 for other domains
    else:  # Fallback if APP_BASE_URL is malformed or missing hostname
        expected_rp_id = "localhost"

    valid_origins = [app_base_url_str]  # Primary origin from APP_BASE_URL
    if "localhost" in app_base_url_str:  # For local dev, window.location.origin might include port
        if "http://localhost:8080" not in valid_origins: valid_origins.append("http://localhost:8080")
        if "http://localhost:5000" not in valid_origins: valid_origins.append("http://localhost:5000")  # if used
    # Add production origins explicitly if different or need subdomains
    if "https://simplecore.app" not in valid_origins: valid_origins.append("https://simplecore.app")
    if "https://simplecorehub.com" not in valid_origins: valid_origins.append("https://simplecorehub.com")

    # --- 5. Perform WebAuthn Verification ---
    try:
        current_sign_count = user.user_pass_pub_persona.get("sign_count")
        if not isinstance(current_sign_count, int):
            try:
                current_sign_count = int(current_sign_count)
            except (ValueError, TypeError):
                get_logger().warning(f"Invalid sign_count for user {data.username}: {current_sign_count}. Assuming 0.")
                current_sign_count = 0  # Or handle as error if strict count checking is critical

        get_logger().debug(
            f"WebAuthn Verify Params for {data.username}: "
            f"expected_rp_id='{expected_rp_id}', "
            f"expected_challenge(start)='{user.challenge[:10]}...', "
            f"expected_origin(s)='{valid_origins}', "
            f"sign_count={current_sign_count}"
        )

        authentication_verification = webauthn.verify_authentication_response(
            credential=auth_cred_struct,
            expected_challenge=user.challenge.encode('utf-8'),  # Plain challenge, UTF-8 encoded
            expected_rp_id=expected_rp_id,
            expected_origin=valid_origins,  # List of allowed origins
            credential_public_key=credential_public_key_bytes,
            credential_current_sign_count=current_sign_count,
            require_user_verification=True,  # Matches client authenticatorSelection
        )
        get_logger().info(
            f"Authentication Verification Success for {user.name}. New sign count: {authentication_verification.new_sign_count}")
        user.user_pass_pub_persona["sign_count"] = authentication_verification.new_sign_count
    except InvalidAuthenticationResponse as e:
        client_data_json_str = auth_response_struct.client_data_json.decode('utf-8',
                                                                            errors='replace') if auth_response_struct.client_data_json else "N/A"
        get_logger().warning(
            f"WebAuthn InvalidAuthenticationResponse for user {data.username}. Error: {e}. "
            f"Challenge used (server): '{user.challenge}'. RP ID used (server): '{expected_rp_id}'. "
            f"ClientDataJSON challenge (from client): '{json.loads(client_data_json_str).get('challenge', 'MISSING') if client_data_json_str != 'N/A' else 'N/A'}'"
        )
        return Result.default_user_error(info=f"Authentication failure: {e}")
    except Exception as e:
        get_logger().error(f"Unexpected error during WebAuthn verification for user {data.username}: {e}",
                           exc_info=True)
        return Result.default_internal_error(info=f"An unexpected error occurred during WebAuthn verification: {e}")

    # --- 6. Post-Authentication ---
    save_result = db_helper_save_user(app, asdict(user))  # Save updated sign_count
    if save_result.is_error():
        return save_result.to_api_result()  # Propagate DB error

    # The redirect URL seems to be for initiating another device registration via magic link.
    # Typically, after login, you'd issue a session token (JWT) or set a session cookie.
    # This part might need review based on your intended post-login flow.
    magic_link_key_segment = Code.one_way_hash(user.user_pass_sync, "CM", "get_magic_link_email")
    redirect_url = f"/web/assets/m_log_in.html?key={quote('01#' + magic_link_key_segment)}&name={user.name}"

    get_logger().info(f"User {data.username} successfully authenticated via WebAuthn. Redirecting to: {redirect_url}")
    return Result.ok(data=redirect_url, info="Authentication successful. Redirecting...")


@export(mod_name=Name, api=True, test=False)
async def validate_device(app: App, data: VdUSER) -> ApiResult:
    if app is None:
        app = get_app(".validate_device")

    if isinstance(data, dict):
        data = VdUSER(**data)

    user_result = get_user_by_name(app, data.username)

    if user_result.is_error() or not user_result.is_data():
        return Result.default_user_error(info=f"Invalid username : {data.username}")

    user: User = user_result.get()

    valid = False

    for divce_keys in user.user_pass_pub_devices:
        valid = Code.verify_signature(signature=from_base64(data.signature),
                                      message=user.challenge,
                                      public_key_str=divce_keys,
                                      salt_length=32)
        if valid:
            user.pub_key = divce_keys
            break

    if not valid:
        return Result.default_user_error(info=f"Invalid signature : {data.username}")

    user.challenge = ""
    if user.user_pass_pri == "":
        user = UserCreator(**asdict(user))
    db_helper_save_user(app, asdict(user))

    claim = {
        "u-key": Code.one_way_hash(user.uid, Name)[:16],
    }

    row_jwt_claim = crate_jwt(claim, user.user_pass_pri)

    encrypt_jwt_claim = Code.encrypt_asymmetric(row_jwt_claim, user.pub_key)

    if encrypt_jwt_claim != "Invalid":
        data = {'key': encrypt_jwt_claim, 'toPrivat': True}
    else:
        data = {'key': row_jwt_claim, 'toPrivat': False}

    return Result.ok(data=data)


@export(mod_name=Name, state=True, interface=ToolBoxInterfaces.remote, api=True, test=False)
async def authenticate_user_get_sync_key(app: App, username: str, signature: str or bytes, get_user=False,
                                         web=False) -> ApiResult:
    if app is None:
        app = get_app(Name + '.authenticate_user_get_sync_key')

    user_r: Result = get_user_by_name(app, username=username)
    user: User = user_r.get()

    if user is None:
        return Result.default_internal_error(info="User not found", exec_code=404)

    if web:
        if not Code.verify_signature_web_algo(signature=signature,
                                              message=to_base64(
                                                  user.challenge),
                                              public_key_str=user.pub_key):
            return Result.default_user_error(info="Verification failed Invalid signature")
    else:
        if not Code.verify_signature(signature=signature,
                                     message=user.challenge,
                                     public_key_str=user.pub_key):
            return Result.default_user_error(info="Verification failed Invalid signature")

    user = UserCreator(**asdict(user))

    db_helper_save_user(app, asdict(user))

    crypt_sync_key = Code.encrypt_asymmetric(user.user_pass_sync, user.pub_key)

    if get_user:
        return Result.ok(data_info="Returned Sync Key, read only for user (withe user_data)",
                         data=(crypt_sync_key, asdict(user)))

    return Result.ok(data_info="Returned Sync Key, read only for user", data=crypt_sync_key)


# local user functions

@export(mod_name=Name, state=True, interface=ToolBoxInterfaces.native, test=False)
async def get_user_sync_key_local(app: App, username: str, ausk=None) -> Result:
    if app is None:
        app = get_app(Name + '.get_user_sync_key')

    user_pri = app.config_fh.get_file_handler("Pk" + Code.one_way_hash(username)[:8])

    sing_r = await get_to_sing_data(app, username=username)
    signature = Code.create_signature(sing_r.get('challenge'), user_pri)

    def authenticate_user_get_sync_key_(*args):
        return authenticate_user_get_sync_key(*args)
    if ausk is not None:
        authenticate_user_get_sync_key_ = ausk

    res = await authenticate_user_get_sync_key_(app, username, signature)
    res = res.as_result()

    if res.info.exec_code != 0:
        return Result.custom_error(data=res, info="user get_user_sync_key failed!", exec_code=res.info.exec_code)

    sync_key = res.get()

    app.config_fh.add_to_save_file_handler("SymmetricK", sync_key)

    return Result.ok(info="Success", data=Code.decrypt_asymmetric(sync_key, user_pri))


# jwt claim

@export(mod_name=Name, state=True, interface=ToolBoxInterfaces.remote, api=True, test=False)
async def jwt_get_claim(app: App, username: str, signature: str or bytes, web=False) -> ApiResult:
    if app is None:
        app = get_app(Name + '.jwt_claim_server_side_sync')

    res = await authenticate_user_get_sync_key(app, username, signature, get_user=True, web=web)
    res = res.as_result()

    if res.info.exec_code != 0:
        return res.custom_error(data=res)

    channel_key, userdata = res.get()
    claim = {
        "u-key": Code.one_way_hash(userdata.get("uid"), Name)[:16],
    }

    row_jwt_claim = crate_jwt(claim, userdata.get("user_pass_pri"))

    return Result.ok(
        data={'claim': Code.encrypt_symmetric(row_jwt_claim, userdata.get("pub_key")), 'key': channel_key})


@export(mod_name=Name, state=True, interface=ToolBoxInterfaces.remote, api=False, test=False)
async def jwt_claim_local_decrypt(app: App, username: str, crypt_sing_jwt_claim: str, aud=None) -> Result:
    if app is None:
        app = get_app(Name + '.jwt_claim_server_side_sync_local')

    user_sync_key_res = await get_user_sync_key_local(app, username, ausk=aud)

    if user_sync_key_res.info.exec_code != 0:
        return Result.custom_error(data=user_sync_key_res)

    user_sync_key = user_sync_key_res.get()

    sing_jwt_claim = Code.decrypt_symmetric(crypt_sing_jwt_claim, user_sync_key)
    claim = await jwt_check_claim_server_side(app, username, sing_jwt_claim)
    return claim.as_result().lazy_return('raise')


@export(mod_name=Name, state=True, interface=ToolBoxInterfaces.remote, api=True, test=False)
async def jwt_check_claim_server_side(app: App, username: str, jwt_claim: str) -> ApiResult:
    res = get_user_by_name(app, username)
    if res.info.exec_code != 0:
        return Result.custom_error(data=res)
    user: User = res.get()

    data = validate_jwt(jwt_claim, user.user_pass_pub)
    print("data::::::::::::", username, data, type(data))
    # InvalidSignatureError
    if isinstance(data, str):
        return Result.custom_error(info="Invalid", data=False)

    return Result.ok(data_info='Valid JWT', data=True)


# ============================= Unit tests ===========================================

# set up
@export(mod_name=Name, test_only=True, initial=True, state=False)
def prep_test():
    app = get_app(f"{Name}.prep_test")
    app.run_any(TBEF.DB.EDIT_PROGRAMMABLE, mode=DatabaseModes.LC)


def get_test_app_gen(app=None):
    if app is None:
        app = get_app('test-app', name='test-debug')
    yield app
    # Teardown-Logik hier, falls benötigt


def helper_gen_test_app():
    _ = get_test_app_gen(None)
    TestAppGen.t = _, next(_)
    prep_test()
    return TestAppGen


class TestAppGen:
    t: tuple

    @staticmethod
    def get():
        return TestAppGen.t


@test_only
async def helper_test_user():
    app: App
    test_app, app = helper_gen_test_app().get()
    username = "testUser123" + uuid.uuid4().hex
    email = "test_mainqmail.com"
    db_helper_delete_user(app, username, "*", matching=True)
    # Benutzer erstellen
    r = crate_local_account(app, username, email, get_invitation(app, username).get())
    assert not r.is_error(), r.print(show=False)
    r = crate_local_account(app, username, email, get_invitation(app, username).get())
    assert r.is_error(), r.print(show=False)
    # Aufräumen
    db_helper_delete_user(app, username, "*", matching=True)
    app.config_fh.remove_key_file_handler("Pk" + Code.one_way_hash(username, "dvp-k")[:8])
    return Result.ok()


@test_only
async def helper_test_create_user_and_login():
    app: App
    test_app, app = helper_gen_test_app().get()
    username = "testUser123" + uuid.uuid4().hex
    email = "test_mainqmail.com"
    r = crate_local_account(app, username, email, get_invitation(app, username).get())
    r2 = await local_login(app, username)
    assert not r.is_error(), r.print(show=False)
    assert not r2.is_error(), r2.print(show=False)
    app.config_fh.remove_key_file_handler("Pk" + Code.one_way_hash(username, "dvp-k")[:8])
    db_helper_delete_user(app, username, "*", matching=True)
    return Result.ok()


@test_only
async def helper_test_validate_device(app: App = None):
    test_app, app = helper_gen_test_app().get()

    # Schritt 1: Benutzer erstellen
    username = "testUser" + uuid.uuid4().hex
    email = "test@example.com"
    pub_key, pri_key = Code.generate_asymmetric_keys()
    user = UserCreator(name=username, email=email, user_pass_pub_devices=[pub_key], pub_key=pub_key)
    db_helper_save_user(app, asdict(user))

    # Schritt 2: Signatur generieren
    s = await get_to_sing_data(app, username=username)
    signature = Code.create_signature(s.as_result().get('challenge'),
                                      pri_key, row=False, salt_length=32)

    # Schritt 3: Testdaten vorbereiten
    test_data = VdUSER(username=username, signature=signature)
    # Schritt 4: validate_device Funktion testen
    result = await validate_device(app, test_data)
    result = result.as_result()
    result.print()
    # Schritt 5: Ergebnisse überprüfen
    assert not result.is_error(), f"Test fehlgeschlagen: {result.print(show=False)}"
    assert result.is_data(), "Kein Schlüssel im Ergebnis gefunden"

    # Aufräumen: Benutzer aus der Datenbank entfernen
    db_helper_delete_user(app, username, user.uid)

    return Result.ok()


def test_helper0():
    asyncio.run(helper_test_user())


def test_helper1():
    asyncio.run(helper_test_create_user_and_login())


def test_helper2():
    asyncio.run(helper_test_validate_device())
