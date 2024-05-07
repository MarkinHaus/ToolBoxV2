import asyncio
import base64
import datetime
import json
import os
import uuid
from dataclasses import asdict
from urllib.parse import quote

import jwt
import webauthn
from pydantic import BaseModel, field_validator
from pydantic import validator
from webauthn.helpers.exceptions import InvalidAuthenticationResponse, InvalidRegistrationResponse
from webauthn.helpers.structs import AuthenticationCredential, RegistrationCredential

from toolboxv2 import get_app, App, Result, tbef, ToolBox_over, get_logger
from toolboxv2.mods.DB.types import DatabaseModes
from toolboxv2.utils.security.cryp import Code
from toolboxv2.utils.system.types import ToolBoxInterfaces, ApiResult
from .types import UserCreator, User

Name = 'CloudM.AuthManager'
export = get_app(f"{Name}.Export").tb
default_export = export(mod_name=Name, test=False)
test_only = export(mod_name=Name, test_only=True)
version = '0.0.1'
instance_bios = str(uuid.uuid4())


def b64decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s.encode())


class CustomAuthenticationCredential(AuthenticationCredential):
    @field_validator('raw_id')
    def convert_raw_id(cls, v: str):
        assert isinstance(v, str), 'raw_id is not a string'
        return b64decode(v)

    @field_validator('response')
    def convert_response(cls, data: dict):
        assert isinstance(data, dict), 'response is not a dictionary'
        return {k: b64decode(v) for k, v in data.items()}


class CustomRegistrationCredential(RegistrationCredential):
    @field_validator('raw_id')
    def convert_raw_id(cls, v: str):
        assert isinstance(v, str), 'raw_id is not a string'
        return b64decode(v)

    @field_validator('response')
    def convert_response(cls, data: dict):
        assert isinstance(data, dict), 'response is not a dictionary'
        return {k: b64decode(v) for k, v in data.items()}


# app Helper functions interaction with the db

async def db_helper_test_exist(app: App, username: str):
    c = app.run_any(tbef.DB.IF_EXIST, query=f"USER::{username}::*", get_results=True)
    if c.is_error(): raise RuntimeError(f"DB - error {c.print(show=False)}")
    b = c.get() > 0
    get_logger().info(f"TEST IF USER EXIST : {username} {b}")
    return b


def db_delete_invitation(app: App, invitation: str):
    return app.run_any(tbef.DB.DELETE, query=f"invitation::{invitation}", get_results=True)


def db_valid_invitation(app: App, invitation: str):
    inv_key = app.run_any(tbef.DB.GET, query=f"invitation::{invitation}", get_results=False)
    if inv_key is None:
        return False
    inv_key = inv_key[0]
    if isinstance(inv_key, bytes):
        inv_key = inv_key.decode()
    return Code.decrypt_symmetric(inv_key, invitation) == invitation


def db_crate_invitation(app: App):
    invitation = Code.generate_symmetric_key()
    inv_key = Code.encrypt_symmetric(invitation, invitation)
    res = app.run_any(tbef.DB.SET, query=f"invitation::{invitation}", data=inv_key, get_results=True)
    return invitation


def db_helper_save_user(app: App, user_data: dict) -> Result:
    # db_helper_delete_user(app, user_data['name'], user_data['uid'], matching=True)
    return app.run_any(tbef.DB.SET, query=f"USER::{user_data['name']}::{user_data['uid']}",
                       data=user_data,
                       get_results=True)


def db_helper_get_user(app: App, username: str, uid: str = '*'):
    return app.run_any(tbef.DB.GET, query=f"USER::{username}::{uid}",
                       get_results=True)


def db_helper_delete_user(app: App, username: str, uid: str, matching=False):
    return app.run_any(tbef.DB.DELETE, query=f"USER::{username}::{uid}", matching=matching,
                       get_results=True)


# jwt helpers


def add_exp(massage: dict, hr_ex=2):
    massage['exp'] = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(hours=hr_ex)
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
async def get_user_by_name(app: App, username: str, uid: str = '*') -> Result:
    if app is None:
        app = get_app(Name + '.get_user_by_name')

    if not await db_helper_test_exist(app, username):
        return Result.default_user_error(info=f"get_user_by_name failed username'{username}'not registered")

    user_data = db_helper_get_user(app, username, uid)

    if user_data.is_error():
        return Result.default_internal_error(info="get_user_by_name failed no User data found is_error")

    user_data = await user_data.aget()

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
    return db_helper_save_user(app, asdict(user)).lazy_return('intern', data={
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


@export(mod_name=Name, state=True, interface=ToolBoxInterfaces.api, api=True, test=False)
async def create_user(app: App, data: CreateUserObject = None, username: str = 'test-user',
                      email: str = 'test@user.com',
                      pub_key: str = '',
                      invitation: str = '', web_data=False, as_base64=False) -> ApiResult:
    username = data.name if data is not None else username
    email = data.email if data is not None else email
    pub_key = data.pub_key if data is not None else pub_key
    invitation = data.invitation if data is not None else invitation
    web_data = data.web_data if data is not None else web_data
    as_base64 = data.as_base64 if data is not None else as_base64

    if app is None:
        app = get_app(Name + '.crate_user')

    if await db_helper_test_exist(app, username):
        return Result.default_user_error(info=f"Username '{username}' already taken",
                                         interface=ToolBoxInterfaces.remote)

    if not invitation.startswith("00#"):  # not db_valid_invitation(app, invitation):
        return Result.default_user_error(info=f"Invalid invitation", interface=ToolBoxInterfaces.remote)

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

    result_s = db_helper_save_user(app, asdict(user))

    return Result.ok(info=f"User created successfully: {username}",
                     data=Code().encrypt_asymmetric(str(user.name), pub_key)
                     , interface=ToolBoxInterfaces.remote)


@export(mod_name=Name, state=True, interface=ToolBoxInterfaces.api, api=True, test=False)
async def get_magick_link_email(app: App, username):
    if app is None:
        app = get_app(Name + '.get_magick_link_email')

    if not await db_helper_test_exist(app, username):
        return Result.default_user_error(info=f"Username '{username}' not known", interface=ToolBoxInterfaces.remote)

    user_r: Result = await get_user_by_name(app, username=username)
    user: User = user_r.get()

    if user.challenge == '':
        user = UserCreator(**asdict(user))

    invitation = "01#" + Code.one_way_hash(user.user_pass_sync, "CM", "get_magick_link_email")
    nl = len(user.name)
    email_data_result = app.run_any(tbef.EMAIL_WAITING_LIST.CRATE_MAGICK_LICK_DEVICE_EMAIL,
                                    user_email=user.email,
                                    user_name=user.name,
                                    link_id=invitation, nl=nl, get_results=True)

    if email_data_result.is_error() and not email_data_result.is_data():
        return email_data_result

    email_data = email_data_result.get()

    return app.run_any(tbef.EMAIL_WAITING_LIST.SEND_EMAIL, data=email_data, get_results=True)

    # if not invitation.endswith(user.challenge[12:]):


@export(mod_name=Name, state=True, interface=ToolBoxInterfaces.api, api=True, test=False)
async def add_user_device(app: App, data: AddUserDeviceObject = None, username: str = 'test-user',
                          pub_key: str = '',
                          invitation: str = '', web_data=False, as_base64=False) -> ApiResult:
    username = data.name if data is not None else username
    pub_key = data.pub_key if data is not None else pub_key
    invitation = data.invitation if data is not None else invitation
    web_data = data.web_data if data is not None else web_data
    as_base64 = data.as_base64 if data is not None else as_base64

    if app is None:
        app = get_app(Name + '.add_user_device')

    if not await db_helper_test_exist(app, username):
        return Result.default_user_error(info=f"Username '{username}' not known", interface=ToolBoxInterfaces.remote)

    if not invitation.startswith("01#"):  # not db_valid_invitation(app, invitation):
        return Result.default_user_error(info=f"Invalid invitation", interface=ToolBoxInterfaces.remote)
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

    user_r: Result = await get_user_by_name(app, username=username)
    user: User = user_r.get()

    if invitation != Code.one_way_hash(user.user_pass_sync, "CM", "get_magick_link_email"):
        return Result.default_user_error(info=f"Invalid invitation", interface=ToolBoxInterfaces.remote)

    user.user_pass_pub_devices.append(pub_key)
    user.pub_key = pub_key

    db_delete_invitation(app, invitation)

    if web_data:
        return initialize_and_return(app, user)

    result_s = db_helper_save_user(app, asdict(user))

    return Result.ok(info=f"User created successfully: {username}",
                     data=Code().encrypt_asymmetric(str(user.name), pub_key)
                     , interface=ToolBoxInterfaces.remote)


class PersonalData(BaseModel):
    userId: str
    username: str
    pk: str  # arrayBufferToBase64
    pkAlgo: int
    authenticatorData: str  # arrayBufferToBase64
    clientJson: str  # arrayBufferToBase64
    sing: str
    rawId: str  # arrayBufferToBase64
    registration_credential: CustomRegistrationCredential


@export(mod_name=Name, api=True, test=False)
async def register_user_personal_key(app: App, data: PersonalData) -> ApiResult:
    if not await db_helper_test_exist(app, data.username):
        return Result.default_user_error(info=f"Username '{data.username}' not known")

    user_result = await get_user_by_name(app, data.username, from_base64(data.userId).decode())

    if user_result.is_error() and not user_result.is_data():
        return Result.default_internal_error(info="No user found", data=user_result)

    client_json = json.loads(from_base64(data.clientJson))
    challenge = client_json.get("challenge")
    origin = client_json.get("origin")
    # crossOrigin = client_json.get("crossOrigin")

    if challenge is None:
        return Result.default_user_error(info="No challenge found in data invalid date parsed", data=user_result)

    valid_origen = ["https://simplecore.app", "http://localhost:5000"] + (
        ["http://localhost:5000"] if app.debug else [])

    if origin not in valid_origen:
        return Result.default_user_error(info=f'Invalid origen: {origin} not in {valid_origen}', data=user_result)

    user: User = user_result.get()

    if not challenge == to_base64(user.challenge).decode():
        return Result.default_user_error(info="Invalid challenge returned", data=user)

    if not Code.verify_signature(signature=from_base64(data.sing), message=user.challenge, public_key_str=user.pub_key,
                                 salt_length=32):
        return Result.default_user_error(info="Verification failed Invalid signature")
    # c = {   "id": data.rawId,   "rawId": data.rawId,   "response": {       "attestationObject": data.attestationObj,
    # "clientDataJSON": data.clientJSON,       "transports": ["usb", "nfc", "ble", "internal", "cable", "hybrid"],
    # },   "type": "public-key",   "clientExtensionResults": {},   "authenticatorAttachment": "platform",}
    try:
        registration_verification = webauthn.verify_registration_response(
            credential=data.registration_credential,
            expected_challenge=user.challenge.encode(),
            expected_origin=valid_origen,
            expected_rp_id=os.environ.get('HOSTNAME', 'localhost'),  # simplecore.app
            require_user_verification=True,
        )
    except InvalidRegistrationResponse as e:
        return Result.default_user_error(info=f"Registration failure : {e}")

    if not registration_verification.user_verified:
        return Result.default_user_error(info="Invalid registration not user verified")

    user_persona_pub_key = {
        'public_key': registration_verification.credential_public_key,
        'sign_count': registration_verification.sign_count,
        'credential_id': registration_verification.credential_id,
        'rawId': data.rawId,
        'attestation_object': registration_verification.attestation_object,
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

    key = "01#" + Code.one_way_hash(user.user_pass_sync, "CM", "get_magick_link_email")
    url = f"/web/assets/m_log_in.html?key={quote(key)}&name={user.name}"

    return Result.ok(info="User registered successfully", data=url)


@export(mod_name=Name, state=True, interface=ToolBoxInterfaces.cli, test=False)
async def crate_local_account(app: App, username: str, email: str = '', invitation: str = '', create=None) -> Result:
    if app is None:
        app = get_app(Name + '.crate_local_account')
    user_pri = app.config_fh.get_file_handler("Pk" + Code.one_way_hash(username, "dvp-k")[:8])
    if user_pri is not None and await db_helper_test_exist(app=app, username=username):
        return Result.default_user_error(info="User already registered on this device")
    pub, pri = Code.generate_asymmetric_keys()
    app.config_fh.add_to_save_file_handler("Pk" + Code.one_way_hash(username, "dvp-k")[:8], pri)
    if ToolBox_over == 'root' and invitation == '':
        invitation = db_crate_invitation(app)
    if invitation == '':
        return Result.default_user_error(info="No Invitation key provided")

    create_user_ = lambda *args: create_user(app, None, *args)
    if create is not None:
        create_user_ = create

    res = await create_user_(username, email, pub, invitation)

    if res.info.exec_code != 0:
        return Result.custom_error(data=res, info="user creation failed!", exec_code=res.info.exec_code)

    return Result.ok(info="Success")


@export(mod_name=Name, state=True, interface=ToolBoxInterfaces.cli, test=False)
async def local_login(app: App, username: str) -> Result:
    if app is None:
        app = get_app(Name + '.local_login')
    user_pri = app.config_fh.get_file_handler("Pk" + Code.one_way_hash(username, "dvp-k")[:8])
    if user_pri is None:
        return Result.ok(info="No User registered on this device")

    s = await get_to_sing_data(app, username=username)

    signature = Code.create_signature(s.as_result().get('challenge'), user_pri
                                      , row=True)

    res = await jwt_get_claim(app, username, signature, web=False)
    res = res.as_result()

    if res.info.exec_code != 0:
        return Result.custom_error(data=res, info="user login failed!", exec_code=res.info.exec_code)

    return Result.ok(info="Success", data=res.get())


@export(mod_name=Name, api=True, test=False)
async def get_to_sing_data(app: App, username, personal_key=False):
    if app is None:
        app = get_app(from_=Name + '.get_to_sing_data')

    user_result = await get_user_by_name(app, username)

    if user_result.is_error() and not user_result.is_data():
        return Result.default_user_error(info=f"User {username} is not a valid user")

    user: User = user_result.get()

    if user.challenge == "":
        user.challenge = Code.encrypt_asymmetric(str(uuid.uuid4()), user.user_pass_pub)
        db_helper_save_user(app, asdict(user))

    data = {'challenge': user.challenge}

    if personal_key:
        data['rowId'] = user.user_pass_pub_persona.get("rawId")

    return Result.ok(data=data)


@export(mod_name=Name, state=True, interface=ToolBoxInterfaces.native, api=False, level=999, test=False)
def get_invitation(app: App) -> Result:
    if app is None:
        app = get_app(Name + '.test_invations')

    invitation = "00#" + str(Code.generate_seed())  # db_crate_invitation(app)
    return Result.ok(data=invitation)


# a sync contention between server and user

class VdUSER(BaseModel):
    username: str
    signature: str


class VpUSER(VdUSER, BaseModel):
    authentication_credential: CustomAuthenticationCredential


@export(mod_name=Name, api=True, test=False)
async def validate_persona(app: App, data: VpUSER) -> ApiResult:
    if app is None:
        app = get_app(".validate_persona")

    user_result = await get_user_by_name(app, data.username)

    if user_result.is_error() or not user_result.is_data():
        return Result.default_user_error(info=f"Invalid username : {data.username}")
    # from_base64(data.signature)
    user: User = user_result.get()

    if user.is_persona == "":
        return Result.default_user_error(info="No Persona key registered")

    valid_origen = ["https://simplecore.app"] + (
        ["http://localhost:5000"] if app.debug else [])

    try:
        authentication_verification = webauthn.verify_authentication_response(
            # daemonstrating the ability to handle a stringified JSON version of the WebAuthn response
            credential=data.authentication_credential,
            expected_challenge=user.challenge.encode(),
            expected_rp_id=os.environ.get('HOSTNAME', 'localhost'),
            expected_origin=valid_origen,
            credential_public_key=user.user_pass_pub_persona.get("public_key"),
            credential_current_sign_count=user.user_pass_pub_persona.get("sign_count"),
            require_user_verification=True,
        )
        get_logger().info(f"\n[Authentication Verification {user.name}]")
        user.user_pass_pub_persona["sign_count"] = authentication_verification.new_sign_count
    except InvalidAuthenticationResponse as e:
        get_logger().warning(f"0Error authenticating user {data.username}, {e}")
        return Result.default_user_error(info=f"Authentication failure : {e}")

    save_result = db_helper_save_user(app, asdict(user))
    if save_result.is_error():
        return save_result.to_api_result()

    key = "01#" + Code.one_way_hash(user.user_pass_sync, "CM", "get_magick_link_email")
    url = f"/web/assets/m_log_in.html?key={quote(key)}&name={user.name}"
    return Result.ok(data=url, info="Auto redirect")


@export(mod_name=Name, api=True, test=False)
async def validate_device(app: App, data: VdUSER) -> ApiResult:
    if app is None:
        app = get_app(".validate_device")

    user_result = await get_user_by_name(app, data.username)

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
        "u-key": user.uid,
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

    user_r: Result = await get_user_by_name(app, username=username)
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

    authenticate_user_get_sync_key_ = lambda *args: authenticate_user_get_sync_key(*args)
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
        "u-key": userdata.get("uid"),
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
    res = await get_user_by_name(app, username)
    if res.info.exec_code != 0:
        return Result.custom_error(data=res)
    user: User = res.get()

    data = validate_jwt(jwt_claim, user.user_pass_pub)
    # InvalidSignatureError
    if isinstance(data, str):
        return Result.custom_error(info="Invalid", data=False)

    return Result.ok(data_info='Valid JWT', data=True)


# ============================= Unit tests ===========================================

# set up
@export(mod_name=Name, test_only=True, initial=True, state=False)
def prep_test():
    app = get_app(f"{Name}.prep_test")
    app.run_any(tbef.DB.EDIT_PROGRAMMABLE, mode=DatabaseModes.LC)


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
    r = await crate_local_account(app, username, email, get_invitation(app).get())
    assert not r.is_error(), r.print(show=False)
    r = await crate_local_account(app, username, email, get_invitation(app).get())
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
    r = await crate_local_account(app, username, email, get_invitation(app).get())
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
