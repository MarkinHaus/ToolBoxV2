import base64
import datetime
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Union, List

import jwt
from pydantic import BaseModel

from toolboxv2.mods.DB.types import DatabaseModes
from toolboxv2.utils.types import ToolBoxInterfaces, ApiResult
from toolboxv2 import get_app, App, Result, tbef, ToolBox_over

from toolboxv2.utils.cryp import Code

Name = 'CloudM.AuthManager'
export = get_app(f"{Name}.Export").tb
default_export = export(mod_name=Name, test=False)
test_only = export(mod_name=Name, test_only=True)
version = '0.0.1'
instance_bios = str(uuid.uuid4())


@dataclass
class User:
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    pub_key: str = field(default="")
    email: str = field(default="")
    name: str = field(default="")
    user_pass_pub: str = field(default="")
    user_pass_pub_persona: str = field(default="")
    user_pass_pub_persona_id: str = field(default="")
    user_pass_pub_devices: List[str] = field(default_factory=[])
    user_pass_pri: str = field(default="")
    user_pass_sync: str = field(default="")
    creation_time: str = field(default_factory=lambda: time.strftime("%Y-%m-%d::%H:%M:%S", time.localtime()))
    challenge: str = field(default="")


@dataclass
class UserCreator(User):
    def __post_init__(self):
        self.user_pass_pub, self.user_pass_pri = Code.generate_asymmetric_keys()
        self.user_pass_sync = Code.generate_symmetric_key()
        self.challenge = Code.encrypt_asymmetric(str(uuid.uuid4()), self.user_pass_pub)


# app Helper functions interaction with the db

def db_helper_test_exist(app: App, username: str):
    return not app.run_any(tbef.DB.GET, query=f"USER::{username}::*", get_results=True).is_data()


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


@export(mod_name=Name, state=False, test=False)
def get_user_by_name(app: App, username: str, uid: str = '*') -> Result:
    if app is None:
        app = get_app(Name + '.get_user_by_name')

    if not db_helper_test_exist(app, username):
        return Result.default_user_error(info=f"get_user_by_name failed username'{username}'not registered")

    user_data = db_helper_get_user(app, username, uid)

    if user_data.is_error():
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

        return Result.ok(data=User(**eval(user_data[0].decode())))
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
        "dSync": Code().encrypt_asymmetric(user.user_pass_sync, user.user_pass_pub_devices[0])})


class CreateUserObject(BaseModel):
    name: str
    email: str
    pub_key: str
    invitation: str
    web_data: bool = True
    as_base64: bool = True


@export(mod_name=Name, state=True, interface=ToolBoxInterfaces.api, api=True, test=False)
def create_user(app: App, data: CreateUserObject = None, username: str = 'test-user', email: str = 'test@user.com',
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

    if db_helper_test_exist(app, username):
        return Result.default_user_error(info=f"Username '{username}' already taken", interface=ToolBoxInterfaces.remote)

    if not db_valid_invitation(app, invitation):
        return Result.default_user_error(info=f"Invalid invitation", interface=ToolBoxInterfaces.remote)

    test_bub_key = ""

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

    user = User(name=username, email=email, user_pass_pub_devices=[pub_key], pub_key=pub_key)

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
    attestationObj: str  # arrayBufferToBase64
    clientJSON: str  # arrayBufferToBase64
    pk: str  # arrayBufferToBase64
    pkAlgo: int
    authenticatorData: str  # arrayBufferToBase64
    sing: str
    rawId: str  # arrayBufferToBase64


@export(mod_name=Name, api=True, test=False)
def register_user_personal_key(app: App, data: PersonalData) -> ApiResult:
    if not db_helper_test_exist(app, data.username):
        return Result.default_user_error(info=f"Username '{data.username}' not known")

    user_result = get_user_by_name(app, data.username, from_base64(data.userId).decode())

    if user_result.is_error() and not user_result.is_data():
        return Result.default_internal_error(info="No user found", data=user_result)

    client_json = json.loads(from_base64(data.clientJSON))
    challenge = client_json.get("challenge")
    origin = client_json.get("origin")
    crossOrigin = client_json.get("crossOrigin")

    if challenge is None:
        return Result.default_user_error(info="No challenge found in data invalid date parsed", data=user_result)

    valid_origen = ["simpleCore.app", "http://localhost:5000"] + (["http://localhost:5000"] if app.debug else [])

    if origin not in valid_origen:
        return Result.default_user_error(info=f'Invalid origen:{origin} not in {valid_origen}', data=user_result)

    user: User = user_result.get()

    if not challenge == to_base64(user.challenge).decode():
        return Result.default_user_error(info="Invalid challenge returned", data=user)

    if not Code.verify_signature(signature=from_base64(data.sing), message=user.challenge, public_key_str=user.pub_key,
                                 salt_length=32):
        return Result.default_user_error(info="Verification failed Invalid signature")

    user.challenge = ""
    user.user_pass_pub_persona = data.pk
    user.user_pass_pub_persona_id = data.rawId

    # Speichern des neuen Benutzers in der Datenbank
    save_result = db_helper_save_user(app, asdict(user))
    if save_result.is_error():
        return save_result.to_api_result()

    return Result.ok(info="User registered successfully")


@export(mod_name=Name, state=True, interface=ToolBoxInterfaces.cli, test=False)
def crate_local_account(app: App, username: str, email: str = '', invitation: str = '', create=None) -> Result:
    if app is None:
        app = get_app(Name + '.crate_local_account')
    user_pri = app.config_fh.get_file_handler("Pk" + Code.one_way_hash(username, "dvp-k")[:8])
    if user_pri is not None:
        return Result.ok(info="User already registered on this device")
    pub, pri = Code.generate_asymmetric_keys()
    app.config_fh.add_to_save_file_handler("Pk" + Code.one_way_hash(username, "dvp-k")[:8], pri)
    if ToolBox_over == 'root' and invitation == '':
        invitation = db_crate_invitation(app)
    if invitation == '':
        return Result.default_user_error(info="No Invitation key provided")

    create_user_ = lambda *args: create_user(app, None, *args)
    if create is not None:
        create_user_ = create

    res = create_user_(username, email, pub, invitation)

    if res.info.exec_code != 0:
        return Result.custom_error(data=res, info="user creation failed!", exec_code=res.info.exec_code)

    return Result.ok(info="Success")


@export(mod_name=Name, state=True, interface=ToolBoxInterfaces.cli, test=False)
def local_login(app: App, username: str) -> Result:
    if app is None:
        app = get_app(Name + '.crate_local_account')
    user_pri = app.config_fh.get_file_handler("Pk" + Code.one_way_hash(username, "dvp-k")[:8])
    if user_pri is None:
        return Result.ok(info="No User registered on this device")

    signature = Code.create_signature(get_to_sing_data(app, username=username).as_result().get(), user_pri)

    res = jwt_get_claim(app, username, signature).as_result()

    if res.info.exec_code != 0:
        return Result.custom_error(data=res, info="user login failed!", exec_code=res.info.exec_code)

    return Result.ok(info="Success", data=res.get())


@export(mod_name=Name, api=True, test=False)
def get_to_sing_data(app: App, username, personal_key=False):
    if app is None:
        app = get_app(from_=Name + '.get_to_sing_data')

    user_result = get_user_by_name(app, username)

    if user_result.is_error() and not user_result.is_data():
        return Result.default_user_error(info=f"User {username} is not a valid user")

    user: User = user_result.get()

    if user.challenge == "":
        user.challenge = Code.encrypt_asymmetric(str(uuid.uuid4()), user.user_pass_pub)

    data = {'challenge': user.challenge}

    if personal_key:
        data['rowId'] = user.user_pass_pub_persona_id

    return Result.ok(data=data)


@export(mod_name=Name, state=True, interface=ToolBoxInterfaces.native, api=False, level=999, test=False)
def get_invitation(app: App) -> Result:
    if app is None:
        app = get_app(Name + '.test_invations')
    print("Test invations api")
    invitation = db_crate_invitation(app)
    return Result.ok(data=invitation)


# a sync contention between server and user

class VdUSER(BaseModel):
    username: str
    signature: str


class VpUSER(VdUSER, BaseModel):
    clientJSON: str
    authenticatorData: str


@export(mod_name=Name, api=True, test=False)
def validate_persona(app: App, data: VpUSER) -> ApiResult:
    if app is None:
        app = get_app(".validate_device")

    user_result = get_user_by_name(app, data.username)

    if user_result.is_error() or not user_result.is_data():
        return Result.default_user_error(info=f"Invalid username : {data.username}")

    jwt_claim = jwt_get_claim(app, data.username, from_base64(data.signature))

    if isinstance(jwt_claim, ApiResult):
        jwt_claim = jwt_claim.as_result()

    if jwt_claim.is_error():
        return jwt_claim.custom_error(data=jwt_claim, info="Error Processing user data")

    return Result.ok(data=jwt_claim.get())


@export(mod_name=Name, api=True, test=False)
def validate_device(app: App, data: VdUSER) -> ApiResult:
    if app is None:
        app = get_app(".validate_device")

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

        user.pub_key = divce_keys

        if valid:
            break

    if not valid:
        return Result.default_user_error(info=f"Invalid signature : {data.username}")

    user.challenge = ""
    if user.user_pass_pri == "":
        user = UserCreator(**asdict(user))
    db_helper_save_user(app, asdict(user))

    claim = {
        "pub": user.user_pass_pub,
        "u-key": user.uid,
    }

    row_jwt_claim = crate_jwt(claim, user.user_pass_pri)

    return Result.ok(data=Code.encrypt_symmetric(row_jwt_claim, user.pub_key))


@export(mod_name=Name, state=True, interface=ToolBoxInterfaces.remote, api=True, test=False)
def authenticate_user_get_sync_key(app: App, username: str, signature: str, get_user=False) -> ApiResult:
    if app is None:
        app = get_app(Name + '.authenticate_user_get_sync_key')

    user: User = get_user_by_name(app, username).get()

    if user is None:
        return Result.default_internal_error(info="User not found", exec_code=404)

    if not Code.verify_signature(signature=signature,
                                 message=user.challenge if user.challenge else username + app.id + instance_bios,
                                 public_key_str=user.user_pass_pub_persona):
        return Result.default_user_error(info="Verification failed Invalid signature")

    user = UserCreator(**asdict(user))

    db_helper_save_user(app, asdict(user))

    crypt_sync_key = Code.encrypt_asymmetric(user.user_pass_sync, user.pub_key)

    if get_user:
        Result.ok(data_info="Returned Sync Key, read only for user", data=(crypt_sync_key, asdict(user)))

    return Result.ok(data_info="Returned Sync Key, read only for user", data=crypt_sync_key)


# local user functions

@export(mod_name=Name, state=True, interface=ToolBoxInterfaces.native, test=False)
def get_user_sync_key_local(app: App, username: str, ausk=None) -> Result:
    if app is None:
        app = get_app(Name + '.get_user_sync_key')

    user_pri = app.config_fh.get_file_handler("Pk" + Code.one_way_hash(username)[:8])

    signature = Code.create_signature(username + app.id + instance_bios, user_pri)

    authenticate_user_get_sync_key_ = lambda *args: authenticate_user_get_sync_key(*args)
    if ausk is not None:
        authenticate_user_get_sync_key_ = ausk

    res = authenticate_user_get_sync_key_(app, username, signature).as_result()

    if res.info.exec_code != 0:
        return Result.custom_error(data=res, info="user get_user_sync_key failed!", exec_code=res.info.exec_code)

    sync_key = res.get()

    app.config_fh.add_to_save_file_handler("SymmetricK", sync_key)

    return Result.ok(info="Success", data=Code.decrypt_asymmetric(sync_key, user_pri))


# jwt claim

@export(mod_name=Name, state=True, interface=ToolBoxInterfaces.remote, api=True, test=False)
def jwt_get_claim(app: App, username: str, signature: str or bytes) -> ApiResult:
    if app is None:
        app = get_app(Name + '.jwt_claim_server_side_sync')

    res = authenticate_user_get_sync_key(app, username, signature, get_user=True).as_result()

    if res.info.exec_code != 0:
        return res.custom_error(data=res)

    channel_key, userdata = res.get()
    claim = {
        "pub": userdata.get("user_pass_pub"),
        "u-key": userdata.get("uid"),
    }

    row_jwt_claim = crate_jwt(claim, userdata.get("user_pass_pri"))

    return Result.ok(
        data={'claim': Code.encrypt_symmetric(row_jwt_claim, userdata.get("pub_key")), 'key': channel_key})


@export(mod_name=Name, state=True, interface=ToolBoxInterfaces.remote, api=False, test=False)
def jwt_claim_local_decrypt(app: App, username: str, crypt_sing_jwt_claim: str, aud=None) -> Result:
    if app is None:
        app = get_app(Name + '.jwt_claim_server_side_sync_local')

    user_sync_key_res = get_user_sync_key_local(app, username, ausk=aud)

    if user_sync_key_res.info.exec_code != 0:
        return Result.custom_error(data=user_sync_key_res)

    user_sync_key = user_sync_key_res.get()

    sing_jwt_claim = Code.decrypt_symmetric(crypt_sing_jwt_claim, user_sync_key)
    return jwt_check_claim_server_side(app, username, sing_jwt_claim).as_result().lazy_return('raise')


@export(mod_name=Name, state=True, interface=ToolBoxInterfaces.remote, api=True, test=False)
def jwt_check_claim_server_side(app: App, username: str, jwt_claim: str) -> ApiResult:
    res = get_user_by_name(app, username)
    if res.info.exec_code != 0:
        return Result.custom_error(data=res)
    user: User = res.get()

    data = validate_jwt(jwt_claim, user.pub_key)

    if isinstance(data, str):
        return Result.custom_error(info=data, data=False)

    return Result.ok(data_info='Valid JWT', data=True)


# ============================= Unit tests ===========================================

# set up
@export(mod_name=Name, test_only=True, initial=True, state=False)
def prep_test():
    app = get_app(f"{Name}.prep_test")
    app.run_any(tbef.DB.EDIT_PROGRAMMABLE, mode=DatabaseModes.LC)


@test_only
def crate_user_test(app: App):
    if app is None:
        app = get_app(f"{Name}.crate_user_test")
    r = crate_local_account(app, "testUser123", "test_mainqmail.com", get_invitation(app).get())
    r2 = local_login(app, "testUser123").lazy_return("crate_local_account + local_login failed")
    res = [[r.is_error() is False, "r.is_error() "], [r2.is_error() is False, "r2.is_error()"],
           [r.is_data() is True, "r.is_data() i"], [r2.is_data() is True, "r2.is_data() "],
           [r.result.data_info == "Success", "r.result.data"], [r2.result.data_info == "Success", "r2.result.dat"]]
    return res



if __name__ == '__main__':
    singen = "HhSdUTQO0wKhKpBPg34422My3kciycjZPHnFRHboVFL5pDQikPTcLGhxjqP0xWUIVh+FbpOsIywCCPjRL7IlPBxmA3Zrb0YcWq3j4I11GORF6mCVoiJX3Qq2eDFBg27lR7bmP3QfFFjEDJrFXs2pNKSaCKfZrEH9tgnm7lq9tXnzOMoeHi+kO1StnWjO5qmlvodymeyAGMnBKpomgxP2we+cznlgb9Mk7GY9Mm0S4YFj2fNX6Fra7xCxtB+ErycTRCVPrTayQ2cwqBZ0LI5jNRaqJVOPXD5U8fiMnaNlOu1bMu9FtTVnqHfszK5qy9OVyKia0xyo2UdGOfymNJSFyJ+y3JJGnhdhuS0I2za25p/K7I5xwGEL5lIBpj+JaRV5pOwDcxGhn21Q/82VsmAseT7V4TbdrSSOo6bjycRiARiwl1LgzsUhQ8CPa86x3KxzYWddvNh3aRTqCoaPWfsnJHywZXi6gPZbsCVcytOnkxOZep6TfdULstyFNjvKO9X/UmLl9Qzc5rpmjGg8gYFju05lN+IWe5qCbbgNrydn615Y30Z15G88ZwWkzs/q/n2CkzUyBLK1nrt4hPt4SXzF02/3EYTl2JhUS1je0GDfgmcVSQ+Yhr/mfHhKfS0zQBtCFm0ua9AmcqrJu7B8+InONh16WBRdHmFUUundyifj3QcZBEJRvU4gF1xq+0cDNta0XnLgpl+FVJsliKycGYkAUuohMAysznLen1bG0TuorPNHAutWOu3JBdx/BHdc2001bUIuzJ5Gg8Mnx8PUy83g2JSE4CFtNFoutlhaY5XdD4ElzPdbFgMX6N+e9ZGYIYG/4H1QBzB3qkkWbpDLVOqtNKDnTma602pHQJOtns9qkIcmwlYmJK/4USZDjAj1zSX3n+u4RttBdsvmRiehMyJ4e1A5y5Rf7r2mLtzo3EaTYvu27evKrMmWfiR0/cgTdhxhlM9gXAi8Sq///HX+vL38lcibIypScBtGV09cHSCEl/Dmm4ZsFM2ppsbUqRkcWhcI"
    msg = "0c538a042e321b968daab931ec9445dcafd7b12e1c30a338ae3a5fdd87a122188521d832fcc05741017921673f16a1491affc176f40feda81c9dc4621526748fb97af5a953df0b99be7fb0e4fa80679ebe3dbab973897d4a8baa0a209b6296757396297f0bb2cd0d3a97b892da12e1da108a7724126afcb1c3a70abb40ce3429b6333252999478731da82e0b2be355085cc5e80ecfd9fdf5d908e6fe66bf980bb9fcdf6d578dd83cf4d9f373a4bf65e9285d789e1a1b352dd12301b69f8a75ffce1679c1d767cc53bcee73da5e8cb6fb4f699dbbd429e3ffbf2e67ef57820706369e07d45f11198e528ba7ee83748ac3492060d2d611f61c4004cf8b522770d04091d277c884374970eda7c27b1bb3374f0d264f5b0e1384a83ebe29079bb274b41692f8fcec85147727c055ecfbdac51a136eb94de1f9b80f673463e6fb0aaa400db63f48a9c502cf551be39309b13fac027ccb8e69c384c0f127d56beb80d45fee60e93d69cae39dac74d2db6a0cb4a495d087ddbfb85ef3402ba19b82bc098287d9e53bf50402c7edf57e2c462c4db12ea94dff900a734e504c4c3ed90c80bc3cbc183a3e3510c86a8b123783c5b400a003fb1d8f299a897b209df8a18541550c82841867b642efd1e53435ea32bc105c7dc58d12c9d7b3eea41d63f6ab2a5e7424024fd87b35a91523d463f6016e40deccdea9ae559850b60a75e917120e6cbd209e1e00ff6f0e2ddd3c30c02a39af8ba9cf11ae9d7651afbfe9736df3d38a021297ccb11bdbe2b03a2ee603fdb5523f5f891ad482928eecba31f3e541793e87a0d8327ce7f1aa40c3f07015e712ea9af42ba51fddd42f5ce53c17ea01463f2b8722d94f928c285a35722e30694984fa79443d5ef91008205361b3936a50608436176fd23ffcece86339d79c510dcd73f7db5147d744625610ef2a776218d484c0e6173ffdaea28b6c131d62cef5d82f751c35fd7f70a02ae9c29197ac86a784b6957c2bf872aa408199b0e729e8712a30d73c745a061f914fcccd84ded5da9450ff47f8fd56ad8d803e4fbb1f0885801ed242fd30b9296a4d29dbd8556f', '"
    key = "-----BEGIN PUBLIC KEY-----\nMIIDIjANBgkqhkiG9w0BAQEFAAOCAw8AMIIDCgKCAwEAjuWAq+adexJOA4SYLNjl\nSvY0QirC/kVAX2hWcC/93qer2WT1rd4CbK0h5zRhglgBpG5zYmwllqROfJdpfQSN\nHomq7zwz+IGqs7KKGvpXgIRyDdoFcvc5IfMjaVCu8tq7OEU7w1jjIRxcInlucgsk\nayJ9qeTtzbZo/4b1wiaPwIEhn1H//30IA0WOMNO0pffXRFFY7tnbuMBmSrPPkMTY\ngycTUKIOR+8bAyozLRMOwBZzT0f1EcJWL14QDIhb7FNAjjyw3a/QK62Laq2yEoUn\nTB90Ej/AuN133mhOQfqLKo4UR39mmVTP9PvqgCy/JlwMbiV8qHaX46WjmV4qB6Lt\nTQMQFBiulVHrKJJP99Nw7aEW0PLCaRPfJt/UN+YLgisUNSZIO/Cru7uF+kbsCoa0\nqGUldDmeqQaANLASR1SW2AHNl/m/L3yS/ZBSqw6p20RjlIwWh/2Pj5q81T/Z6hgG\nuib5BEU6U3D/MPWmdhxeaO9tVgF1gz0ENXlm7B9FZrZrpuIUMJh8tX+NAL9uLRGx\nnR28WQxGnnyh9vS6VBFq0Oo1xc8GaComL+UgiMi/L3mtzCUO3yyX3eoezJOZPNm+\nuCyz+I6RGXdxr5Vewksyqh80YyCjXj8NnP+k43nAM2GJf9toM8dP7RqOrQfpn3NB\njWaQLgAVkEZmat3blA/JSof/yb+fzmmAcsMhUYWGQttxW6KF8pKKzd6EPdeZu3Mu\nMzDfcwY6c3Pqmcn5cSF+S9dOLJBhehnILYatrlHdyohCdqiaxcBL5BM+N9AaEwHQ\nFPs26HgyjRgHuqSKRlF34n+KnLInSh4ReNXD05N9RSRNU4zRDW1WAB4jTl29z87C\ns7sGb1jhfu6jVCGnqelKhhoRcH2WX9+e49/mXYtNn/LSDXO3wEXvE4f3wVC5ck1l\n3cKCEDm9Fs6Ix1E90YUW17p+g7ZT+adJtChP8RYMv2n29FlautuFxjmIsncShAi7\nCXYfjyawdkDTGme78G2DvM09fw7u6LMbwzYn9rQMk6RtAgMBAAE=\n-----END PUBLIC KEY-----\n"

    print(Code.verify_signature(singen, msg, key))
    print(Code.verify_signature(singen, to_base64(msg).decode(), key))
