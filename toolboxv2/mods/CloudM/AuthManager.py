import datetime
import time
import uuid
from dataclasses import dataclass, field, asdict

import jwt

from toolboxv2.utils.types import ToolBoxInterfaces
from toolboxv2 import get_app, App, Result, tbef, ToolBox_over

from toolboxv2.utils.cryp import Code

Name = 'CloudM.AuthManager'
export = get_app(f"{Name}.Export").tb
default_export = export(mod_name=Name)
test_only = export(mod_name=Name, test_only=True)
version = '0.0.1'


@dataclass
class User:
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    pub_key: str = field(default="")
    email: str = field(default="")
    name: str = field(default="")
    user_pass_pub: str = field(init=False)
    user_pass_pri: str = field(init=False)
    user_pass_sync: str = field(init=False)
    creation_time: str = field(default_factory=lambda: time.strftime("%Y-%m-%d::%H:%M:%S", time.localtime()))
    User_key: str = field(init=False)


@dataclass
class UserCreator(User):
    def __post_init__(self):
        self.user_pass_pub, self.user_pass_pri = Code.generate_asymmetric_keys()
        self.user_pass_sync = Code.generate_symmetric_key()
        self.User_key = Code.encrypt_asymmetric(self.uid, self.user_pass_pub)


# app Helper functions interaction with the db

def db_helper_test_exist(app: App, username: str):
    return -2 != app.run_any(tbef.DB.GET, query=f"USER::{username}::*", get_results=True).info.exec_code


def db_delete_invitation(app: App, invitation: str):
    return app.run_any(tbef.DB.DELETE, query=f"invitation::{invitation}", get_results=True)


def db_valid_invitation(app: App, invitation: str):
    inv_key = app.run_any(tbef.DB.GET, query=f"invitation::{invitation}")
    if inv_key is None:
        return False
    inv_key = inv_key[0]
    return Code.decrypt_symmetric(inv_key, invitation) == invitation


def db_crate_invitation(app: App):
    invitation = Code.generate_symmetric_key()
    inv_key = Code.encrypt_symmetric(invitation, invitation)
    res = app.run_any(tbef.DB.SET, query=f"invitation::{invitation}", data=inv_key, get_results=True)
    return invitation


def db_helper_save_user(app: App, user_data: dict):
    return app.run_any(tbef.DB.SET, query=f"USER::{user_data['name']}::{user_data['uid']}",
                       data=user_data,
                       get_results=True)


def db_helper_save_user_u_key(app: App, user_data: dict):
    db_helper_delete_user_i_clear(app, user_data['name'], user_data['User_key'])
    return app.run_any(tbef.DB.SET, query=f"USER:I::{user_data['name']}::{user_data['User_key']}",
                       data=user_data,
                       get_results=True)


def db_helper_get_user(app: App, username: str, uid: str = '*'):
    return app.run_any(tbef.DB.GET, query=f"USER::{username}::{uid}",
                       get_results=True)


def db_helper_delete_user(app: App, username: str, uid: str):
    return app.run_any(tbef.DB.DELETE, query=f"USER::{username}::{uid}",
                       get_results=True)


def db_helper_delete_user_i_clear(app: App, username: str, uid: str):
    return app.run_any(tbef.DB.DELETE, query=f"USER:I::{username}::{uid}*",
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


def reade_jwt(jwt_key: str, public_key: str) -> dict or str:
    if not jwt_key:
        return "No JWT Key provided"

    try:
        token = jwt.decode(jwt_key,
                           public_key,
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


@export(mod_name=Name, state=False)
def get_user_by_name(app: App, username: str, uid: str = '*') -> Result:
    if app is None:
        app = get_app(Name + '.get_user_by_name')

    if not db_helper_test_exist(app, username):
        return Result.default_user_error(info=f"Authentication failed username'{username}'not registered")

    user_data = db_helper_get_user(app, username, uid)

    if len(user_data) == 0:
        return Result.default_internal_error(info="Authentication failed no User data found")

    if len(user_data) > 1:
        pass

    if user_data[0].get("pub_key") == "" or user_data[0].get("pub_key") is None:
        return Result.default_internal_error(info="Authentication failed no Pub key found")

    return Result.ok(data=User(**user_data[0]))


@export(mod_name=Name, state=False, interface=ToolBoxInterfaces.api, api=True)
def create_user(app: App, name: str = 'test-user', email: str = 'test@user.com', pub_key: str = '',
                invitation: str = '') -> Result:
    if app is None:
        app = get_app(Name + '.crate_user')

    if db_helper_test_exist(app, name):
        return Result.default_user_error(info=f"Username '{name}' already taken")

    if not db_valid_invitation(app, invitation):
        return Result.default_user_error(info=f"Invalid invitation")

    user = User(name=name, email=email, pub_key=pub_key)
    user_object = asdict(user)

    result_s = db_helper_save_user(app, user_object).print()

    db_delete_invitation(app, invitation)

    return Result.ok(info=f"User created successfully: {name}",
                     data=Code().encrypt_asymmetric(str(user.user_pass_sync), pub_key))


@export(mod_name=Name, state=False)
def crate_local_account(app: App, name: str, email: str = '', invitation: str = '', create=None) -> Result:
    if app is None:
        app = get_app(Name + '.crate_local_account')
    user_pri = app.config_fh.get_file_handler("Privat-key")
    if user_pri is not None:
        return Result.ok(info="User already registered on this device")
    pub, pri = Code.generate_asymmetric_keys()
    app.config_fh.add_to_save_file_handler("Privat-key", pri)
    if ToolBox_over == 'root' and invitation == '':
        invitation = db_crate_invitation(app)
    if invitation == '':
        return Result.default_user_error(info="No Invitation key provided")

    create_user_ = lambda *args: create_user(*args)
    if create is not None:
        create_user_ = create

    res = create_user_(app, name, email, pub, invitation)

    if res.info.exec_code != 0:
        return Result.custom_error(data=res, info="user creation failed!", exec_code=res.info.exec_code)

    sync_key = res.get()
    app.config_fh.add_to_save_file_handler("SymmetricK", sync_key)

    return Result.ok(info="Success", data=Code.decrypt_asymmetric(sync_key, pri))


@test_only
def test_invations(app: App):
    if app is None:
        app = get_app(Name + '.test_invations')
    invitation = db_crate_invitation(app)
    print("Invitation", invitation)
    app.run_any(tbef.DB.GET, query=f"invitation::*", get_results=True)
    res = db_valid_invitation(app, invitation)
    print(res, 'invitation test')


# a sync contention between server and user


@export(mod_name=Name, state=False)
def authenticate_user_get_sync_key(app: App, username: str, signature: str, get_user=False) -> Result:
    if app is None:
        app = get_app(Name + '.authenticate_user_get_sync_key')

    user = get_user_by_name(app, username).get()

    if user is None:
        return Result.default_internal_error(info="User not found", exec_code=404)

    if not Code.verify_signature(signature=signature, message=username, public_key_str=user.pub_key):
        return Result.default_user_error(info="Verification failed Invalid signature")

    user.user_pass_sync = Code.generate_symmetric_key()

    db_helper_save_user(app, asdict(user))

    crypt_sync_key = Code.encrypt_asymmetric(user.user_pass_sync, user.pub_key)

    if get_user:
        Result.ok(data_info="Returned Sync Key, read only for user", data=(crypt_sync_key, asdict(user)))

    return Result.ok(data_info="Returned Sync Key, read only for user", data=crypt_sync_key)


@export(mod_name=Name, state=False)
def get_user_sync_key(app: App, username: str, ausk=None) -> Result:
    if app is None:
        app = get_app(Name + '.get_user_sync_key')

    user_pri = app.config_fh.get_file_handler("Privat-key")

    signature = Code.create_signature(username, user_pri)

    authenticate_user_get_sync_key_ = lambda *args: authenticate_user_get_sync_key(*args)
    if ausk is not None:
        authenticate_user_get_sync_key_ = ausk

    res = authenticate_user_get_sync_key_(app, username, signature)

    if res.info.exec_code != 0:
        return Result.custom_error(data=res, info="user get_user_sync_key failed!", exec_code=res.info.exec_code)

    sync_key = res.get()

    app.config_fh.add_to_save_file_handler("SymmetricK", sync_key)

    return Result.ok(info="Success", data=Code.decrypt_asymmetric(sync_key, user_pri))


# jwt claim

@export(mod_name=Name, state=False)
def jwt_claim_server_side_sync(app: App, username: str, signature: str):
    if app is None:
        app = get_app(Name + '.jwt_claim_server_side_sync')

    res = authenticate_user_get_sync_key(app, username, signature, get_user=True)

    if res.info.exec_code != 0:
        return res.custom_error(data=res)

    channel_key, userdata = res.get()

    user = UserCreator(**userdata)
    res1 = db_helper_save_user_u_key(app, asdict(user))
    if res1.info.exec_code != 0:
        return res.custom_error(data=res)

    claim = {
        "pub": user.user_pass_pub,
        "u-key": user.User_key,
    }

    row_jwt_claim = crate_jwt(claim, user.user_pass_pri)

    return Result.ok(data=Code.encrypt_symmetric(row_jwt_claim, channel_key))


@export(mod_name=Name, state=False)
def jwt_claim_server_side_sync_local(app: App, username: str, crypt_jwt_claim: str, aud=None) -> Result:
    if app is None:
        app = get_app(Name + '.jwt_claim_server_side_sync_local')

    user_sync_key_res = get_user_sync_key(app, username, ausk=aud)

    if user_sync_key_res.info.exec_code != 0:
        return Result.custom_error(data=user_sync_key_res)

    user_sync_key = user_sync_key_res.get()

    data = validate_jwt(crypt_jwt_claim, user_sync_key)

    if isinstance(data, str):
        return Result.default_internal_error(info=data)

    res = get_user_by_name(app, username, data.get('User_key'))

    if res.info.exec_code != 0:
        return res.custom_error(data=res)
    # TODO: ab hier stikts
    channel_key, userdata = res.get()

    user = UserCreator(**userdata)
    res1 = db_helper_save_user_u_key(app, asdict(user))
    if res1.info.exec_code != 0:
        return res.custom_error(data=res)

    claim = {
        "pub": user.user_pass_pub,
        "u-key": user.User_key,
    }

    row_jwt_claim = crate_jwt(claim, user.user_pass_pri)

    return Result.ok(data=Code.encrypt_symmetric(row_jwt_claim, channel_key))
    # TODO: ende


# 3Therd Party authe base fuction :

@export(mod_name=Name, interface=ToolBoxInterfaces.api, api=True, test=False, state=False)
def get_user_server_session_pubkey_sync(app: App, username: str, user_id: id) -> Result:
    if app is None:
        return Result.default_internal_error(info="Pleas register an app wit the fuction name use get_app")

    res = get_user_by_name(app, username, user_id)

    if res.info.exec_code != 0:
        return Result.custom_error(data=res)

    return Result.ok(data=res.get().user_pass_sync)


@export(mod_name=Name, interface=ToolBoxInterfaces.api, api=True, test=False, state=False)
def get_user_server_session_pubkey_async(app: App, username: str, user_id: id) -> Result:
    if app is None:
        return Result.default_internal_error(info="Pleas register an app wit the fuction name use get_app")

    res = get_user_by_name(app, username, user_id)

    if res.info.exec_code != 0:
        return Result.custom_error(data=res)

    return Result.ok(data=res.get().user_pass_pub)
