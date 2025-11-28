import os
from dataclasses import asdict
from urllib.parse import quote

from toolboxv2 import Code, App, Result, get_logger, TBEF

from toolboxv2 import TBEF, App, Result, get_app, get_logger
from toolboxv2.utils.security.cryp import Code
from toolboxv2.utils.system.types import ToolBoxInterfaces

from .email_services import send_magic_link_email
from .types import User, UserCreator

version = "0.0.2"
Name = 'CloudM.AuthManager'
export = get_app(f"{Name}.Export").tb
default_export = export(mod_name=Name, test=False)
test_only = export(mod_name=Name, test_only=True)
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
    print("SAVE USER", user_data)
    return app.run_any(TBEF.DB.SET, query=f"USER::{user_data['name']}::{user_data['uid']}",
                       data=user_data,
                       get_results=True)


def db_helper_get_user(app: App, username: str, uid: str = '*'):
    data =  app.run_any(TBEF.DB.GET, query=f"USER::{username}::{uid}",
                       get_results=True)
    return data


def db_helper_delete_user(app: App, username: str, uid: str, matching=False):
    return app.run_any(TBEF.DB.DELETE, query=f"USER::{username}::{uid}", matching=matching,
                       get_results=True)


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


    if '*' in uid:
        user_data = user_data.get(list(user_data.get().keys())[0])
    else:
        user_data = user_data.get()

    if isinstance(user_data, str):
        return Result.ok(data=User(**eval(user_data)))
    else:
        return Result.default_internal_error(info="get_user_by_name failed no User data found", exec_code=2351)
