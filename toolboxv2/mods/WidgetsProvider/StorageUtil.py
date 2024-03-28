import os
from typing import List, Optional

from toolboxv2 import get_app, App, Result, tbef
from fastapi import Request
from ..CloudM import User

Name = 'WidgetsProvider'
export = get_app("WidgetsProvider.Export").tb
default_export = export(mod_name=Name)
version = '0.0.1'
spec = ''

DATABASE_SPACE = os.getenv('DATABASE_SPACE') or Name


def get_db_query(user_name: str, user_id: str):
    return f"{DATABASE_SPACE}::STO::USER::{user_name}::{user_id}"


def get_all_sto_names(app, user_name="", user_id=""):
    query = f'{get_db_query(user_name=user_name, user_id=user_id)}::AllSTOS'
    return app.run_any(tbef.DB.GET, query=query, get_results=True)


def set_sto_names(app, sto_names: List[str], user_name: str = "", user_id: str = "") -> Result:
    query = f'{get_db_query(user_name=user_name, user_id=user_id)}::AllSTOS'
    return app.run_any(tbef.DB.SET, query=query, data=sto_names, get_results=True)


def add_sto_name(app, sto_name: str = "", user_name: str = "", user_id: str = "") -> Result:
    query = f'{get_db_query(user_name=user_name, user_id=user_id)}::AllSTOS'
    return app.run_any(tbef.DB.APPEND_ON_SET, query=query, data=sto_name, get_results=True)


def remove_sto_name(app, sto_name: str = "", user_name: str = "", user_id: str = "") -> Result:
    query = f'{get_db_query(user_name=user_name, user_id=user_id)}::AllSTOS'
    query_sto = f'{get_db_query(user_name=user_name, user_id=user_id)}::STO::{sto_name}'
    result: Result = app.run_any(tbef.DB.DELETE, query=query_sto, get_results=True)
    if result.is_error() or not result.is_data():
        return result

    result0: Result = app.run_any(tbef.DB.GET, query=query, get_results=True)
    if result0.is_error() or not result0.is_data():
        return result0
    sto = result0.get()

    if not isinstance(sto, list):
        print(f"Sto is not list : {type(sto)}, {sto=}")
        sto = list(sto)

    sto.remove(sto_name)
    return set_sto_names(app,
                         sto_names=sto,
                         user_name=user_name,
                         user_id=user_id).lazy_return('raise', "Saving Sto Failed")


def get_sto(app, sto_name: str = "", user_name: str = "", user_id: str = "") -> Result:
    query = f'{get_db_query(user_name=user_name, user_id=user_id)}::STO::{sto_name}'
    return app.run_any(tbef.DB.GET, query=query, get_results=True)


def set_sto(app, sto_name: str = "", user_name: str = "", user_id: str = "", data=None) -> Result:
    query = f'{get_db_query(user_name=user_name, user_id=user_id)}::STO::{sto_name}'
    return app.run_any(tbef.DB.SET, query=query, data=data, get_results=True)


def add_sto(app, sto_name: str = "", user_name: str = "", user_id: str = "") -> Result:
    result0 = add_sto_name(app=app,
                           sto_name=sto_name,
                           user_name=user_name,
                           user_id=user_id)
    if result0.is_error():
        return result0
    result1 = set_sto(app=app,
                      sto_name=sto_name,
                      user_name=user_name,
                      user_id=user_id, data={})
    if result1.is_error():
        return result1
    return Result.ok(info="Success")


@export(mod_name=Name, name='Version', version=version)
def get_version():
    return version


##### Save Sto
##### get Sto
##### remove sto Sto
##### get all Sto names

##### crate widget
##### close widget
##### remove widget

##### chair widget

##### Auto Register all widgets

def get_user_from_request(app, request):
    if app is None:
        app = get_app(from_=f"{Name}.controller")
    if request is None:
        return Result.default_internal_error("No request specified")
    if request.session['live_data'].get('spec') != spec:
        return Result.default_internal_error("Invalid Instance")
    if request.session['live_data'].get('user_name', "") == "":
        return Result.default_internal_error("Invalid User")
    username_c = request.session['live_data'].get('user_name')
    username = app.config_fh.decode_code(username_c)
    user: User = app.run_any(tbef.CLOUDM_AUTHMANAGER.GET_USER_BY_NAME, username=username)
    return Result.ok(user)


@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True, name="get_names")
def get_names(app: App = None, request: Request or None = None):
    if app is None:
        app = get_app(from_=f"{Name}.controller")

    user_result = get_user_from_request(app, request)

    if user_result.is_error() or not user_result.is_data():
        return user_result

    user: User = user_result.get()

    all_names_result = get_all_sto_names(app, user.name, user.uid)
    if all_names_result.is_error():
        return all_names_result
    if not all_names_result.is_data():
        return []
    return all_names_result


@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True, name="get_sto")
def get_sto_by_name(app: App = None, sto_name: Optional[str] = None, request: Request or None = None):
    if sto_name is None:
        return Result.default_user_error(info="No name specified")
    if app is None:
        app = get_app(from_=f"{Name}.controller")

    user_result = get_user_from_request(app, request)

    if user_result.is_error() or not user_result.is_data():
        return user_result

    user: User = user_result.get()

    sto_data_result = get_sto(app, sto_name=sto_name, user_name=user.name, user_id=user.uid)
    if sto_data_result.is_error() or not sto_data_result.is_data():
        return sto_data_result
    return sto_data_result


@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True, name="set_sto")
def set_sto_by_name(app: App = None, sto_name: Optional[str] = None, data=None, request: Request or None = None):
    if sto_name is None:
        return Result.default_user_error(info="No name specified")
    if app is None:
        app = get_app(from_=f"{Name}.controller")

    user_result = get_user_from_request(app, request)

    if user_result.is_error() or not user_result.is_data():
        return user_result

    user: User = user_result.get()

    sto_data_result = set_sto(app, sto_name=sto_name, user_name=user.name, user_id=user.uid, data=data)
    if sto_data_result.is_error() or not sto_data_result.is_data():
        return sto_data_result
    return sto_data_result


@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True, name="add_sto")
def add_sto_by_name(app: App = None, sto_name: Optional[str] = None, request: Request or None = None):
    if sto_name is None:
        return Result.default_user_error(info="No name specified")
    if app is None:
        app = get_app(from_=f"{Name}.controller")

    user_result = get_user_from_request(app, request)

    if user_result.is_error() or not user_result.is_data():
        return user_result

    user: User = user_result.get()

    sto_info_result = add_sto(app, sto_name=sto_name, user_name=user.name, user_id=user.uid)
    if sto_info_result.is_error() or not sto_info_result.is_data():
        return sto_info_result
    return sto_info_result


@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True, name="delet_sto")
def remove_sto_by_name(app: App = None, sto_name: Optional[str] = None, request: Request or None = None):
    if sto_name is None:
        return Result.default_user_error(info="No name specified")
    if app is None:
        app = get_app(from_=f"{Name}.controller")

    user_result = get_user_from_request(app, request)

    if user_result.is_error() or not user_result.is_data():
        return user_result

    user: User = user_result.get()

    sto_info_result = remove_sto_name(app, sto_name=sto_name, user_name=user.name, user_id=user.uid)
    if sto_info_result.is_error() or not sto_info_result.is_data():
        return sto_info_result
    return sto_info_result
