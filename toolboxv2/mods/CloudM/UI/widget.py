import uuid
from dataclasses import asdict
from datetime import datetime

from toolboxv2 import get_app, App, Result, tbef
from toolboxv2.mods.CloudM import User
from toolboxv2.mods.CloudM.AuthManager import db_helper_delete_user, db_helper_save_user
from toolboxv2.mods.SocketManager import get_local_ip
from fastapi import Request

Name = 'CloudM.UI.widget'
export = get_app(f"{Name}.Export").tb
default_export = export(mod_name=Name)
version = '0.0.1'
spec = ''


def load_root_widget(app, uid):
    root = f"/api/{Name}"
    all_users = app.run_any(tbef.DB.GET, query="USER::*")
    print("[all_users]:", all_users)
    if not all_users:
        all_users = [b"{'name': 'root', 'uid': uid}"]
    all_users = [eval(user) for user in all_users]
    all_user = {'name': f"system_users-root",
                'group': [{'name': f'mod-{user_.get("name", "--")}',
                           'file_path': f'./mods/CloudM/UI/assets/user_controller_template.html',
                           'kwargs': {
                               'username': user_.get('name', '--'),
                               'userId': user_.get('uid'),
                               'userLevel': user_.get('level', 0),
                               'root': root
                           }
                           } for user_ in all_users]}
    app.run_any(tbef.MINIMALHTML.ADD_COLLECTION_TO_GROUP, group_name=Name, collection=all_user)
    all_users_config = app.run_any(tbef.MINIMALHTML.GENERATE_HTML, group_name=Name,
                                   collection_name=f"system_users-root")

    root_sys = {'name': "RootSys",
                'group': [
                    {'name': 'infos_root',
                     'file_path': f'./mods/CloudM/UI/assets/system_root.html',
                     'kwargs': {
                         'UserController': app.run_any(tbef.MINIMALHTML.FUSE_TO_STRING, html_elements=all_users_config),
                     }
                     },
                ]}
    root_infos = {'name': "RootInfos",
                  'group': [
                      {'name': 'infos_root',
                       'file_path': f'./mods/CloudM/UI/assets/infos_root.html',
                       'kwargs': {
                           'systemName': app.id,
                           'systemIP': get_local_ip(),
                           'systemUptime': datetime.fromtimestamp(app.called_exit[1]).strftime('%Y-%m-%d %H:%M:%S'),
                           'timeToRestart': '-1s',
                       }
                       },
                  ]}

    app.run_any(tbef.MINIMALHTML.ADD_COLLECTION_TO_GROUP, group_name=Name, collection=root_sys)
    app.run_any(tbef.MINIMALHTML.ADD_COLLECTION_TO_GROUP, group_name=Name, collection=root_infos)


def reload_widget_main(app, user, WidgetID):
    root = f"/api/{Name}"

    widget = {'name': f"MainWidget-{user.uid}",
              'group': [
                  {'name': 'main',
                   'file_path': f'./mods/CloudM/UI/assets/main.html',
                   'kwargs': {
                       'username': user.name,
                       'root': root,
                       'WidgetID': WidgetID,
                   }
                   },
              ]}

    app.run_any(tbef.MINIMALHTML.ADD_COLLECTION_TO_GROUP, group_name=Name, collection=widget)


def reload_widget_info(app, user, WidgetID):
    root = f"/api/{Name}"
    if user.name == 'root':
        load_root_widget(app, user.uid)
    devices = {'name': f"Devices-{user.uid}",
               'group': [{'name': f'divice-{user.user_pass_pub_devices.index(d)}',
                          'template': f'<button hx-get="$root/removed?index=$index" hx-trigger="click">remove $name</button>',
                          'kwargs': {
                              'name': d[12:16],
                              'root': root,
                              'WidgetID': WidgetID,
                              'index': user.user_pass_pub_devices.index(d)
                          }
                          } for d in user.user_pass_pub_devices]}
    app.run_any(tbef.MINIMALHTML.ADD_COLLECTION_TO_GROUP, group_name=Name, collection=devices)
    html_devices = app.run_any(tbef.MINIMALHTML.GENERATE_HTML, group_name=Name, collection_name=f"Devices-{user.uid}")

    infos = {'name': f"infosTab-{user.uid}",
             'group': [
                 {'name': 'infos',
                  'file_path': f'./mods/CloudM/UI/assets/infos.html',
                  'kwargs': {
                      'userName': user.name,
                      'userEmail': user.email,
                      'userLevel': user.level,
                      'addUserPersona': '',
                      'root': root,
                      'WidgetID': WidgetID,
                      "devices": app.run_any(tbef.MINIMALHTML.FUSE_TO_STRING, html_elements=html_devices),
                      'rootInfo':
                          app.run_any(tbef.MINIMALHTML.GENERATE_HTML, group_name=Name, collection_name="RootInfos")[
                              0]['html_element'] if user.name == 'root' else ''
                  }
                  }, ]}

    app.run_any(tbef.MINIMALHTML.ADD_COLLECTION_TO_GROUP, group_name=Name, collection=infos)


def reload_widget_mods(app, user, WidgetID):
    root = f"/api/{Name}"
    user_instance = app.run_any(tbef.CLOUDM_USERINSTANCES.GET_USER_INSTANCE, uid=user.uid)

    mods_a = {'name': f"mods-{user.uid}",
              'group': [{'name': f'mod-{mod_name}',
                         'file_path': f'./mods/CloudM/UI/assets/a_mod.html',
                         'kwargs': {
                             'modId': mod_name,
                             'modName': mod_name,
                             'WidgetID': WidgetID,
                             'root': root
                         }
                         } for mod_name in user_instance.get('save', {}).get('mods', [])]}
    app.run_any(tbef.MINIMALHTML.ADD_COLLECTION_TO_GROUP, group_name=Name, collection=mods_a)
    html_mods = app.run_any(tbef.MINIMALHTML.GENERATE_HTML, group_name=Name, collection_name=f"mods-{user.uid}")

    mods_b = {'name': f"mods_app-{user.uid}",
              'group': [{'name': f'mod-{mod_name}',
                         'file_path': f'./mods/CloudM/UI/assets/b_mod.html',
                         'kwargs': {
                             'modId': mod_name,
                             'modName': mod_name,
                             'WidgetID': WidgetID,
                             'modInfos': '<h6>dose ...</h6>',
                             'root': root
                         }
                         } for mod_name in app.get_all_mods()]}
    app.run_any(tbef.MINIMALHTML.ADD_COLLECTION_TO_GROUP, group_name=Name, collection=mods_b)
    html_mods_a = app.run_any(tbef.MINIMALHTML.GENERATE_HTML, group_name=Name, collection_name=f"mods_app-{user.uid}")

    mods_ = {'name': f"modTab-{user.uid}",
             'group': [
                 {'name': 'mods',
                  'file_path': f'./mods/CloudM/UI/assets/mods.html',
                  'kwargs': {
                      "modsList": app.run_any(tbef.MINIMALHTML.FUSE_TO_STRING, html_elements=html_mods),
                      "AvalabelModsList": app.run_any(tbef.MINIMALHTML.FUSE_TO_STRING, html_elements=html_mods_a),
                  }
                  },
             ]}
    app.run_any(tbef.MINIMALHTML.ADD_COLLECTION_TO_GROUP, group_name=Name, collection=mods_)


def reload_widget_system(app, user, WidgetID):
    root = f"/api/{Name}"
    if user.name == 'root':
        load_root_widget(app, user.uid)

    system_person = {'name': f"sysTab-{user.uid}",
                     'group': [
                         {'name': 'mods',
                          'file_path': f'./mods/CloudM/UI/assets/system.html',
                          'kwargs': {
                              'root': root,
                              'WidgetID': WidgetID,
                              'rootSys': app.run_any(tbef.MINIMALHTML.GENERATE_HTML, group_name=Name,
                                                     collection_name="RootSys")[0][
                                  'html_element'] if user.name == 'root' else ''
                          }
                          },
                     ]}

    app.run_any(tbef.MINIMALHTML.ADD_COLLECTION_TO_GROUP, group_name=Name, collection=system_person)


def load_widget(app, display_name="Cud be ur name", WidgetID=str(uuid.uuid4())[:4]):
    if display_name != "Cud be ur name":
        user = app.run_any(tbef.CLOUDM_AUTHMANAGER.GET_USER_BY_NAME, username=display_name)
    else:
        user = User()

    app.run_any(tbef.MINIMALHTML.ADD_GROUP, command=Name)

    reload_widget_main(app, user, WidgetID)
    reload_widget_info(app, user, WidgetID)
    reload_widget_system(app, user, WidgetID)
    reload_widget_mods(app, user, WidgetID)

    html_widget = app.run_any(tbef.MINIMALHTML.GENERATE_HTML, group_name=Name, collection_name=f"MainWidget-{user.uid}")
    return html_widget[0]['html_element']


def get_user_from_request(app, request):
    name = request.session.get('live_data', {}).get('user_name', "Cud be ur name")
    if name != "Cud be ur name":
        user = app.run_any(tbef.CLOUDM_AUTHMANAGER.GET_USER_BY_NAME, username=app.config_fh.decode_code(name))
    else:
        user = User()
    return user


@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True, row=True)
def removed(app, index, request: Request or None = None):
    if request is None:
        return Result.default_internal_error("No request specified")
    user: User = get_user_from_request(app, request=request)
    if not user:
        return f"<h2>Invalid User</h2>"
    if isinstance(index, str):
        index = int(index)
    user.user_pass_pub_devices.pop(index)
    db_helper_save_user(app, asdict(user))
    return f""


@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True, row=True)
def danger(app, request: Request or None = None):
    if request is None:
        return Result.default_internal_error("No request specified")
    user = get_user_from_request(app, request=request)
    if not user:
        return f"<h2>Invalid User</h2>"
    WidgetID = str(uuid.uuid4())[:4]
    reload_widget_system(app, user, WidgetID)
    html_widget = app.run_any(tbef.MINIMALHTML.GENERATE_HTML, group_name=Name, collection_name=f"sysTab-{user.uid}")
    return html_widget[0]['html_element']


# danger functions

@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True, row=True)
def stop(app, request: Request or None = None):
    if request is None:
        return Result.default_internal_error("No request specified")
    user = get_user_from_request(app, request=request)
    if not user:
        return f"<h2>Invalid User</h2>"
    app.run_any(tbef.CLOUDM_USERINSTANCES.CLOSE_USER_INSTANCE, uid=user.uid)
    if user.name == 'root':
        app.exit()
        exit()
    return f"<h2>Stop system {user.name=}</h2>"


@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True, row=True)
def reset(app, request: Request or None = None):
    if request is None:
        return Result.default_internal_error("No request specified")
    user = get_user_from_request(app, request=request)
    if not user:
        return f"<h2>Invalid User</h2>"
    app.run_any(tbef.CLOUDM_USERINSTANCES.DELETE_USER_INSTANCE, uid=user.uid)
    db_helper_delete_user(app, user.name, user.uid, matching=True)
    return f"<h2>account gelöscht {user.name=}</h2>"


@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True, row=True)
def link(app, request: Request or None = None):
    if request is None:
        return Result.default_internal_error("No request specified")
    user = get_user_from_request(app, request=request)
    if not user:
        return f"<h2>Invalid User</h2>"
    return f"<h2>{app.run_any(tbef.CLOUDM.CREATE_MAGIC_LOG_IN, username=user.name)}</h2>"


@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True, row=True)
def info(app, request: Request or None = None):
    if request is None:
        return Result.default_internal_error("No request specified")
    user = get_user_from_request(app, request=request)
    if not user:
        return f"<h2>Invalid User</h2>"

    WidgetID = str(uuid.uuid4())[:4]
    reload_widget_info(app, user, WidgetID)
    html_widget = app.run_any(tbef.MINIMALHTML.GENERATE_HTML, group_name=Name, collection_name=f"infosTab-{user.uid}")
    return html_widget[0]['html_element']


@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True, row=True)
def deleteUser(app, user: str, request: Request or None = None):
    if request is None:
        return Result.default_internal_error("No request specified")
    user_ob = get_user_from_request(app, request=request)
    if user_ob.name != 'root':
        return f"<h2>Invalid User</h2>"
    user_ed = app.run_any(tbef.CLOUDM_AUTHMANAGER.GET_USER_BY_NAME, username=user)
    app.run_any(tbef.CLOUDM_USERINSTANCES.DELETE_USER_INSTANCE, uid=user_ed.uid)
    db_helper_delete_user(app, user_ed.name, user_ed.uid, matching=True)
    return f"<h2>account gelöscht {user_ed.name=}</h2>"


@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True, row=True)
def sendMagicLink(app, user: str, request: Request or None = None):
    if request is None:
        return Result.default_internal_error("No request specified")
    user_ob = get_user_from_request(app, request=request)
    if user_ob.name != 'root':
        return f"<h2>Invalid User</h2>"
    link = app.run_any(tbef.CLOUDM.CREATE_MAGIC_LOG_IN, username=user)
    user_ed = app.run_any(tbef.CLOUDM_AUTHMANAGER.GET_USER_BY_NAME, username=user)
    msg = app.run_any(tbef.EMAIL_WAITING_LIST.CRATE_MAGICK_LICK_DEVICE_EMAIL,
                      user_email=user_ed.email,
                      user_name=user_ed.name,
                      link_id=link, nl=''
                      )
    app.run_any(tbef.EMAIL_WAITING_LIST.SEND_EMAIL, data=msg)


@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True, row=True)
def setUserLevel(app, user: str, data: dict, request: Request or None = None):
    if request is None:
        return Result.default_internal_error("No request specified")
    user_ob = get_user_from_request(app, request=request)
    if user_ob.name != 'root':
        return f"<h2>Invalid User</h2>"
    userLevel = data.get('userLevel', 0)
    user_ed = app.run_any(tbef.CLOUDM_AUTHMANAGER.GET_USER_BY_NAME, username=user)
    if isinstance(userLevel, str):
        userLevel = int(userLevel)
    user_ed.level = userLevel
    db_helper_save_user(app, asdict(user_ed))
    return f"<h2>Level set to {userLevel}</h2>"


@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True, row=True)
def mods(app, request: Request or None = None):
    if request is None:
        return Result.default_internal_error("No request specified")
    user = get_user_from_request(app, request=request)
    if not user:
        return f"<h2>Invalid User</h2>"

    WidgetID = str(uuid.uuid4())[:4]
    reload_widget_mods(app, user, WidgetID)
    html_widget = app.run_any(tbef.MINIMALHTML.GENERATE_HTML, group_name=Name, collection_name=f"modTab-{user.uid}")
    return html_widget[0]['html_element']


@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True, row=True)
def addMod(app, modId: str, request: Request or None = None):
    if request is None:
        return Result.default_internal_error("No request specified")
    user = get_user_from_request(app, request=request)
    if not user:
        return f"<h2>Invalid User</h2>"
    user_instance = app.run_any(tbef.CLOUDM_USERINSTANCES.GET_USER_INSTANCE, uid=user.uid)
    if modId in user_instance["save"]["mods"]:
        return f"<h2>{modId} is already active</h2>"
    user_instance["save"]["mods"].append(modId)
    user_instance = app.run_any(tbef.CLOUDM_USERINSTANCES.SAVE_USER_INSTANCES, instance=user_instance)
    return f"<h2>{modId} Added</h2>"


@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True, row=True)
def remove(app, modId: str, request: Request or None = None):
    if request is None:
        return Result.default_internal_error("No request specified")
    user = get_user_from_request(app, request=request)
    if not user:
        return f"<h2>Invalid User</h2>"
    user_instance = app.run_any(tbef.CLOUDM_USERINSTANCES.GET_USER_INSTANCE, uid=user.uid)
    if modId not in user_instance["save"]["mods"]:
        return f"<h2>{modId} is already active</h2>"
    user_instance["save"]["mods"].remove(modId)
    user_instance = app.run_any(tbef.CLOUDM_USERINSTANCES.SAVE_USER_INSTANCES, instance=user_instance)
    return f"<h2>{modId} Remove</h2>"


@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True, name="get_widget")
def get_widget(app: App = None, request: Request or None = None):
    if app is None:
        app = get_app(from_=f"{Name}.get_widget")

    if request is None:
        return Result.default_internal_error("No request specified")

    username_c: str = request.session.get('live_data', {}).get('user_name', "Cud be ur name")
    if username_c != "Cud be ur name":
        username = app.config_fh.decode_code(username_c)
    else:
        username = username_c

    widget = load_widget(app, username)

    return widget
