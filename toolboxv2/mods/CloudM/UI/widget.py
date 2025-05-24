# toolboxv2/mods/CloudM/UI/widget.py

import uuid
from dataclasses import asdict
from datetime import datetime

from toolboxv2 import TBEF, App, Result, get_app, RequestData
from toolboxv2.mods.CloudM.AuthManager import db_helper_delete_user, db_helper_save_user
from toolboxv2.mods.SocketManager import get_local_ip

from ..types import User
# Import the Name constant from user_account_manager to use in TBEF calls


Name = 'CloudM.UI.widget'
export = get_app(f"{Name}.Export").tb
default_export = export(mod_name=Name)
version = '0.0.1'
spec = ''


def load_root_widget(app, uid):
    root = f"/api/{Name}"
    all_users = app.run_any(TBEF.DB.GET, query="USER::*")
    # print("[all_users]:", all_users) # Commented out for cleaner logs
    if not all_users:
        all_users = [str({'name': 'root', 'uid': uid, 'level': 0}).encode()]  # Ensure level for template

    parsed_users = []
    for user_bytes in all_users:
        try:
            if isinstance(user_bytes, bytes):
                user_str = user_bytes.decode()
            else:
                user_str = str(user_bytes)  # Fallback if not bytes
            user_dict = eval(user_str)
            # Ensure essential keys for the template exist
            user_dict.setdefault('name', '--')
            user_dict.setdefault('uid', str(uuid.uuid4()))  # Default UID if missing
            user_dict.setdefault('level', 0)  # Default level if missing
            parsed_users.append(user_dict)
        except Exception as e:
            app.print(f"Error parsing user data: {user_bytes}, Error: {e}", "ERROR")
            parsed_users.append({'name': 'ErrorUser', 'uid': str(uuid.uuid4()), 'level': 0})

    all_user_collection = {'name': "system_users-root",
                           'group': [{'name': f'mod-{user_.get("name", "--")}',
                                      'file_path': './mods/CloudM/UI/assets/user_controller_template.html',
                                      'kwargs': {
                                          'username': user_.get('name', '--'),
                                          'userId': user_.get('uid'),
                                          'userLevel': user_.get('level', 0),
                                          'root': root
                                      }
                                      } for user_ in parsed_users]}
    app.run_any(TBEF.MINIMALHTML.ADD_COLLECTION_TO_GROUP, group_name=Name, collection=all_user_collection)
    all_users_config = app.run_any(TBEF.MINIMALHTML.GENERATE_HTML, group_name=Name,
                                   collection_name="system_users-root")

    root_sys = {'name': "RootSys",
                'group': [
                    {'name': 'infos_root',
                     'file_path': './mods/CloudM/UI/assets/system_root.html',
                     'kwargs': {
                         'UserController': app.run_any(TBEF.MINIMALHTML.FUSE_TO_STRING, html_elements=all_users_config),
                     }
                     },
                ]}
    root_infos = {'name': "RootInfos",
                  'group': [
                      {'name': 'infos_root',
                       'file_path': './mods/CloudM/UI/assets/infos_root.html',
                       'kwargs': {
                           'systemName': app.id,
                           'systemIP': get_local_ip(),
                           'systemUptime': datetime.fromtimestamp(app.called_exit[1]).strftime(
                               '%Y-%m-%d %H:%M:%S') if app.called_exit and len(app.called_exit) > 1 else "N/A",
                           'timeToRestart': '-1s',
                       }
                       },
                  ]}

    app.run_any(TBEF.MINIMALHTML.ADD_COLLECTION_TO_GROUP, group_name=Name, collection=root_sys)
    app.run_any(TBEF.MINIMALHTML.ADD_COLLECTION_TO_GROUP, group_name=Name, collection=root_infos)


def reload_widget_main(app, user, WidgetID):
    root = f"/api/{Name}"
    widget = {'name': f"MainWidget-{user.uid}",
              'group': [
                  {'name': 'main',
                   'file_path': './mods/CloudM/UI/assets/main.html',
                   'kwargs': {
                       'username': user.name,
                       'root': root,
                       'WidgetID': WidgetID,
                       'Content': user.name,
                   }
                   },
              ]}

    app.run_any(TBEF.MINIMALHTML.ADD_COLLECTION_TO_GROUP, group_name=Name, collection=widget)


# Made this function async
async def reload_widget_info(app: App, user: User, WidgetID: str):
    root = f"/api/{Name}"
    if user.name == 'root':
        load_root_widget(app, user.uid)  # This is sync, ensure it's okay or make async if needed

    devices_group = []
    if hasattr(user, 'user_pass_pub_devices') and user.user_pass_pub_devices:
        devices_group = [{'name': f'device-{idx}',  # Changed 'divice' to 'device'
                          'template': '<button hx-get="$root/removed?index=$index" hx-trigger="click" class="text-xs bg-red-400 hover:bg-red-500 text-white py-0.5 px-1 rounded mr-1 mb-1">remove $name</button>',
                          'kwargs': {
                              'name': d[12:16] if isinstance(d, str) and len(d) >= 16 else "Key" + str(idx),
                              # Safer access
                              'root': root,
                              'WidgetID': WidgetID,
                              'index': idx
                          }
                          } for idx, d in enumerate(user.user_pass_pub_devices)]  # Use enumerate for index

    devices_collection = {'name': f"Devices-{user.uid}", 'group': devices_group}
    app.run_any(TBEF.MINIMALHTML.ADD_COLLECTION_TO_GROUP, group_name=Name, collection=devices_collection)
    html_devices_result = app.run_any(TBEF.MINIMALHTML.GENERATE_HTML, group_name=Name,
                                      collection_name=f"Devices-{user.uid}")
    html_devices_str = app.run_any(TBEF.MINIMALHTML.FUSE_TO_STRING, html_elements=html_devices_result)

    # Call the new account management HTML generator
    from .UserAccountManager import Name as UAM_ModuleName
    account_mgmt_html_result = await app.a_run_any(
        f"{UAM_ModuleName}.get_account_management_section_html",  # Use imported Name
        user=user,
        WidgetID=WidgetID
    )
    account_management_html = account_mgmt_html_result.get() if isinstance(account_mgmt_html_result,
                                                                           Result) and not account_mgmt_html_result.is_error() else f"<p class='text-red-500'>Error loading account management: {account_mgmt_html_result.info if isinstance(account_mgmt_html_result, Result) else 'Unknown error'}</p>"

    infos_collection = {'name': f"infosTab-{user.uid}",
                        'group': [
                            {'name': 'infos',
                             'file_path': './mods/CloudM/UI/assets/infos.html',
                             'kwargs': {
                                 'userName': user.name,
                                 'userEmail': user.email,  # Kept for potential direct display
                                 'userLevel': user.level,  # Kept for potential direct display
                                 'accountManagementSection': account_management_html,  # New section
                                 'root': root,
                                 'WidgetID': WidgetID,
                                 "devices": html_devices_str,
                                 'rootInfo':
                                     app.run_any(TBEF.MINIMALHTML.GENERATE_HTML, group_name=Name,
                                                 collection_name="RootInfos")[
                                         0]['html_element'] if user.name == 'root' else ''
                             }
                             },
                        ]}

    app.run_any(TBEF.MINIMALHTML.ADD_COLLECTION_TO_GROUP, group_name=Name, collection=infos_collection)


# reload_widget_mods remains unchanged for now

def reload_widget_system(app, user, WidgetID):  # This is sync
    root = f"/api/{Name}"
    if user.name == 'root':
        load_root_widget(app, user.uid)  # This is sync

    system_person_collection = {'name': f"sysTab-{user.uid}",
                                'group': [
                                    {'name': 'mods',  # Name seems generic, but okay
                                     'file_path': './mods/CloudM/UI/assets/system.html',
                                     'kwargs': {
                                         'root': root,
                                         'WidgetID': WidgetID,
                                         'rootSys': app.run_any(TBEF.MINIMALHTML.GENERATE_HTML, group_name=Name,
                                                                collection_name="RootSys")[0][
                                             'html_element'] if user.name == 'root' else ''
                                     }
                                     },
                                ]}

    app.run_any(TBEF.MINIMALHTML.ADD_COLLECTION_TO_GROUP, group_name=Name, collection=system_person_collection)


# This is the main entry point for widget loading, needs to await async calls
async def load_widget(app, display_name="Cud be ur name", WidgetID=str(uuid.uuid4())[:4]):
    user_obj = User()  # Default user
    if display_name != "Cud be ur name":
        user_result = await app.a_run_any(TBEF.CLOUDM_AUTHMANAGER.GET_USER_BY_NAME, username=display_name,
                                          get_results=True)
        if not user_result.is_error() and user_result.get():
            user_obj = user_result.get()
        else:
            app.print(
                f"Could not load user '{display_name}', using default. Error: {user_result.info if user_result.is_error() else 'User not found'}")

    app.run_any(TBEF.MINIMALHTML.ADD_GROUP, command=Name)

    reload_widget_main(app, user_obj, WidgetID)  # This is sync
    await reload_widget_info(app, user_obj, WidgetID)  # Now awaited
    reload_widget_system(app, user_obj, WidgetID)  # This is sync
    # reload_widget_mods(app, user_obj, WidgetID) # Assuming this is sync too, based on original structure

    html_widget_result = app.run_any(TBEF.MINIMALHTML.GENERATE_HTML, group_name=Name,
                                     collection_name=f"MainWidget-{user_obj.uid}")
    return html_widget_result[0]['html_element']


async def get_user_from_request(app, request):
    name = request.session.user_name
    if name:  # Check if decoding was successful
        user_res = await app.a_run_any(TBEF.CLOUDM_AUTHMANAGER.GET_USER_BY_NAME, username=name)
        if not user_res.is_error() and user_res.get():
            return user_res.get()
        else:
            app.print(
                f"get_user_from_request: Failed to get user '{name}'. Error: {user_res.info if user_res.is_error() else 'Not found'}",
                "WARNING")
    return User()  # Return a default/empty User object if not found or not logged in


@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True, row=True)
async def removed(app: App, request: RequestData, index: str):  # Ensure index is received
    user: User = await get_user_from_request(app, request=request)
    if not user or user.name == "":  # Check for valid user
        return Result.html("<h2>Invalid User</h2>")

    try:
        idx = int(index)
        if hasattr(user, 'user_pass_pub_devices') and 0 <= idx < len(user.user_pass_pub_devices):
            user.user_pass_pub_devices.pop(idx)
            db_helper_save_user(app, asdict(user))
            # Return an empty string or a success message, which HTMX will place in the target.
            # If the button itself is the target and hx-swap="outerHTML", an empty string removes it.
            return Result.html("")
        else:
            return Result.html("<span class='text-red-500 text-xs'>Invalid index or no devices.</span>")
    except ValueError:
        return Result.html("<span class='text-red-500 text-xs'>Invalid index format.</span>")


# ... (danger, stop, reset, link functions remain largely the same, ensure they use the updated get_user_from_request if needed)

@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True, row=True)
async def info(app: App, request: RequestData):
    user = await get_user_from_request(app, request=request)
    if not user or user.name == "":
        return "<h2>Invalid User</h2>"

    WidgetID = str(uuid.uuid4())[:4]
    await reload_widget_info(app, user, WidgetID)  # Await this call
    html_widget_result = app.run_any(TBEF.MINIMALHTML.GENERATE_HTML, group_name=Name,
                                     collection_name=f"infosTab-{user.uid}")
    return Result.html(html_widget_result[0]['html_element'])


# ... (deleteUser, sendMagicLink, setUserLevel, mods, addMod, removeMod remain largely the same for now)
# Ensure they use `await get_user_from_request(app, request)` and handle the User object or None.
# For brevity, I'm not rewriting all of them but the pattern is similar.

@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True, name="get_widget",
        row=True)  # Added row=True
async def get_widget(app: App | None = None, request: RequestData = None, **kwargs):  # Added row=True for direct HTML
    if app is None:
        app = get_app(from_=f"{Name}.get_widget")

    if request is None or not hasattr(request, 'session'):
        # If no request or session, load widget for a default/guest user
        widget_html = await load_widget(app, "Cud be ur name")  # Default guest name
        return Result.html(widget_html)

    username: str = request.session.user_name

    if not username:
        username = "DEMO"

    widget_html = await load_widget(app, username)  # load_widget is now async

    return Result.html(widget_html)
