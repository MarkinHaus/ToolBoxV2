import json

from toolboxv2 import Style, get_app, tbef
from toolboxv2.utils import Singleton
from toolboxv2.utils.cryp import Code
from toolboxv2.utils.types import Result

app = get_app("UserInstances")
logger = app.logger
Name = "CloudM.UserInstances"
version = "0.0.1"
export = app.tb
e = export(mod_name=Name, api=False)
in_mem_chash_150 = export(mod_name=Name, memory_cache=True, memory_cache_max_size=150, version=version)


class UserInstances(metaclass=Singleton):
    live_user_instances = {}
    user_instances = {}

    @staticmethod
    @in_mem_chash_150
    def get_si_id(uid: str):
        return Code.one_way_hash(uid, app.id, 'SiID')

    @staticmethod
    @in_mem_chash_150
    def get_vt_id(uid: str):
        return Code.one_way_hash(uid, app.id, 'VirtualInstanceID')

    @staticmethod
    @in_mem_chash_150
    def get_web_socket_id(uid: str):
        return Code.one_way_hash(uid, app.id, 'CloudM-Signed')

    # UserInstanceManager.py


@e
def close_user_instance(uid: str):
    if uid is None:
        return
    if UserInstances.get_si_id(uid).get()not in UserInstances().live_user_instances.keys():
        logger.warning("User instance not found")
        return "User instance not found"
    instance = UserInstances().live_user_instances[UserInstances.get_si_id(uid).get()]
    UserInstances().user_instances[instance['SiID']] = instance['webSocketID']
    app.run_any(
        'db', 'set',
        query=f"User::Instance::{uid}", data=
        json.dumps({"saves": instance['save']}))
    if not instance['live']:
        save_user_instances(instance)
        logger.info("No modules to close")
        return "No modules to close"
    for key, val in instance['live'].items():
        if key.startswith('v-'):
            continue
        try:
            val._on_exit()
        except Exception as e:
            logger.error(f"Error closing {key}, {str(e)}")
    del instance['live']
    instance['live'] = {}
    logger.info("User instance live removed")
    save_user_instances(instance)


@e
def validate_ws_id(ws_id):  # ToDo refactor
    logger.info(f"validate_ws_id 1 {len(UserInstances().user_instances)}")
    if len(UserInstances().user_instances) == 0:
        data = app.run_any('DB', 'get',
                           query=f"user_instances::{app.id}")
        logger.info(f"validate_ws_id 2 {type(data)} {data}")
        if isinstance(data, str):
            try:
                UserInstances().user_instances = json.loads(data)
                logger.info(Style.GREEN("Valid instances"))
            except Exception as e:
                logger.info(Style.RED(f"Error : {str(e)}"))
    logger.info(f"validate_ws_id ::{UserInstances().user_instances}::")
    for key in list(UserInstances().user_instances.keys()):
        value = UserInstances().user_instances[key]
        logger.info(f"validate_ws_id ::{value == ws_id}:: {key} {value}")
        if value == ws_id:
            return True, key
    return False, ""


@e
def delete_user_instance(uid: str):
    if uid is None:
        return
    si_id = UserInstances.get_si_id(uid).get()
    if si_id not in UserInstances().user_instances.keys():
        return "User instance not found"
    if si_id in UserInstances().live_user_instances.keys():
        del UserInstances().live_user_instances[si_id]

    del UserInstances().user_instances[si_id]
    app.run_any('db', 'del', query=f"User::Instance::{uid}")
    return "Instance deleted successfully"


@export(mod_name=Name, state=False)
def set_user_level():  # TODO Ad to user date default

    if not UserInstances().live_user_instances.items():
        app.print(f"User: No users registered")
        return

    users, keys = [(u['save'], _) for _, u in UserInstances().live_user_instances.items()]
    users_names = [u['username'] for u in users]
    for user in users:
        app.print(f"User: {user['username']} level : {user['level']}")

    rot_input = input("Username: ")
    if not rot_input:
        app.print(Style.YELLOW("Please enter a username"))
        return "Please enter a username"
    if rot_input not in users_names:
        app.print(Style.YELLOW("Please enter a valid username"))
        return "Please enter a valid username"

    user = users[users_names.index(rot_input)]

    app.print(Style.WHITE(f"Usr level : {user['level']}"))

    level = input("set level :")
    level = int(level)

    instance = UserInstances().live_user_instances[keys[users_names.index(rot_input)]]

    instance['save']['level'] = level

    save_user_instances(instance)

    app.print("done")

    return True


@e
def save_user_instances(instance: dict):
    if instance is None:
        return
    logger.info("Saving instance")
    UserInstances().user_instances[instance['SiID']] = instance['webSocketID']
    UserInstances().live_user_instances[instance['SiID']] = instance
    print(UserInstances().user_instances)
    app.run_any(
        'DB', 'set',
        query=f"user_instances::{app.id}",
        data=json.dumps(UserInstances().user_instances))


@e
def get_instance_si_id(si_id):
    if si_id in UserInstances().live_user_instances:
        return UserInstances().live_user_instances[si_id]
    return False


@e
def get_user_instance(uid: str,
                      username: str or None = None,
                      token: str or None = None,
                      hydrate: bool = True):
    # Test if an instance exist locally -> instance = set of data a dict
    if uid is None:
        return
    instance = {
        'save': {
            'uid': uid,
            'level': 0,
            'mods': [],
            'username': username
        },
        'live': {},
        'webSocketID': UserInstances.get_web_socket_id(uid).get(),
        'SiID': UserInstances.get_si_id(uid).get(),
        'token': token
    }

    if instance['SiID'] in UserInstances().live_user_instances.keys():
        instance_live = UserInstances().live_user_instances.get([instance['SiID']], {})
        if 'live' in instance_live.keys():
            if instance_live['live'] and instance_live['save']['mods']:
                logger.info(Style.BLUEBG2("Instance returned from live"))
                return instance_live
            if instance_live['token']:
                instance = instance_live
                instance['live'] = {}
    chash = {}
    if instance['SiID'] in UserInstances().user_instances.keys(
    ):  # der nutzer ist der server instanz bekannt
        instance['webSocketID'] = UserInstances().user_instances[instance['SiID']]
    else:
        chash_data = app.run_any('DB', 'get', query=f"User::Instance::{uid}", get_results=True)
        if not chash_data.is_data():
            chash = {"saves": instance['save']}
        else:
            chash = chash_data.get()
    if chash != {}:
        app.print(chash)
        try:
            instance['save'] = json.loads(chash)["saves"]
        except Exception as er:
            instance['save'] = chash["saves"]
            logger.error(Style.YELLOW(f"Error loading instance {er}"))

    logger.info(Style.BLUEBG(f"Init mods : {instance['save']['mods']}"))

    app.print(Style.MAGENTA(f"instance : {instance}"))

    #   if no instance is local available look at the upper instance.
    #       if instance is available download and install the instance.
    #   if no instance is available create a new instance
    # upper = instance['save']
    # # get from upper instance
    # # upper = get request ...
    # instance['save'] = upper
    if hydrate:
        instance = hydrate_instance(instance)
    save_user_instances(instance)

    return instance


@e
def hydrate_instance(instance: dict):
    # instance = {
    # 'save': {'uid':'INVADE_USER','level': -1, 'mods': []},
    # 'live': {},
    # 'webSocketID': 0000,
    # 'SiID': 0000,
    # }

    if instance is None:
        return

    chak = instance['live'].keys()
    level = instance['save']['level']

    # app . key generator
    user_instance_name = UserInstances.get_vt_id(instance['save']['uid']).get()

    for mod_name in instance['save']['mods']:

        if mod_name in chak:
            continue

        user_instance_name_mod = mod_name + '-' + user_instance_name

        mod = app.get_mod(mod_name, user_instance_name)
        app.print(f"{mod_name}.instance_{mod.spec} online")

        instance['live'][mod_name] = mod
        instance['live']['v-' + mod_name] = user_instance_name_mod

    return instance


@export(mod_name=Name, state=False)
def save_close_user_instance(ws_id: str):
    valid, key = validate_ws_id(ws_id)
    if valid:
        user_instance = UserInstances().live_user_instances[key]
        logger.info(f"Log out User : {user_instance['save']['username']}")
        for key, mod in user_instance['live'].items():
            logger.info(f"Closing {key}")
            if isinstance(mod, str):
                continue
            try:
                mod.on_exit()
            except Exception as e:
                logger.error(f"error closing mod instance {key}:{e}")
        close_user_instance(user_instance['save']['uid'])

        return Result.ok()
    return Result.default_user_error(info="invalid ws id")
