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
e = export(mod_name=Name, api=True)

in_mem_chash_150 = export(mod_name=Name, memory_cache=True, memory_cache_max_size=150, version=version)


class UserInstances(metaclass=Singleton):
    live_user_instances = {}
    user_instances = {}

    @in_mem_chash_150
    def get_si_id(self, uid: str):
        return Code.one_way_hash(uid, app.id, 'SiID')

    @in_mem_chash_150
    def get_vt_id(self, uid: str):
        return Code.one_way_hash(uid, app.id, 'VirtualInstanceID')

    @in_mem_chash_150
    def get_web_socket_id(self, uid: str):
        return Code.one_way_hash(uid, app.id, 'CloudM-Signed')

    # UserInstanceManager.py
    @e
    def close_user_instance(self, uid):
        if self.get_si_id(uid) not in self.live_user_instances.keys():
            logger.warning("User instance not found")
            return "User instance not found"
        instance = self.live_user_instances[self.get_si_id(uid)]
        self.user_instances[instance['SiID']] = instance['webSocketID']
        app.run_any(
            'db', 'set',
            query=f"User::Instance::{uid}", data=
            json.dumps({"saves": instance['save']}))
        if not instance['live']:
            self.save_user_instances(instance)
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
        self.save_user_instances(instance)

    @e
    def validate_ws_id(self, ws_id):  # ToDo refactor
        logger.info(f"validate_ws_id 1 {len(self.user_instances)}")
        if len(self.user_instances) == 0:
            data = app.run_any('db', 'get',
                               query=f"user_instances::{app.id}")
            logger.info(f"validate_ws_id 2 {type(data)} {data}")
            if isinstance(data, str):
                try:
                    self.user_instances = json.loads(data)
                    logger.info(Style.GREEN("Valid instances"))
                except Exception as e:
                    logger.info(Style.RED(f"Error : {str(e)}"))
        logger.info(f"validate_ws_id ::{self.user_instances}::")
        for key in list(self.user_instances.keys()):
            value = self.user_instances[key]
            logger.info(f"validate_ws_id ::{value == ws_id}:: {key} {value}")
            if value == ws_id:
                return True, key
        return False, ""

    @e
    def delete_user_instance(self, uid):
        si_id = self.get_si_id(uid)
        if si_id not in self.user_instances.keys():
            return "User instance not found"
        if si_id in self.live_user_instances.keys():
            del self.live_user_instances[si_id]

        del self.user_instances[si_id]
        app.run_any('db', 'del', query=f"User::Instance::{uid}")
        return "Instance deleted successfully"

    @e
    def set_user_level(self):  # TODO Ad to user date default

        users, keys = [(u['save'], _) for _, u in self.live_user_instances.items()]
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

        instance = self.live_user_instances[keys[users_names.index(rot_input)]]

        instance['save']['level'] = level

        self.save_user_instances(instance)

        app.print("done")

        return True

    @e
    def save_user_instances(self, instance):
        logger.info("Saving instance")
        self.user_instances[instance['SiID']] = instance['webSocketID']
        self.live_user_instances[instance['SiID']] = instance
        app.run_any(
            'db', 'set',
            query=f"user_instances::{app.id}",
            data=json.dumps(self.user_instances))

    @e
    def get_instance_si_id(self, si_id):
        if si_id in self.live_user_instances:
            return self.live_user_instances[si_id]
        return False

    @e
    def get_user_instance(self,
                          uid: str,
                          username: str or None = None,
                          token: str or None = None,
                          hydrate: bool = True):
        # Test if an instance exist locally -> instance = set of data a dict

        instance = {
            'save': {
                'uid': uid,
                'level': 0,
                'mods': [],
                'username': username
            },
            'live': {},
            'webSocketID': self.get_web_socket_id(uid),
            'SiID': self.get_si_id(uid),
            'token': token
        }

        if instance['SiID'] in self.live_user_instances.keys():
            instance_live = self.live_user_instances[instance['SiID']]
            if 'live' in instance_live.keys():
                if instance_live['live'] and instance_live['save']['mods']:
                    logger.info(Style.BLUEBG2("Instance returned from live"))
                    return instance_live
                if instance_live['token']:
                    instance = instance_live
                    instance['live'] = {}

        if instance['SiID'] in self.user_instances.keys(
        ):  # der nutzer ist der server instanz bekannt
            instance['webSocketID'] = self.user_instances[instance['SiID']]

        chash_data = app.run_any('db', 'get', query=f"User::Instance::{uid}")
        if chash_data:
            app.print(chash_data)
            try:
                instance['save'] = json.loads(chash_data)["saves"]
            except Exception as e:
                instance['save'] = chash_data["saves"]
                logger.error(Style.YELLOW(f"Error loading instance {e}"))

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
            instance = self.hydrate_instance(instance)
        self.save_user_instances(instance)

        return instance

    @e
    def hydrate_instance(self, instance):

        # instance = {
        # 'save': {'uid':'INVADE_USER','level': -1, 'mods': []},
        # 'live': {},
        # 'webSocketID': 0000,
        # 'SiID': 0000,
        # }

        chak = instance['live'].keys()
        level = instance['save']['level']

        # app . key generator
        user_instance_name = self.get_vt_id(instance['save']['uid'])

        for mod_name in instance['save']['mods']:

            if mod_name in chak:
                continue

            user_instance_name_mod = mod_name + '-' + user_instance_name

            mod = app.get_mod(mod_name, user_instance_name)
            app.print(f"{mod_name}.instance_{mod.spec} online")

            instance['live'][mod_name] = mod
            instance['live']['v-' + mod_name] = user_instance_name_mod

        return instance

    @e
    def save_close_user_instance(self, ws_id: str):
        valid, key = self.validate_ws_id([ws_id])
        if valid:
            user_instance = self.live_user_instances[key]
            logger.info(f"Log out User : {user_instance['save']['username']}")
            for key, mod in user_instance['live'].items():
                logger.info(f"Closing {key}")
                if isinstance(mod, str):
                    continue
                try:
                    mod.on_exit()
                except Exception as e:
                    logger.error(f"error closing mod instance {key}:{e}")
            self.close_user_instance(user_instance['save']['uid'])

            return Result.ok()
        return Result.default_user_error(info="invalid ws id")
