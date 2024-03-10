"""Automatic generated by ToolBox v = 0.1.8"""
from enum import Enum
from dataclasses import dataclass





@dataclass
class API_MANAGER(Enum):
    NAME = 'api_manager'
    START_API: str = 'start_api'  # Input: (['self', 'api_name', 'live', 'reload', 'test_override']), Output: <class 'inspect._empty'>
    RESTART_API: str = 'restart_api'  # Input: (['self', 'api_name']), Output: <class 'inspect._empty'>
    APP_INSTANCE: str = 'app_instance'  # Input: (), Output: None
    APP_INSTANCE_TYPE: str = 'app_instance_type'  # Input: (), Output: None
    VERSION: str = 'Version'  # Input: ([]), Output: <class 'inspect._empty'>
    EDITAPI: str = 'edit-api'  # Input: (['api_name', 'host', 'port']), Output: <class 'inspect._empty'>
    STARTAPI: str = 'start-api'  # Input: (['api_name', 'live', 'reload', 'test_override']), Output: <class 'inspect._empty'>
    STOPAPI: str = 'stop-api'  # Input: (['api_name', 'delete']), Output: <class 'inspect._empty'>
    INFO: str = 'info'  # Input: ([]), Output: <class 'inspect._empty'>
    RESTARTAPI: str = 'restart-api'  # Input: (['api_name']), Output: <class 'inspect._empty'>


@dataclass
class WELCOME(Enum):
    NAME = 'welcome'
    VERSION: str = 'Version'  # Input: ([]), Output: <class 'inspect._empty'>
    PRINTT: str = 'printT'  # Input: ([]), Output: <class 'inspect._empty'>
    ON_START: str = 'on_start'  # Input: (), Output: None
    ANIMATION1: str = 'Animation1'  # Input: ([]), Output: <class 'inspect._empty'>
    ANIMATION: str = 'Animation'  # Input: ([]), Output: <class 'inspect._empty'>
    APP_INSTANCE: str = 'app_instance'  # Input: (), Output: None
    APP_INSTANCE_TYPE: str = 'app_instance_type'  # Input: (), Output: None


@dataclass
class SCHEDULER_MANAGER(Enum):
    NAME = 'SchedulerManager'
    INIT: str = 'init'  # Input: (['app']), Output: <class 'inspect._empty'>
    ON_START: str = 'on_start'  # Input: (), Output: None
    INSTANCE: str = 'instance'  # Input: (['app']), Output: <class 'inspect._empty'>
    ON_EXIT: str = 'on_exit'  # Input: (), Output: None
    SCHEDULER_MANAGER_0: str = 'scheduler_manager_0'  # Input: ([]), Output: <class 'inspect._empty'>
    START: str = 'start'  # Input: ([]), Output: <class 'inspect._empty'>
    STOP: str = 'stop'  # Input: ([]), Output: <class 'inspect._empty'>
    CANCEL: str = 'cancel'  # Input: (['job_id']), Output: <class 'inspect._empty'>
    DEALT: str = 'dealt'  # Input: (['job_id']), Output: <class 'inspect._empty'>
    ADD: str = 'add'  # Input: (['job_data']), Output: <class 'inspect._empty'>
    APP_INSTANCE: str = 'app_instance'  # Input: (), Output: None
    APP_INSTANCE_TYPE: str = 'app_instance_type'  # Input: (), Output: None


@dataclass
class CLOUDM_USERINSTANCES(Enum):
    NAME = 'CloudM.UserInstances'
    GET_SI_ID: str = 'get_si_id'  # Input: (['uid']), Output: <class 'toolboxv2.utils.types.Result'>
    GET_VT_ID: str = 'get_vt_id'  # Input: (['uid']), Output: <class 'toolboxv2.utils.types.Result'>
    GET_WEB_SOCKET_ID: str = 'get_web_socket_id'  # Input: (['uid']), Output: <class 'toolboxv2.utils.types.Result'>
    CLOSE_USER_INSTANCE: str = 'close_user_instance'  # Input: (['uid']), Output: <class 'inspect._empty'>
    VALIDATE_WS_ID: str = 'validate_ws_id'  # Input: (['ws_id']), Output: <class 'inspect._empty'>
    DELETE_USER_INSTANCE: str = 'delete_user_instance'  # Input: (['uid']), Output: <class 'inspect._empty'>
    SAVE_USER_INSTANCES: str = 'save_user_instances'  # Input: (['instance']), Output: <class 'inspect._empty'>
    GET_INSTANCE_SI_ID: str = 'get_instance_si_id'  # Input: (['si_id']), Output: <class 'inspect._empty'>
    GET_USER_INSTANCE: str = 'get_user_instance'  # Input: (['uid', 'hydrate']), Output: <class 'inspect._empty'>
    HYDRATE_INSTANCE: str = 'hydrate_instance'  # Input: (['instance']), Output: <class 'inspect._empty'>
    SAVE_CLOSE_USER_INSTANCE: str = 'save_close_user_instance'  # Input: (['ws_id']), Output: <class 'inspect._empty'>


@dataclass
class EMAIL_WAITING_LIST(Enum):
    NAME = 'email_waiting_list'
    ADD: str = 'add'  # Input: (['app', 'email']), Output: <class 'toolboxv2.utils.types.ApiResult'>
    SEND_EMAIL_TO_ALL: str = 'send_email_to_all'  # Input: ([]), Output: <class 'inspect._empty'>
    SEND_EMAIL: str = 'send_email'  # Input: (['data']), Output: <class 'inspect._empty'>
    CRATE_SING_IN_EMAIL: str = 'crate_sing_in_email'  # Input: (['user_email', 'user_name']), Output: <class 'inspect._empty'>
    CRATE_MAGICK_LICK_DEVICE_EMAIL: str = 'crate_magick_lick_device_email'  # Input: (['user_email', 'user_name', 'link_id', 'nl']), Output: <class 'inspect._empty'>
    APP_INSTANCE: str = 'app_instance'  # Input: (), Output: None
    APP_INSTANCE_TYPE: str = 'app_instance_type'  # Input: (), Output: None


@dataclass
class SOCKETMANAGER(Enum):
    NAME = 'SocketManager'
    CREATE_SOCKET: str = 'create_socket'  # Input: (['self', 'name', 'host', 'port', 'type_id', 'max_connections', 'endpoint_port', 'return_full_object', 'keepalive_interval', 'test_override', 'package_size', 'start_keep_alive']), Output: <class 'inspect._empty'>
    RUN_AS_IP_ECHO_SERVER_A: str = 'run_as_ip_echo_server_a'  # Input: (['self', 'name', 'host', 'port', 'max_connections', 'test_override']), Output: <class 'inspect._empty'>
    RUN_AS_SINGLE_COMMUNICATION_SERVER: str = 'run_as_single_communication_server'  # Input: (['self', 'name', 'host', 'port', 'test_override']), Output: <class 'inspect._empty'>
    SEND_FILE_TO_SEVER: str = 'send_file_to_sever'  # Input: (['self', 'filepath', 'host', 'port']), Output: <class 'inspect._empty'>
    RECEIVE_AND_DECOMPRESS_FILE_AS_SERVER: str = 'receive_and_decompress_file_as_server'  # Input: (['self', 'save_path', 'listening_port']), Output: <class 'inspect._empty'>
    SEND_FILE_TO_PEER: str = 'send_file_to_peer'  # Input: (['self', 'filepath', 'host', 'port']), Output: <class 'inspect._empty'>
    RECEIVE_AND_DECOMPRESS_FILE: str = 'receive_and_decompress_file'  # Input: (['self', 'save_path', 'listening_port', 'sender_ip']), Output: <class 'inspect._empty'>
    APP_INSTANCE: str = 'app_instance'  # Input: (), Output: None
    APP_INSTANCE_TYPE: str = 'app_instance_type'  # Input: (), Output: None
    SOCKETMANAGER_0: str = 'SocketManager_0'  # Input: (['name', 'host', 'port', 'type_id', 'max_connections', 'endpoint_port', 'return_full_object', 'keepalive_interval', 'test_override', 'package_size', 'start_keep_alive']), Output: <class 'inspect._empty'>
    TBSOCKETCONTROLLER: str = 'tbSocketController'  # Input: (['name', 'host', 'port', 'test_override']), Output: <class 'inspect._empty'>
    VERSION: str = 'Version'  # Input: ([]), Output: <class 'inspect._empty'>


@dataclass
class DB(Enum):
    NAME = 'DB'
    VERSION: str = 'Version'  # Input: (['self']), Output: <class 'inspect._empty'>
    GET: str = 'get'  # Input: (['self', 'query']), Output: <class 'toolboxv2.utils.types.Result'>
    IF_EXIST: str = 'if_exist'  # Input: (['self', 'query']), Output: <class 'toolboxv2.utils.types.Result'>
    SET: str = 'set'  # Input: (['self', 'query', 'data']), Output: <class 'toolboxv2.utils.types.Result'>
    INITIALIZED: str = 'initialized'  # Input: (['self']), Output: <class 'bool'>
    DELETE: str = 'delete'  # Input: (['self', 'query', 'matching']), Output: <class 'toolboxv2.utils.types.Result'>
    APPEND_ON_SET: str = 'append_on_set'  # Input: (['self', 'query', 'data']), Output: <class 'toolboxv2.utils.types.Result'>
    INITIALIZE_DATABASE: str = 'initialize_database'  # Input: (['self']), Output: <class 'toolboxv2.utils.types.Result'>
    ON_START: str = 'on_start'  # Input: (), Output: None
    CLOSE_DB: str = 'close_db'  # Input: (['self']), Output: <class 'toolboxv2.utils.types.Result'>
    ON_EXIT: str = 'on_exit'  # Input: (), Output: None
    EDIT_PROGRAMMABLE: str = 'edit_programmable'  # Input: (['self', 'mode']), Output: <class 'inspect._empty'>
    EDIT_CLI: str = 'edit_cli'  # Input: (['self', 'mode']), Output: <class 'inspect._empty'>
    EDIT_DEV_WEB_UI: str = 'edit_dev_web_ui'  # Input: (['self', 'mode']), Output: <class 'inspect._empty'>
    APP_INSTANCE: str = 'app_instance'  # Input: (), Output: None
    APP_INSTANCE_TYPE: str = 'app_instance_type'  # Input: (), Output: None


@dataclass
class MINIMALHTML(Enum):
    NAME = 'MinimalHtml'
    APP_INSTANCE: str = 'app_instance'  # Input: (), Output: None
    APP_INSTANCE_TYPE: str = 'app_instance_type'  # Input: (), Output: None
    VERSION: str = 'Version'  # Input: ([]), Output: <class 'inspect._empty'>
    ADD_GROUP: str = 'add_group'  # Input: (['command']), Output: <class 'inspect._empty'>
    ADD_COLLECTION_TO_GROUP: str = 'add_collection_to_group'  # Input: (['command']), Output: <class 'inspect._empty'>
    GENERATE_HTML: str = 'generate_html'  # Input: (['group_name', 'collection_name']), Output: <class 'inspect._empty'>
    FUSE_TO_STRING: str = 'fuse_to_string'  # Input: (['html_elements', 'join_chat']), Output: <class 'inspect._empty'>


@dataclass
class CLI_FUNCTIONS(Enum):
    NAME = 'cli_functions'
    VERSION: str = 'Version'  # Input: ([]), Output: <class 'inspect._empty'>
    GET_CHARACTER: str = 'get_character'  # Input: ([]), Output: <class 'inspect._empty'>
    GET_GENERATOR: str = 'get_generator'  # Input: ([]), Output: <class 'inspect._empty'>
    UPDATE_AUTOCOMPLETION_MODS: str = 'update_autocompletion_mods'  # Input: (['app', 'autocompletion_dict']), Output: <class 'inspect._empty'>
    UPDATE_AUTOCOMPLETION_LIST_OR_KEY: str = 'update_autocompletion_list_or_key'  # Input: (['list_or_key', 'autocompletion_dict', 'raise_e', 'do_lower']), Output: <class 'inspect._empty'>
    USER_INPUT: str = 'user_input'  # Input: (['app', 'completer_dict', 'get_rprompt', 'bottom_toolbar', 'active_modul', 'password', 'bindings', 'message']), Output: <class 'toolboxv2.utils.types.CallingObject'>
    CO_EVALUATE: str = 'co_evaluate'  # Input: (['app', 'obj', 'build_in_commands', 'threaded', 'helper', 'return_parm']), Output: <class 'inspect._empty'>
    APP_INSTANCE: str = 'app_instance'  # Input: (), Output: None
    APP_INSTANCE_TYPE: str = 'app_instance_type'  # Input: (), Output: None


@dataclass
class CLOUDM_MODMANAGER(Enum):
    NAME = 'CloudM.ModManager'
    INSTALLER: str = 'installer'  # Input: (['url', 'debug']), Output: <class 'inspect._empty'>
    DELETE_PACKAGE: str = 'delete_package'  # Input: (['url']), Output: <class 'inspect._empty'>
    LIST_MODULES: str = 'list_modules'  # Input: (['app']), Output: <class 'inspect._empty'>


@dataclass
class CLOUDM_AUTHMANAGER(Enum):
    NAME = 'CloudM.AuthManager'
    GET_USER_BY_NAME: str = 'get_user_by_name'  # Input: (['app', 'username', 'uid']), Output: <class 'toolboxv2.utils.types.Result'>
    CREATE_USER: str = 'create_user'  # Input: (['app', 'data', 'username', 'email', 'pub_key', 'invitation', 'web_data', 'as_base64']), Output: <class 'toolboxv2.utils.types.ApiResult'>
    GET_MAGICK_LINK_EMAIL: str = 'get_magick_link_email'  # Input: (['app', 'username']), Output: <class 'inspect._empty'>
    ADD_USER_DEVICE: str = 'add_user_device'  # Input: (['app', 'data', 'username', 'pub_key', 'invitation', 'web_data', 'as_base64']), Output: <class 'toolboxv2.utils.types.ApiResult'>
    REGISTER_USER_PERSONAL_KEY: str = 'register_user_personal_key'  # Input: (['app', 'data']), Output: <class 'toolboxv2.utils.types.ApiResult'>
    CRATE_LOCAL_ACCOUNT: str = 'crate_local_account'  # Input: (['app', 'username', 'email', 'invitation', 'create']), Output: <class 'toolboxv2.utils.types.Result'>
    LOCAL_LOGIN: str = 'local_login'  # Input: (['app', 'username']), Output: <class 'toolboxv2.utils.types.Result'>
    GET_TO_SING_DATA: str = 'get_to_sing_data'  # Input: (['app', 'username', 'personal_key']), Output: <class 'inspect._empty'>
    GET_INVITATION: str = 'get_invitation'  # Input: (['app']), Output: <class 'toolboxv2.utils.types.Result'>
    VALIDATE_PERSONA: str = 'validate_persona'  # Input: (['app', 'data']), Output: <class 'toolboxv2.utils.types.ApiResult'>
    VALIDATE_DEVICE: str = 'validate_device'  # Input: (['app', 'data']), Output: <class 'toolboxv2.utils.types.ApiResult'>
    AUTHENTICATE_USER_GET_SYNC_KEY: str = 'authenticate_user_get_sync_key'  # Input: (['app', 'username', 'signature', 'get_user', 'web']), Output: <class 'toolboxv2.utils.types.ApiResult'>
    GET_USER_SYNC_KEY_LOCAL: str = 'get_user_sync_key_local'  # Input: (['app', 'username', 'ausk']), Output: <class 'toolboxv2.utils.types.Result'>
    JWT_GET_CLAIM: str = 'jwt_get_claim'  # Input: (['app', 'username', 'signature', 'web']), Output: <class 'toolboxv2.utils.types.ApiResult'>
    JWT_CLAIM_LOCAL_DECRYPT: str = 'jwt_claim_local_decrypt'  # Input: (['app', 'username', 'crypt_sing_jwt_claim', 'aud']), Output: <class 'toolboxv2.utils.types.Result'>
    JWT_CHECK_CLAIM_SERVER_SIDE: str = 'jwt_check_claim_server_side'  # Input: (['app', 'username', 'jwt_claim']), Output: <class 'toolboxv2.utils.types.ApiResult'>


@dataclass
class CLOUDM(Enum):
    NAME = 'CloudM'
    NEW_MODULE: str = 'new_module'  # Input: (['self', 'mod_name', 'options']), Output: <class 'inspect._empty'>
    CREATE_ACCOUNT: str = 'create_account'  # Input: (['self']), Output: <class 'inspect._empty'>
    INIT_GIT: str = 'init_git'  # Input: (['_']), Output: <class 'inspect._empty'>
    UPDATE_CORE: str = 'update_core'  # Input: (['self', 'backup', 'name']), Output: <class 'inspect._empty'>
    REGISTER_INITIAL_ROOT_USER: str = 'register_initial_root_user'  # Input: (['app']), Output: <class 'inspect._empty'>
    CLEAR_DB: str = 'clear_db'  # Input: (['self', 'do_root']), Output: <class 'inspect._empty'>
    SHOW_VERSION: str = 'show_version'  # Input: (['self']), Output: <class 'inspect._empty'>
    APP_INSTANCE: str = 'app_instance'  # Input: (), Output: None
    APP_INSTANCE_TYPE: str = 'app_instance_type'  # Input: (), Output: None
    MODINSTALLER: str = 'mod-installer'  # Input: (['name']), Output: <class 'inspect._empty'>
    MODUNINSTALLER: str = 'mod-uninstaller'  # Input: (['name']), Output: <class 'inspect._empty'>


@dataclass
class DASHPROVIDER(Enum):
    NAME = 'DashProvider'
    VERSION: str = 'Version'  # Input: ([]), Output: <class 'inspect._empty'>
    GET_CONTROLLER: str = 'get_controller'  # Input: (['app', 'request']), Output: <class 'inspect._empty'>
    GETSMSG: str = 'getsMSG'  # Input: (['app']), Output: <class 'inspect._empty'>
    GETSINSIGHTSWIDGET: str = 'getsInsightsWidget'  # Input: (['app']), Output: <class 'inspect._empty'>
    GETTEXTWIDGET: str = 'getTextWidget'  # Input: (['app']), Output: <class 'inspect._empty'>
    GETPATHWIDGET: str = 'getPathWidget'  # Input: (['app']), Output: <class 'inspect._empty'>
    GETWIDGETNAVE: str = 'getWidgetNave'  # Input: (['app']), Output: <class 'inspect._empty'>
    GETDRAG: str = 'getDrag'  # Input: (['app']), Output: <class 'inspect._empty'>
    GETCONTROLS: str = 'getControls'  # Input: (['app']), Output: <class 'inspect._empty'>
    SERVICEWORKER: str = 'serviceWorker'  # Input: (['app']), Output: <class 'inspect._empty'>
    APP_INSTANCE: str = 'app_instance'  # Input: (), Output: None
    APP_INSTANCE_TYPE: str = 'app_instance_type'  # Input: (), Output: None


@dataclass
class QUICKNOTE(Enum):
    NAME = 'quickNote'
    APP_INSTANCE: str = 'app_instance'  # Input: (), Output: None
    APP_INSTANCE_TYPE: str = 'app_instance_type'  # Input: (), Output: None
    VERSION: str = 'Version'  # Input: ([]), Output: <class 'inspect._empty'>
    ADD: str = 'ADD'  # Input: (['note']), Output: <class 'inspect._empty'>
    REMOVE: str = 'REMOVE'  # Input: (['note']), Output: <class 'inspect._empty'>
    VIEW: str = 'VIEW'  # Input: (['show', 'data']), Output: <class 'inspect._empty'>
    FIND: str = 'Find'  # Input: (['show', 'data']), Output: <class 'inspect._empty'>
    INIT: str = 'init'  # Input: (['username', 'sign', 'jwt']), Output: <class 'inspect._empty'>


@dataclass
class WEBSOCKETMANAGER(Enum):
    NAME = 'WebSocketManager'
    APP_INSTANCE: str = 'app_instance'  # Input: (), Output: None
    APP_INSTANCE_TYPE: str = 'app_instance_type'  # Input: (), Output: None
    VERSION: str = 'Version'  # Input: ([]), Output: <class 'inspect._empty'>
    CONNECT: str = 'connect'  # Input: (['websocket', 'websocket_id']), Output: <class 'inspect._empty'>
    DISCONNECT: str = 'disconnect'  # Input: (['websocket', 'websocket_id']), Output: <class 'inspect._empty'>
    SEND_MESSAGE: str = 'send_message'  # Input: (['message', 'websocket', 'websocket_id']), Output: <class 'inspect._empty'>
    LIST: str = 'list'  # Input: ([]), Output: <class 'inspect._empty'>
    GET: str = 'get'  # Input: (['name']), Output: <class 'inspect._empty'>
    SRQW: str = 'srqw'  # Input: (['url', 'websocket_id']), Output: <class 'inspect._empty'>
    CONSTRUCT_RENDER: str = 'construct_render'  # Input: (['content', 'element_id', 'externals', 'placeholder_content', 'from_file', 'to_str']), Output: <class 'inspect._empty'>


@dataclass
class AUDIO(Enum):
    NAME = 'audio'
    ON_STARTUP: str = 'on_startup'  # Input: (['app']), Output: <class 'inspect._empty'>
    ON_START: str = 'on_start'  # Input: (), Output: None
    INIT_SPEECH: str = 'init_speech'  # Input: (['app']), Output: <class 'inspect._empty'>
    ON_CLOSE: str = 'on_close'  # Input: (['args']), Output: <class 'inspect._empty'>
    ON_EXIT: str = 'on_exit'  # Input: (), Output: None
    SPEECH: str = 'speech'  # Input: (['text', 'voice_index', 'use_cache', 'provider', 'config']), Output: <class 'inspect._empty'>
    SPEECH_STREAM: str = 'speech_stream'  # Input: (['text', 'voice_index', 'use_cache', 'provider', 'config']), Output: <class 'inspect._empty'>
    TRANSCRIPT: str = 'transcript'  # Input: (['model', 'rate', 'chunk_duration', 'amplitude_min', 'microphone_index']), Output: <class 'inspect._empty'>
    WAKE_WORD: str = 'wake_word'  # Input: (['word', 'variants', 'microphone_index', 'amplitude_min', 'model', 'do_exit', 'do_stop', 'ques']), Output: <class 'inspect._empty'>
    TRANSCRIPT_STREAM: str = 'transcript_stream'  # Input: ([]), Output: <class 'inspect._empty'>
    APP_INSTANCE: str = 'app_instance'  # Input: (), Output: None
    APP_INSTANCE_TYPE: str = 'app_instance_type'  # Input: (), Output: None


@dataclass
class DIFFUSER(Enum):
    NAME = 'diffuser'
    ON_STARTUP: str = 'on_startup'  # Input: (['app']), Output: <class 'inspect._empty'>
    ON_START: str = 'on_start'  # Input: (), Output: None
    ON_CLOSE: str = 'on_close'  # Input: (['app']), Output: <class 'inspect._empty'>
    ON_EXIT: str = 'on_exit'  # Input: (), Output: None
    START_UI: str = 'start_ui'  # Input: (['app']), Output: <class 'inspect._empty'>
    GET_PIPLINE_MANAGER: str = 'get_pipline_manager'  # Input: (['args']), Output: <class 'inspect._empty'>
    APP_INSTANCE: str = 'app_instance'  # Input: (), Output: None
    APP_INSTANCE_TYPE: str = 'app_instance_type'  # Input: (), Output: None


@dataclass
class ISAA(Enum):
    NAME = 'isaa'
    APP_INSTANCE: str = 'app_instance'  # Input: (), Output: None
    APP_INSTANCE_TYPE: str = 'app_instance_type'  # Input: (), Output: None
    VERSION: str = 'Version'  # Input: ([]), Output: <class 'inspect._empty'>
    ADD_TASK: str = 'add_task'  # Input: (['name', 'task']), Output: <class 'inspect._empty'>
    SAVE_TASK: str = 'save_task'  # Input: (['name']), Output: <class 'inspect._empty'>
    LOAD_TASK: str = 'load_task'  # Input: (['name']), Output: <class 'inspect._empty'>
    GET_TASK: str = 'get_task'  # Input: (['name']), Output: <class 'inspect._empty'>
    LIST_TASK: str = 'list_task'  # Input: ([]), Output: <class 'inspect._empty'>
    SAVE_TO_MEM: str = 'save_to_mem'  # Input: ([]), Output: <class 'inspect._empty'>
    SET_LOCAL_FILES_TOOLS: str = 'set_local_files_tools'  # Input: (['local_files_tools']), Output: <class 'inspect._empty'>
