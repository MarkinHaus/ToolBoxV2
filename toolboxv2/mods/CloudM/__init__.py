from .AdminDashboard import Name as AdminDashboard
from .email_services import Name as EmailServicesName
from .mini import *
from .ModManager import Name as ModManagerName
from .ModManager_tests import run_mod_manager_tests
from .module import Tools
from .types import User
from .UI.widget import get_widget
from .UserAccountManager import Name as UserAccountManagerName
from .UserDashboard import Name as UserDashboardName
from .UserInstances import UserInstances
from .LogInSystem import open_web_login_web
from .extras import Name as ExtrasName

tools = Tools
Name = 'CloudM'
version = Tools.version
__all__ = ["mini"]
