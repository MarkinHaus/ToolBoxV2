from .extras import login
from .module import Tools
from .types import User
from .UI.widget import get_widget
from .UserInstances import UserInstances
from .mini import *

from .AdminDashboard import Name as AdminDashboard
from .UserAccountManager import Name as UserAccountManagerName
from .UserDashboard import Name as UserDashboardName
from .ModManager_tests import run_mod_manager_tests

tools = Tools
Name = 'CloudM'
version = Tools.version
__all__ = ["mini"]
