from .prox_util import ProxyUtil
from ..singelton_class import Singleton
from ..toolbox import App


class ProxyApp(ProxyUtil, metaclass=Singleton):
    def __init__(self, app: App, host='0.0.0.0', port=6587, timeout=15):
        super().__init__(class_instance=app, host=host, port=port, timeout=timeout, app=app)
