import asyncio
import functools
import inspect
import json
import threading
from collections.abc import Callable

from toolboxv2 import TBEF, ApiResult, App, Result, get_app
from toolboxv2.utils.system.session import RequestSession


def bottle_up(tb_app, user='root', main_route=None, threaded=False, **kwargs):
    import os
    try:
        from bottle import Bottle, HTTPResponse, request, static_file
    except ImportError:
        print("Bottle is not available auto installation")
        os.system("pip install bottle")
        return bottle_up(tb_app, user=user, main_route=main_route, **kwargs)

    def e404(*args):
        return static_file("/assets/404.html", root=get_app().start_dir + '/web')

    def e401(*args):
        return static_file("/assets/401.html", root=get_app().start_dir + '/web')

    class TBAppBottle(Bottle):
        def __init__(self, username=None):
            super().__init__()

            self.tb_app: App = tb_app

            self.username = username if username is not None else tb_app.get_username()

            self._auto_generate_routes(main_route)

            self._setup_user()
            self.populate_session()
            # self._setup_middleware()

            if main_route is None:
                self.route('/', method='GET')(self.index_h)
            # Dynamische Proxy-Route einrichten
            self.route('/<path:path>', 'ANY', self.proxy_route)

            self.error(404, e404)
            self.error(401, e401)

        def proxy_route(self, path):
            # Direct to static files if there's an extension (e.g., CSS, JS)
            if '.' in path:
                return static_file(path, root=os.path.join(self.tb_app.start_dir, 'dist'))

            # Direct requests to API endpoints as usual
            if path.startswith('api'):
                return None  # Allow Bottle to attempt to match API routes

            if main_route is not None:
                return None
            # SPA fallback route to serve index.html
            return static_file('index.html', root=os.path.join(self.tb_app.start_dir, 'dist'))

        def _setup_user(self):
            self.tb_app.get_mod("CloudM")
            self.user = self.tb_app.run_any(TBEF.CLOUDM_AUTHMANAGER.GET_USER_BY_NAME, username=self.username,
                                            get_results=True)
            self.user.print()
            self.user_instance = self.tb_app.run_any(TBEF.CLOUDM_USERINSTANCES.GET_USER_INSTANCE,
                                                     uid=self.user.get().uid,
                                                     hydrate=False)

        def populate_session(self):

            self.session = {
                'SiID': self.user_instance.get('SiID'),
                'level': self.user.get().level if self.user.get().level > 1 else 1,
                'spec': self.user_instance.get('VtID'),
                'user_name': self.tb_app.config_fh.encode_code(self.user.get().name)
            }

        def index_j(self):
            return static_file('index.js', root=self.tb_app.start_dir)

        def index_h(self):
            return static_file('index.html', root=self.tb_app.start_dir + '/dist/')

        def _auto_generate_routes(self, main_rout):
            """Auto-generate routes based on tb_app functions"""

            self.route('/index.js', method='GET')(self.index_j)

            try:
                with open(os.path.join(self.tb_app.start_dir, 'dist', 'helper.html')) as f:
                    f.read()
            except Exception:
                self.tb_app.debug_rains(RuntimeError(f"pleas build the main app or the (js) and save in {os.path.join(self.tb_app.start_dir, 'dist', 'helper.html')}"))

            for mod_name, functions in self.tb_app.functions.items():
                for func_name, func_data in functions.items():
                    if not isinstance(func_data, dict) or func_data.get('api') is False:
                        continue

                    # Get function and check for errors
                    # tb_func, error = self.tb_app.get_function(
                    #     (mod_name, func_name),
                    #     state=func_data.get('state', False),
                    #     specification="app"
                    # )

                    # if error != 0 or not tb_func:
                    #     continue

                    tb_func = func_data.get("func")

                    if tb_func is None:
                        continue

                    request_as_kwarg = func_data.get('request_as_kwarg', False)

                    # Handle web main entries as first-class routes
                    if 'main' in func_name and 'web' in func_name:

                        if request_as_kwarg:
                            def tb_func_(**kw):
                                return open(os.path.join(self.tb_app.start_dir, 'dist', 'helper.html')).read()+tb_func(**kw)
                        else:
                            def tb_func_():
                                return open(os.path.join(self.tb_app.start_dir, 'dist', 'helper.html')).read() + tb_func()
                        self.route(f'/{mod_name}', method='GET')(tb_func_)
                        print("adding root:", f'/{mod_name}')
                        if mod_name == main_rout:
                            self.route('/', method='GET')(tb_func_)
                        continue

                    # Handle websocket routes
                    if 'websocket' in func_name:
                        # Note: Bottle doesn't have native WebSocket support
                        # You might want to use a WebSocket library like gevent-websocket
                        continue

                    # Regular API routes
                    route_path = f'/api/{mod_name}/{func_name}'
                    methods = func_data.get('api_methods', ['GET'])

                    if methods == ['AUTO']:
                        methods = ['GET'] if len(func_data.get('params')) <= 2 else ['POST']

                    for method in methods:
                        self.route(route_path, method=method)(self._wrap_tb_func(tb_func, request_as_kwarg))
                        print("adding api:", f'{method} {route_path}')

        def _wrap_tb_func(self, func: Callable, request_as_kwarg=False):
            """Wrap tb_app functions to handle async/await and return proper responses"""

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                def req_to_rsq():
                    self.tb_app.get_username()
                    return RequestSession(
                        session=self.session, body=request.body, json=request.json, row=request
                    )
                if request_as_kwarg:
                    kwargs['request'] = req_to_rsq()
                kwargs.update(request.query)
                #try:
                # Handle async functions
                if inspect.iscoroutinefunction(func):
                    result = asyncio.run(func(*args, **kwargs))
                else:
                    result = func(*args, **kwargs)

                # Handle different response types
                if isinstance(result, ApiResult):
                    result = result.as_result()

                if isinstance(result, Result):
                    result = result.get()

                if isinstance(result, str):
                    return result

                if isinstance(result, HTTPResponse):
                    return result

                return HTTPResponse(
                    body=json.dumps(result),
                    headers={'Content-Type': 'application/json'}
                )
                # except Exception as e:
                #     return HTTPResponse(
                #         status=500,
                #         body=json.dumps({"error": str(e)}),
                #         headers={'Content-Type': 'application/json'}
                #     )

            return wrapper

    def create_bottle_app() -> TBAppBottle:
        """Create and configure a Bottle application with TB app integration"""
        app = TBAppBottle(username=user)

        # Add HotReload functionality if needed
        if "HotReload" in tb_app.id:
            @app.get('/HotReload')
            def hot_reload():
                if tb_app.debug:
                    tb_app.remove_all_modules()
                    asyncio.run(tb_app.load_all_mods_in_file())
                    return "OK"
                return "Not found"

        # Write PID file
        with open(f"./.data/api_pid_{tb_app.id}", "w") as f:
            f.write(str(os.getpid()))

        return app

    if threaded is None:
        return create_bottle_app()
    elif threaded:
        threading.Thread(target=create_bottle_app().run, kwargs=kwargs, daemon=True).run()
    else:
        create_bottle_app().run(**kwargs)
