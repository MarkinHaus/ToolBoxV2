# toolboxv2/mods/CloudM/UI/widget.py

from toolboxv2 import App, RequestData, get_app, Result
from toolboxv2.mods.CloudM.AuthClerk import LocalUserData

Name = 'CloudM.UI.widget'
export = get_app(f"{Name}.Export").tb
version = '0.0.1'  # Keep version consistent or update as needed


# Most of the old widget loading logic (load_root_widget, reload_widget_main, reload_widget_info, etc.)
# is now obsolete as the new Admin Dashboard is a single, more complex SPA-like page
# served by CloudM.AdminDashboard.get_dashboard_main_page.
@export(mod_name=Name, version=version, name="test_get_widget", test=True)
async def test_get_widget(app: App | None = None, **kwargs):
    if app is None:
        app = get_app(from_=f"{Name}.get_widget")

    request = kwargs.get('request')
    if request is None:
        request = {'request':
        {'content_type': '',
        'headers': {'accept':
                        'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                    'accept-encoding': 'gzip, deflate, br, zstd',
                    'accept-language': 'de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7', 'cache-control': 'max-age=0',
                    'connection': 'keep-alive','host': 'localhost:8080', 'sec-fetch-dest': 'document',
                    'sec-fetch-mode': 'navigate', 'sec-fetch-site': 'none', 'sec-fetch-user': '?1',
                    'upgrade-insecure-requests':'1',
                    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36',
                    'sec-ch-ua': '"Chromium";v="136", "Google Chrome";v="136",'
                                 ' "Not.A/Brand";v="99"', 'sec-ch-ua-mobile': '?0',
                    'sec-ch-ua-platform': '"Windows"'},
         'method': 'GET', 'path': '/api/CloudM.UI.widget/get_widget', 'query_params': {}},
                   'session': {'SiID': '7a87978efc6ea68385a1ab606273bcc8c491fc70516cccb42419d211de556e7f',
                               'level': '1', 'spec': 'bab0cbf7978cec9570a5ee78e4182b490e0b8bb09433cd3accfbe851bfa38c04',
                               'user_name': 'root'}, 'session_id': '0xce7cd7b545b5e92b'}

    return await get_widget(app, request=RequestData.from_dict(request))



# The primary entry point `get_widget` will now delegate to the AdminDashboard's main page.
# toolboxv2/mods/CloudM/UI/widget.py
# ... (imports) ...
@export(mod_name=Name, version=version, request_as_kwarg=True, level=1, api=True, name="get_widget")
async def get_widget(app: App | None = None, request: RequestData = None, **kwargs):
    if app is None: app = get_app(from_=Name + ".get_widget")

    from ..UserAccountManager import get_current_user_from_request as get_user_for_widget

    user: LocalUserData = await get_user_for_widget(app, request)
    if user:
        if user.username == 'root' or user.username == 'loot':  # Admin
            from ..AdminDashboard import Name as AdminDashboard_ModuleName
            dashboard_result = await app.a_run_any(
                AdminDashboard_ModuleName + ".main", request=request, get_results=True
            )
        else:  # Regular User
            from ..UserDashboard import Name as UserDashboard_ModuleName
            dashboard_result = await app.a_run_any(
                UserDashboard_ModuleName + ".main", request=request, get_results=True
            )

        if dashboard_result.is_error():
            return dashboard_result  # Or some error HTML
        return dashboard_result  # Should be HTML string
    else:
        # No user logged in - redirect to login page or show guest content
        # For now, let tbjs on the client handle redirect if session is invalid
        # This might serve a very basic lander page with login prompt
        # Or if tbjs router handles unauth routes to login, this could be an empty shell.
        # For simplicity here, returning a basic prompt.
        # A robust solution would involve TB.router on client side.
        login_url = "/login.html"  # Placeholder, should come from config or TB.config
        # This assumes tbjs is already loaded and will handle routing/state if not logged in.
        # If this endpoint is hit directly without auth, it might show this.
        # Better: The endpoint serving this widget should require auth, and tbjs handles redirects.
        # For now, if UserDashboard.main itself checks auth and tbjs handles it, this is fine.
        # This could also be an API call that returns JSON which tbjs uses to decide to route to login.
        # But since get_widget is row=True, it expects HTML.

        # Fallback to UserDashboard which itself checks auth and might return an error/redirect.
        from ..UserDashboard import Name as UserDashboard_ModuleName
        dashboard_result = await app.a_run_any(
            UserDashboard_ModuleName + ".main", request=request, get_results=True
        )  # It will return 401 if no user.
        return dashboard_result if not dashboard_result.is_error() else Result.html("<p>Please log in.</p>")
# Removed functions:
# - load_root_widget
# - reload_widget_main
# - reload_widget_info (its functionality is now part of AdminDashboard JS and My Account)
# - reload_widget_system
# - load_widget
# - get_user_from_request (UserAccountManager.get_current_user_from_request is preferred)
# - removed (device removal will be part of My Account in AdminDashboard JS)
# - info (tab content now part of AdminDashboard JS sections)
# - Other specific widget interaction endpoints (danger, stop, reset, etc.) unless they have a new role.

# Minimal TBEF usage for now, as most is delegated.
