"""
ToolBox V2 - User Account Manager
Provides account management endpoints for the web dashboard with Clerk integration
"""

from toolboxv2 import App, RequestData, Result, get_app

Name = 'CloudM.UserAccountManager'
export = get_app(f"{Name}.Export").tb
version = '0.1.0'


async def get_current_user_from_request(app: App, request: RequestData):
    """
    Get current user from Clerk session in request.
    Returns user data from local BlobFile storage.
    """
    if not request or not hasattr(request, 'session') or not request.session:
        app.logger.warning("No session found in request for UAM")
        return None

    # Try to get Clerk user ID from session
    # The session might have user_name or clerk_user_id depending on how it's set
    user_identifier = None

    if hasattr(request.session, 'clerk_user_id'):
        user_identifier = request.session.clerk_user_id
    elif hasattr(request.session, 'user_name'):
        user_identifier = request.session.user_name
    elif hasattr(request.session, 'user_id'):
        user_identifier = request.session.user_id

    if not user_identifier or user_identifier == "Cud be ur name":
        return None

    # Load user data from AuthClerk
    try:
        from .AuthClerk import load_local_user_data, _db_load_user_sync_data

        # Try to load local data
        local_data = load_local_user_data(user_identifier)

        if local_data:
            return local_data

        # Try to load from database
        db_data = _db_load_user_sync_data(app, user_identifier)
        if db_data:
            from .AuthClerk import LocalUserData
            return LocalUserData.from_dict(db_data)

        return None

    except Exception as e:
        app.logger.error(f"UAM: Error loading user data: {e}")
        return None


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, row=False)
async def get_current_user(app: App, request: RequestData):
    """
    API endpoint to get current user data.
    Used by frontend to display user information.
    """
    user = await get_current_user_from_request(app, request)

    if not user:
        return Result.default_user_error(
            info="User not authenticated or found.",
            data=None,
            exec_code=401
        )

    # Return public user data
    user_data = {
        "clerk_user_id": user.clerk_user_id,
        "username": user.username,
        "email": user.email,
        "level": user.level,
        "settings": user.settings,
        "mod_data": user.mod_data
    }

    return Result.ok(data=user_data)


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, row=True)
async def update_email(app: App, request: RequestData, new_email: str):
    """
    Update user email address.
    Note: With Clerk, email changes should go through Clerk's UI/API.
    This is kept for compatibility but redirects to Clerk.
    """
    user = await get_current_user_from_request(app, request)

    if not user:
        return """
            <p class='text-red-500'>Error: User not authenticated.</p>
            <p class='text-yellow-500'>Please use Clerk's profile settings to update your email.</p>
        """

    # With Clerk, email changes are managed through Clerk's dashboard
    return f"""
        <div class='text-yellow-500'>
            <p><strong>Current Email:</strong> {user.email}</p>
            <p class='text-sm mt-2'>
                Email changes are managed through Clerk's security settings.
                <a href='#' onclick='window.TB?.user?.getClerkInstance()?.openUserProfile()'
                   class='text-blue-500 underline'>
                    Open Profile Settings
                </a>
            </p>
        </div>
    """


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, row=True)
async def update_setting(app: App, request: RequestData, setting_key: str, setting_value: str):
    """
    Update a user setting.
    Settings are stored in local BlobFile and synced to database.
    """
    user = await get_current_user_from_request(app, request)

    if not user:
        return "<div class='text-red-500'>Error: User not authenticated.</div>"

    # Parse value
    if setting_value.lower() == 'true':
        actual_value = True
    elif setting_value.lower() == 'false':
        actual_value = False
    elif setting_value.isdigit():
        actual_value = int(setting_value)
    else:
        actual_value = setting_value

    # Update settings
    if user.settings is None:
        user.settings = {}

    user.settings[setting_key] = actual_value

    # Save
    try:
        from .AuthClerk import save_local_user_data, _db_save_user_sync_data
        import time

        user.last_sync = time.time()
        save_local_user_data(user)
        _db_save_user_sync_data(app, user.clerk_user_id, user.to_dict())

        # Return success HTML for HTMX
        if setting_key == "experimental_features":
            is_checked = "checked" if actual_value else ""
            return f"""
                <label class="tb-label tb-flex tb-items-center tb-cursor-pointer">
                    <input type="checkbox" name="exp_features_val" {is_checked}
                           data-hx-post="/api/{Name}/update_setting"
                           data-hx-vals='{{"setting_key": "experimental_features", "setting_value": "{'false' if actual_value else 'true'}"}}'
                           data-hx-target="closest div" data-hx-swap="innerHTML"
                           class="tb-checkbox tb-mr-2">
                    Enable Experimental Features
                </label>
                <span class='text-green-500 ml-2 text-xs'>Saved!</span>
            """

        return f"<div class='text-green-500'>Setting '{setting_key}' updated to '{actual_value}'</div>"

    except Exception as e:
        return f"<div class='text-red-500'>Error saving setting: {str(e)}</div>"


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, row=False)
async def update_mod_data(app: App, request: RequestData, mod_name: str, data: dict):
    """
    Update mod-specific data for the current user.
    """
    user = await get_current_user_from_request(app, request)

    if not user:
        return Result.default_user_error(info="User not authenticated", exec_code=401)

    try:
        from .AuthClerk import save_local_user_data, _db_save_user_sync_data
        import time

        # Update mod data
        if user.mod_data is None:
            user.mod_data = {}

        if mod_name not in user.mod_data:
            user.mod_data[mod_name] = {}

        user.mod_data[mod_name].update(data)
        user.last_sync = time.time()

        # Save
        save_local_user_data(user)
        _db_save_user_sync_data(app, user.clerk_user_id, user.to_dict())

        return Result.ok(data=user.mod_data[mod_name])

    except Exception as e:
        return Result.default_internal_error(f"Error updating mod data: {str(e)}")


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, row=False)
async def get_mod_data(app: App, request: RequestData, mod_name: str):
    """
    Get mod-specific data for the current user.
    """
    user = await get_current_user_from_request(app, request)

    if not user:
        return Result.default_user_error(info="User not authenticated", exec_code=401)

    mod_data = user.mod_data.get(mod_name, {}) if user.mod_data else {}
    return Result.ok(data=mod_data)


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, row=True)
async def get_account_section_html(app: App, request: RequestData):
    """
    Get HTML for the account management section in the dashboard.
    Uses Clerk's user management UI.
    """
    user = await get_current_user_from_request(app, request)

    if not user:
        return """
            <div class="tb-card tb-p-4">
                <h3 class="tb-text-lg tb-font-semibold tb-mb-4">Account Settings</h3>
                <p class="text-yellow-500">Please sign in to view account settings.</p>
                <button onclick="window.TB?.user?.signIn()" class="tb-btn tb-btn-primary tb-mt-4">
                    Sign In
                </button>
            </div>
        """

    return f"""
        <div class="tb-card tb-p-4">
            <h3 class="tb-text-lg tb-font-semibold tb-mb-4">Account Settings</h3>

            <div class="tb-space-y-4">
                <!-- User Info -->
                <div class="tb-border-b tb-pb-4">
                    <p><strong>Username:</strong> {user.username}</p>
                    <p><strong>Email:</strong> {user.email}</p>
                    <p><strong>Level:</strong> {user.level}</p>
                </div>

                <!-- Clerk Profile Button -->
                <div>
                    <button onclick="window.TB?.user?.getClerkInstance()?.openUserProfile()"
                            class="tb-btn tb-btn-secondary">
                        Open Clerk Profile Settings
                    </button>
                </div>

                <!-- Settings -->
                <div class="tb-border-t tb-pt-4">
                    <h4 class="tb-font-semibold tb-mb-2">Application Settings</h4>

                    <div id="setting-experimental" class="tb-mb-2">
                        <label class="tb-label tb-flex tb-items-center tb-cursor-pointer">
                            <input type="checkbox"
                                   {'checked' if user.settings.get('experimental_features') else ''}
                                   data-hx-post="/api/{Name}/update_setting"
                                   data-hx-vals='{{"setting_key": "experimental_features", "setting_value": "{'false' if user.settings.get('experimental_features') else 'true'}"}}'
                                   data-hx-target="closest div"
                                   data-hx-swap="innerHTML"
                                   class="tb-checkbox tb-mr-2">
                            Enable Experimental Features
                        </label>
                    </div>
                </div>

                <!-- Sign Out -->
                <div class="tb-border-t tb-pt-4">
                    <button onclick="window.TB?.user?.signOut()" class="tb-btn tb-btn-danger">
                        Sign Out
                    </button>
                </div>
            </div>
        </div>
    """
