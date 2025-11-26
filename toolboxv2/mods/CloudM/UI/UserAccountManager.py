# toolboxv2/mods/CloudM/UI/UserAccountManager.py
"""
User Account Manager with Clerk Integration
Handles user settings, profile management, and account operations
"""

from dataclasses import asdict
from typing import Optional

from toolboxv2 import App, RequestData, Result, get_app

Name = 'CloudM.UI.UserAccountManager'
export = get_app(f"{Name}.Export").tb
version = '0.1.0'


# =================== Helper Functions ===================

async def get_current_user_from_request(app: App, request: RequestData):
    """
    Get current user from Clerk session in request.
    Returns LocalUserData from AuthClerk module.
    """
    if not request or not hasattr(request, 'session') or not request.session:
        app.logger.warning("No session found in request for UAM")
        return None

    # Try to get user identifier from session
    user_identifier = None

    # Check for Clerk user ID first
    if hasattr(request.session, 'clerk_user_id') and request.session.clerk_user_id:
        user_identifier = request.session.clerk_user_id
    elif hasattr(request.session, 'user_id') and request.session.user_id:
        user_identifier = request.session.user_id
    elif hasattr(request.session, 'user_name') and request.session.user_name:
        # Legacy: Try to decode username if encoded
        username_c = request.session.user_name
        if username_c and username_c != "Cud be ur name":
            if hasattr(app.config_fh, 'decode_code'):
                decoded = app.config_fh.decode_code(username_c)
                user_identifier = decoded if decoded else username_c
            else:
                user_identifier = username_c

    if not user_identifier:
        app.logger.debug("No valid user identifier found in session")
        return None

    # Load user data from AuthClerk
    try:
        from ..AuthClerk import load_local_user_data, _db_load_user_sync_data, LocalUserData

        # Try to load local data first
        local_data = load_local_user_data(user_identifier)

        if local_data:
            return local_data

        # Try to load from database
        db_data = _db_load_user_sync_data(app, user_identifier)
        if db_data:
            return LocalUserData.from_dict(db_data)

        return None

    except ImportError:
        # Fallback: Try legacy AuthManager
        app.logger.debug("AuthClerk not available, trying legacy AuthManager")
        try:
            from ..AuthManager import get_user_by_name
            from ..types import User

            user_result = await app.a_run_any(
                'CloudM.AuthManager.get_user_by_name',
                username=user_identifier,
                get_results=True
            )

            if user_result and not user_result.is_error():
                return user_result.get()
        except Exception as e:
            app.logger.error(f"UAM: Error loading user via legacy method: {e}")

        return None
    except Exception as e:
        app.logger.error(f"UAM: Error loading user data: {e}")
        return None


def _save_user_data(app: App, user_data) -> Result:
    """
    Save user data - works with both Clerk and legacy systems.
    """
    try:
        # Try Clerk method first
        from ..AuthClerk import save_local_user_data, _db_save_user_sync_data
        import time

        if hasattr(user_data, 'to_dict'):
            # LocalUserData from Clerk
            user_data.last_sync = time.time()
            save_local_user_data(user_data)
            _db_save_user_sync_data(app, user_data.clerk_user_id, user_data.to_dict())
            return Result.ok("User data saved")
        else:
            # Legacy User object
            from ..AuthManager import db_helper_save_user
            return db_helper_save_user(app, asdict(user_data))

    except ImportError:
        # Fallback to legacy
        from ..AuthManager import db_helper_save_user
        return db_helper_save_user(app, asdict(user_data))
    except Exception as e:
        return Result.default_internal_error(f"Failed to save user data: {e}")


# =================== API Endpoints ===================

@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, row=True, level=1)
async def update_email(app: App, request: RequestData, new_email: str):
    """
    Update user email - with Clerk, this redirects to Clerk profile.
    """
    user = await get_current_user_from_request(app, request)

    if not user:
        return """
            <div class="tb-alert tb-alert-error">
                <p>Error: User not authenticated or found.</p>
            </div>
            <input type="email" name="new_email" value="" class="tb-input tb-border tb-p-1 tb-my-1" disabled>
            <button class="tb-btn tb-btn-disabled" disabled>Update Email</button>
        """

    # With Clerk, email changes should go through Clerk's UI
    current_email = getattr(user, 'email', '') or ''

    return f"""
        <div class="tb-mb-2">
            <p><strong>Current Email:</strong> {current_email}</p>
            <p class="tb-text-sm tb-text-muted tb-mt-2">
                Email changes are managed through your Clerk profile for security.
            </p>
            <button onclick="window.TB?.user?.getClerkInstance()?.openUserProfile()"
                    class="tb-btn tb-btn-secondary tb-mt-2">
                <span class="material-symbols-outlined tb-mr-1">settings</span>
                Open Profile Settings
            </button>
        </div>
    """


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, row=True, level=1)
async def update_setting(app: App, request: RequestData, setting_key: str, setting_value: str):
    """
    Update a user setting.
    Settings are stored in local BlobFile and synced to database.
    """
    user = await get_current_user_from_request(app, request)

    # Get target ID for HTMX response
    target_id_suffix = "default"
    if request and hasattr(request, 'request') and hasattr(request.request, 'headers'):
        hx_trigger = getattr(request.request.headers, 'hx_trigger', None)
        if hx_trigger:
            target_id_suffix = hx_trigger.split("-")[-1] if "-" in hx_trigger else hx_trigger

    if not user:
        return "<div class='tb-alert tb-alert-error'>Error: User not authenticated.</div>"

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
    if hasattr(user, 'settings'):
        if user.settings is None:
            user.settings = {}
        user.settings[setting_key] = actual_value
    else:
        # Legacy User object might use different attribute
        if not hasattr(user, 'settings') or user.settings is None:
            user.settings = {}
        user.settings[setting_key] = actual_value

    # Save
    save_result = _save_user_data(app, user)

    if save_result.is_error():
        return f"""
            <div class="tb-alert tb-alert-error">
                Error saving setting {setting_key}: {save_result.info.help_text if hasattr(save_result.info, 'help_text') else str(save_result.info)}
            </div>
        """

    # Return success HTML for specific settings
    if setting_key == "experimental_features":
        is_checked = "checked" if actual_value else ""
        next_value = "false" if actual_value else "true"
        return f"""
            <label class="tb-label tb-flex tb-items-center tb-cursor-pointer">
                <input type="checkbox" name="experimental_features_val" {is_checked}
                       data-hx-post="/api/{Name}/update_setting"
                       data-hx-vals='{{"setting_key": "experimental_features", "setting_value": "{next_value}"}}'
                       data-hx-target="#setting-experimental-features-{target_id_suffix}"
                       data-hx-swap="innerHTML"
                       class="tb-checkbox tb-mr-2">
                <span class="tb-text-sm">Enable Experimental Features</span>
            </label>
            <span class="tb-text-success tb-ml-2 tb-text-xs">✓ Saved!</span>
        """

    return f"<div class='tb-text-success'>Setting '{setting_key}' updated to '{actual_value}'</div>"


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, row=False)
async def get_current_user_api(app: App, request: RequestData):
    """
    API endpoint to get current user data.
    Used by frontend (tbjs) to display user information.
    """
    user = await get_current_user_from_request(app, request)

    if not user:
        return Result.default_user_error(
            info="User not authenticated or found.",
            data=None,
            exec_code=401
        )

    # Build public user data dict
    if hasattr(user, 'to_dict'):
        # LocalUserData from Clerk
        user_dict = user.to_dict()
        pub_user_data = {
            "clerk_user_id": user_dict.get("clerk_user_id"),
            "username": user_dict.get("username"),
            "email": user_dict.get("email"),
            "level": user_dict.get("level", 1),
            "settings": user_dict.get("settings", {}),
            "mod_data": user_dict.get("mod_data", {})
        }
    else:
        # Legacy User object
        user_dict = asdict(user) if hasattr(user, '__dataclass_fields__') else {}
        pub_user_data = {
            "username": getattr(user, 'name', None),
            "email": getattr(user, 'email', None),
            "level": getattr(user, 'level', 1),
            "settings": getattr(user, 'settings', {}),
            "is_persona": getattr(user, 'is_persona', False)
        }

    return Result.ok(data=pub_user_data)


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, row=False)
async def update_mod_data(app: App, request: RequestData, mod_name: str, data: dict):
    """
    Update mod-specific data for the current user.
    """
    user = await get_current_user_from_request(app, request)

    if not user:
        return Result.default_user_error(info="User not authenticated", exec_code=401)

    try:
        # Update mod data
        if hasattr(user, 'mod_data'):
            if user.mod_data is None:
                user.mod_data = {}
            if mod_name not in user.mod_data:
                user.mod_data[mod_name] = {}
            user.mod_data[mod_name].update(data)
        else:
            # Legacy: Store in settings
            if user.settings is None:
                user.settings = {}
            if 'mod_data' not in user.settings:
                user.settings['mod_data'] = {}
            if mod_name not in user.settings['mod_data']:
                user.settings['mod_data'][mod_name] = {}
            user.settings['mod_data'][mod_name].update(data)

        # Save
        save_result = _save_user_data(app, user)

        if save_result.is_error():
            return save_result

        mod_data = user.mod_data.get(mod_name, {}) if hasattr(user, 'mod_data') else user.settings.get('mod_data',
                                                                                                       {}).get(mod_name,
                                                                                                               {})
        return Result.ok(data=mod_data)

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

    if hasattr(user, 'mod_data') and user.mod_data:
        mod_data = user.mod_data.get(mod_name, {})
    elif hasattr(user, 'settings') and user.settings:
        mod_data = user.settings.get('mod_data', {}).get(mod_name, {})
    else:
        mod_data = {}

    return Result.ok(data=mod_data)


@export(mod_name=Name, version=version, request_as_kwarg=False)
async def get_account_management_section_html(app: App, user, WidgetID: str) -> str:
    """
    Generate HTML for the account management section in the dashboard.
    Works with both Clerk and legacy auth systems.
    """
    # Determine user attributes based on type
    if hasattr(user, 'to_dict'):
        # LocalUserData from Clerk
        user_email = user.email or "Not set"
        user_level = user.level or 1
        user_settings = user.settings or {}
        username = user.username
        is_clerk = True
    else:
        # Legacy User object
        user_email = getattr(user, 'email', None) or "Not set"
        user_level = getattr(user, 'level', 1)
        user_settings = getattr(user, 'settings', {}) or {}
        username = getattr(user, 'name', 'Unknown')
        is_clerk = False

    # Email Management Section
    email_section_id = f"email-value-updater-{WidgetID}"

    if is_clerk:
        # With Clerk, email changes go through Clerk profile
        email_section = f"""
            <div class="tb-mb-4">
                <h4 class="tb-text-md tb-font-semibold tb-mb-1">Email Address</h4>
                <div id="{email_section_id}">
                    <p><strong>Current:</strong> {user_email}</p>
                    <p class="tb-text-sm tb-text-muted tb-mt-1">
                        Managed via Clerk profile settings
                    </p>
                    <button onclick="window.TB?.user?.getClerkInstance()?.openUserProfile()"
                            class="tb-btn tb-btn-secondary tb-btn-sm tb-mt-2">
                        Open Profile Settings
                    </button>
                </div>
            </div>
        """
    else:
        # Legacy: Direct email update
        email_section = f"""
            <div class="tb-mb-4">
                <h4 class="tb-text-md tb-font-semibold tb-mb-1">Email Address</h4>
                <div id="{email_section_id}">
                    <p><strong>Current:</strong> {user_email}</p>
                    <input type="email" name="new_email" value="{user_email if user_email != 'Not set' else ''}"
                           class="tb-input tb-border tb-p-1 tb-my-1 tb-w-full sm:tb-w-auto">
                    <button data-hx-post="/api/{Name}/update_email"
                            data-hx-include="[name='new_email']"
                            data-hx-target="#{email_section_id}"
                            data-hx-swap="innerHTML"
                            class="tb-btn tb-btn-primary tb-btn-sm">
                        Update Email
                    </button>
                </div>
            </div>
        """

    # User Level Section
    user_level_section = f"""
        <div class="tb-mb-4">
            <h4 class="tb-text-md tb-font-semibold tb-mb-1">User Level</h4>
            <p class="tb-text-sm">{user_level}</p>
        </div>
    """

    # Settings Section
    setting_experimental_features_id = f"setting-experimental-features-{WidgetID}"
    experimental_features_checked = "checked" if user_settings.get("experimental_features", False) else ""
    next_value = "false" if user_settings.get("experimental_features", False) else "true"

    settings_section = f"""
        <div class="tb-mb-4">
            <h4 class="tb-text-md tb-font-semibold tb-mb-1">Application Settings</h4>
            <div id="{setting_experimental_features_id}">
                <label class="tb-label tb-flex tb-items-center tb-cursor-pointer">
                    <input type="checkbox" name="experimental_features_val" {experimental_features_checked}
                           data-hx-post="/api/{Name}/update_setting"
                           data-hx-vals='{{"setting_key": "experimental_features", "setting_value": "{next_value}"}}'
                           data-hx-target="#{setting_experimental_features_id}"
                           data-hx-swap="innerHTML"
                           class="tb-checkbox tb-mr-2">
                    <span class="tb-text-sm">Enable Experimental Features</span>
                </label>
            </div>
        </div>
    """

    # Sign Out Section
    signout_section = f"""
        <div class="tb-mb-4 tb-border-t tb-pt-4">
            <button onclick="window.TB?.user?.signOut().then(() => window.location.href='/web/assets/login.html')"
                    class="tb-btn tb-btn-danger">
                <span class="material-symbols-outlined tb-mr-1">logout</span>
                Sign Out
            </button>
        </div>
    """

    return f"""
        <div class="tb-p-3 tb-border tb-rounded tb-bg-surface tb-mt-4">
            <h3 class="tb-text-lg tb-font-bold tb-mb-3">Account Management</h3>
            {email_section}
            {user_level_section}
            {settings_section}
            {signout_section}
        </div>
    """
