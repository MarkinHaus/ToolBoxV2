# toolboxv2/mods/CloudM/UI/user_account_manager.py

import uuid
from dataclasses import asdict


from toolboxv2 import TBEF, App, Result, get_app, RequestData
from toolboxv2.mods.CloudM.AuthManager import db_helper_save_user  # Assuming AuthManager functions are accessible
from ..types import User  # From toolboxv2/mods/CloudM/types.py

Name = 'CloudM.UI.UserAccountManager'
export = get_app(f"{Name}.Export").tb
version = '0.0.1'


# Helper to get current user from request
async def get_current_user_from_request(app: App, request: RequestData) -> User | None:
    if not request or not hasattr(request, 'session') or not request.session:
        app.print("No session found in request for UAM", level="WARNING")
        return None

    username_c = request.session.user_name
    if not username_c or username_c == "Cud be ur name":
        app.print(f"No valid user_name in session live_data for UAM: {username_c}", level="WARNING")
        return None

    decoded_username = app.config_fh.decode_code(username_c)
    if not decoded_username:
        app.print(f"Failed to decode username_c for UAM: {username_c}", level="WARNING")
        return None

    user_result = await app.a_run_any(TBEF.CLOUDM_AUTHMANAGER.GET_USER_BY_NAME, username=decoded_username)
    if user_result.is_error() or not user_result.get():
        app.print(f"UAM: Failed to get user by name '{decoded_username}': {user_result.info}", level="ERROR")
        return None

    retrieved_user = user_result.get()
    if not isinstance(retrieved_user, User):
        app.print(f"UAM: Retrieved data for '{decoded_username}' is not a User instance.", level="ERROR")
        return None

    return retrieved_user


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, row=True, level=1)
async def update_email(app: App, request: RequestData, new_email: str):
    user = await get_current_user_from_request(app, request)
    if not user:
        return """
            <p class='text-red-500'>Error: User not authenticated or found.</p>
            <input type="email" name="new_email" value="" class="border p-1 my-1">
            <button class="bg-gray-300 text-white p-1 text-sm" disabled>Update Email</button>
        """

    if not new_email or "@" not in new_email:  # Basic validation
        return f"""
            <p><strong>Email:</strong> {user.email if user.email else "Not set"} <span class='text-red-500 ml-2'>Invalid email format.</span></p>
            <input type="email" name="new_email" value="{new_email}" class="border p-1 my-1 border-red-500">
            <button hx-post="/api/{Name}/update_email" hx-include="[name='new_email']"
                    hx-target="closest div" hx-swap="innerHTML" class="bg-blue-500 text-white p-1 text-sm">Update Email</button>
        """

    user.email = new_email
    save_result = db_helper_save_user(app, asdict(user))
    if save_result.is_error():
        return f"""
            <p><strong>Email:</strong> {user.email if user.email else "Not set"} <span class='text-red-500 ml-2'>Error saving: {save_result.info.message}.</span></p>
            <input type="email" name="new_email" value="{user.email}" class="border p-1 my-1">
            <button hx-post="/api/{Name}/update_email" hx-include="[name='new_email']"
                    hx-target="closest div" hx-swap="innerHTML" class="bg-blue-500 text-white p-1 text-sm">Update Email</button>
        """

    return f"""
        <p><strong>Email:</strong> {user.email} <span class='text-green-500 ml-2'>Updated!</span></p>
        <input type="email" name="new_email" value="{user.email}" class="border p-1 my-1">
        <button hx-post="/api/{Name}/update_email" hx-include="[name='new_email']"
                hx-target="closest div" hx-swap="innerHTML" class="bg-blue-500 text-white p-1 text-sm">Update Email</button>
    """


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, row=True, level=1)
async def update_setting(app: App, request: RequestData, setting_key: str, setting_value: str):
    user = await get_current_user_from_request(app, request)
    target_id_suffix = request.request.headers.hx_trigger  # To keep unique IDs if multiple widgets
    if target_id_suffix:
        target_id_suffix = target_id_suffix.split("-")[-1]
    if not user:
        return f"<div class='text-red-500'>Error: User not authenticated or found.</div>"

    if setting_key == "experimental_features":
        actual_value = setting_value.lower() == 'true'
    else:
        actual_value = setting_value

    if user.settings is None:
        user.settings = {}

    user.settings[setting_key] = actual_value
    save_result = db_helper_save_user(app, asdict(user))

    if save_result.is_error():
        return f"""
            <label class="flex items-center text-red-500">
                <input type="checkbox" name="experimental_features_val" class="mr-2">
                Error saving setting {setting_key}: {save_result.info.message}
            </label>
        """

    if setting_key == "experimental_features":
        is_checked = "checked" if actual_value else ""
        return f"""
            <label class="flex items-center">
                <input type="checkbox" name="experimental_features_val" {is_checked}
                       hx-post="/api/{Name}/update_setting"
                       hx-vals='{{"setting_key": "experimental_features", "setting_value": event.target.checked ? "true" : "false"}}'
                       hx-target="#setting-experimental-features-{target_id_suffix}" hx-swap="innerHTML" class="mr-2">
                Enable Experimental Features
            </label>
            <span class='text-green-500 ml-2 text-xs'>Saved!</span>
        """
    return f"<div class='text-green-500'>Setting {setting_key} updated.</div>"


@export(mod_name=Name, version=version, request_as_kwarg=False)  # Not an API endpoint itself
async def get_account_management_section_html(app: App, user: User, WidgetID: str) -> str:
    # Email Management
    email_section_id = f"email-value-updater-{WidgetID}"
    email_section = f"""
        <div class="mb-4">
            <h4 class="text-md font-semibold mb-1">Email Address</h4>
            <div id="{email_section_id}">
                 <p><strong>Current:</strong> {user.email if user.email else "Not set"}</p>
                 <input type="email" name="new_email" value="{user.email if user.email else ''}" class="border p-1 my-1 w-full sm:w-auto">
                 <button hx-post="/api/{Name}/update_email" hx-include="[name='new_email']"
                         hx-target="#{email_section_id}" hx-swap="innerHTML" class="bg-blue-500 hover:bg-blue-600 text-white p-1 text-sm rounded">Update Email</button>
            </div>
        </div>
    """

    persona_status_id = f"persona-status-{WidgetID}"
    persona_button_html = ""
    if not user.is_persona:
        persona_button_html = f"""
            <button onclick="handleRegisterPersona_{WidgetID}()" class="bg-green-500 hover:bg-green-600 text-white p-1 text-sm rounded">Add Persona Device (WebAuthn)</button>
            <div id="{persona_status_id}" class="text-sm mt-1"></div>
            <script>
                async function handleRegisterPersona_{WidgetID}() {{
                    const statusDiv = document.getElementById('{persona_status_id}');
                    statusDiv.innerHTML = 'Initiating WebAuthn registration...';
                    if (window.TB && window.TB.user) {{
                        const result = await window.TB.user.registerWebAuthnForCurrentUser('{user.name}');
                        if (result.success) {{
                            statusDiv.innerHTML = '<p class="text-green-500">' + result.message + ' The relevant information will update after a refresh or navigating back to this tab.</p>';
                            // Attempt to refresh the info tab content
                            // setTimeout(() => {{
                            //    const infoTabButton = document.querySelector('button[hx-get$="/info"]'); // Adjust selector if needed
                            //    if(infoTabButton) infoTabButton.click();
                            // }}, 2000);
                        }} else {{
                            statusDiv.innerHTML = '<p class="text-red-500">Error: ' + result.message + '</p>';
                        }}
                    }} else {{
                        statusDiv.innerHTML = '<p class="text-red-500">TB.user library not available.</p>';
                    }}
                }}
            </script>
        """
    else:
        persona_button_html = "<p class='text-sm text-gray-700'>Persona (WebAuthn) is configured for this account.</p>"

    persona_section = f"""
        <div class="mb-4">
            <h4 class="text-md font-semibold mb-1">Persona Device (WebAuthn)</h4>
            {persona_button_html}
        </div>
    """

    user_level_section = f"""
        <div class="mb-4">
            <h4 class="text-md font-semibold mb-1">User Level</h4>
            <p class='text-sm text-gray-700'>{user.level}</p>
        </div>
    """

    log_level_val = getattr(user, 'log_level', 'INFO')  # getattr in case field is missing on older user objects
    log_level_section = f"""
        <div class="mb-4">
            <h4 class="text-md font-semibold mb-1">Log Level Preference</h4>
            <p class='text-sm text-gray-700'>{log_level_val}</p>
            <!-- Add UI to change log_level if needed in future -->
        </div>
    """

    setting_experimental_features_id = f"setting-experimental-features-{WidgetID}"
    experimental_features_checked = "checked" if user.settings.get("experimental_features", False) else ""
    settings_section = f"""
        <div class="mb-4">
            <h4 class="text-md font-semibold mb-1">Application Settings</h4>
            <div id="{setting_experimental_features_id}">
                <label class="flex items-center cursor-pointer">
                    <input type="checkbox" name="experimental_features_val" {experimental_features_checked}
                           hx-post="/api/{Name}/update_setting"
                           hx-vals='{{"setting_key": "experimental_features", "setting_value": event.target.checked ? "true" : "false"}}'
                           hx-target="#{setting_experimental_features_id}" hx-swap="innerHTML" class="mr-2">
                    <span class='text-sm text-gray-700'>Enable Experimental Features</span>
                </label>
            </div>
        </div>
    """

    return f"""
        <div class="p-3 border rounded bg-gray-50 mt-4">
            <h3 class="text-lg font-bold mb-3">Account Management</h3>
            {email_section}
            {persona_section}
            {user_level_section}
            {log_level_section}
            {settings_section}
        </div>
    """
