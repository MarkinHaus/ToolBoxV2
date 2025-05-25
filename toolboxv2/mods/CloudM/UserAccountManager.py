# toolboxv2/mods/CloudM/UI/user_account_manager.py

import uuid
from dataclasses import asdict

from toolboxv2 import TBEF, App, Result, get_app, RequestData
from toolboxv2.mods.CloudM.AuthManager import db_helper_save_user  # Assuming AuthManager functions are accessible
from .types import User  # From toolboxv2/mods/CloudM/types.py

Name = 'CloudM.UserAccountManager'
export = get_app(f"{Name}.Export").tb
version = '0.0.1'


# Helper to get current user from request
async def get_current_user_from_request(app: App, request: RequestData) -> User | None:
    if not request or not hasattr(request, 'session') or not request.session:
        app.logger.warning("No session found in request for UAM")
        return None

    username_c = request.session.user_name
    if not username_c or username_c == "Cud be ur name":  # "Cud be ur name" is a default/guest
        # app.print(f"No valid user_name in session for UAM: {username_c}", level="DEBUG")
        return None

    # No need to decode here if session.user_name is already the plain username
    # If session.user_name is still encoded from an older system part, then decode
    # Assuming session.user_name IS the actual username
    decoded_username = username_c
    # if app.config_fh.is_encoded(username_c): # Hypothetical check
    #    decoded_username = app.config_fh.decode_code(username_c)
    #    if not decoded_username:
    #        app.print(f"Failed to decode username_c for UAM: {username_c}", level="WARNING")
    #        return None

    if not decoded_username:  # Should not happen if username_c is valid
        return None

    user_result = await app.a_run_any(TBEF.CLOUDM_AUTHMANAGER.GET_USER_BY_NAME, username=decoded_username, get_results=True)
    if user_result.is_error() or not user_result.get():
        app.logger.warning(f"UAM: Failed to get user by name '{decoded_username}': {user_result.info}")
        return None

    retrieved_user = user_result.get()
    #if not hasattr(retrieved_user, 'user_pass_pub_persona'):
    #    app.logger.warning(f"UAM: Retrieved data for '{decoded_username}' is not a User instance. is {type(retrieved_user)}")
    #    return None

    return retrieved_user


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, row=True, level=1)
async def update_email(app: App, request: RequestData, new_email: str):
    user = await get_current_user_from_request(app, request)
    if not user:
        # This response is for HTMX. If called from tbjs, it might prefer JSON.
        return """
            <p class='text-red-500'>Error: User not authenticated or found.</p>
            <input type="email" name="new_email_admin" value="" class="tb-input tb-border tb-p-1 tb-my-1">
            <button class="tb-btn tb-btn-disabled" disabled>Update Email</button>
        """

    if not new_email or "@" not in new_email:  # Basic validation
        # For HTMX, returning HTML that includes the input field again
        return f"""
            <p><strong>Current:</strong> {user.email if user.email else "Not set"} <span class='text-red-500 ml-2'>Invalid email format.</span></p>
            <input type="email" name="new_email_admin" value="{new_email}" class="tb-input tb-border-red-500 tb-p-1 tb-my-1">
            <button data-hx-post="/api/{Name}/update_email" data-hx-include="[name='new_email_admin']"
                    data-hx-target="closest div" data-hx-swap="innerHTML" class="tb-btn tb-btn-primary">Update Email</button>
        """

    user.email = new_email
    save_result = db_helper_save_user(app, asdict(user))  # db_helper_save_user is sync

    # Simulate async save if db_helper_save_user were async:
    # save_result = await app.a_run_any(TBEF.CLOUDM_AUTHMANAGER.DB_HELPER_SAVE_USER_ASYNC_WRAPPER, user_data=asdict(user))

    if save_result.is_error():
        return f"""
            <p><strong>Current:</strong> {user.email if user.email else "Not set"} <span class='text-red-500 ml-2'>Error saving: {save_result.info}.</span></p>
            <input type="email" name="new_email_admin" value="{user.email}" class="tb-input tb-border tb-p-1 tb-my-1">
            <button data-hx-post="/api/{Name}/update_email" data-hx-include="[name='new_email_admin']"
                    data-hx-target="closest div" data-hx-swap="innerHTML" class="tb-btn tb-btn-primary">Update Email</button>
        """

    return f"""
        <p><strong>Current:</strong> {user.email} <span class='text-green-500 ml-2'>Updated!</span></p>
        <input type="email" name="new_email_admin" value="{user.email}" class="tb-input tb-border tb-p-1 tb-my-1">
        <button data-hx-post="/api/{Name}/update_email" data-hx-include="[name='new_email_admin']"
                data-hx-target="closest div" data-hx-swap="innerHTML" class="tb-btn tb-btn-primary">Update Email</button>
    """


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, row=True, level=1)
async def update_setting(app: App, request: RequestData, setting_key: str, setting_value: str):
    user = await get_current_user_from_request(app, request)
    # hx_trigger might not be reliable or always present if not an HTMX direct call.
    # Use a fixed or uniquely generated ID from the tbjs side if needed, or ensure HTMX context.
    target_id_suffix = request.request.headers.get('HX-Trigger')
    if target_id_suffix:
        target_id_suffix = target_id_suffix.split("-")[-1] if "-" in target_id_suffix else target_id_suffix
    else:
        target_id_suffix = "default"  # Fallback

    if not user:
        return f"<div class='text-red-500'>Error: User not authenticated or found.</div>"  # Basic error for HTMX

    if setting_key == "experimental_features":
        actual_value = setting_value.lower() == 'true'
    else:
        actual_value = setting_value  # Handle other types if necessary

    if user.settings is None:
        user.settings = {}

    user.settings[setting_key] = actual_value
    save_result = db_helper_save_user(app, asdict(user))  # db_helper_save_user is sync

    if save_result.is_error():
        # Return state that reflects error, possibly including the input to retry
        return f"""
            <label class="tb-label tb-flex tb-items-center tb-text-red-500">
                <input type="checkbox" name="exp_features_admin_val" class="tb-checkbox tb-mr-2">
                Error saving setting {setting_key}: {save_result.info}. Retry.
            </label>
        """

    if setting_key == "experimental_features":
        is_checked = "checked" if actual_value else ""
        # Note: hx-target needs to be stable or passed correctly if this widget is reused.
        # For the admin panel, the target_id_suffix logic might need adjustment based on how it's embedded.
        return f"""
            <label class="tb-label tb-flex tb-items-center tb-cursor-pointer">
                <input type="checkbox" name="exp_features_admin_val" {is_checked}
                       data-hx-post="/api/{Name}/update_setting"
                       data-hx-vals='{{"setting_key": "experimental_features", "setting_value": event.target.checked ? "true" : "false"}}'
                       data-hx-target="closest div" data-hx-swap="innerHTML" class="tb-checkbox tb-mr-2">
                Enable Experimental Features
            </label>
            <span class='text-green-500 ml-2 text-xs'>Saved!</span>
        """
    # Fallback for other settings if ever used this way by HTMX
    return f"<div class='text-green-500'>Setting {setting_key} updated. Refresh may be needed.</div>"


@export(mod_name=Name, api=True, version=version, request_as_kwarg=True, row=False)  # row=False to return JSON
async def get_current_user_from_request_api_wrapper(app: App, request: RequestData):
    """ API callable version of get_current_user_from_request for tbjs/admin panel """
    user = await get_current_user_from_request(app, request)
    if not user:
        # Return error that tbjs can handle
        return Result.default_user_error(info="User not authenticated or found.", data=None, exec_code=401)
    return Result.ok(data=asdict(user))

# get_account_management_section_html is now largely obsolete.
# Its functionality will be replicated by client-side JS (tbjs) in the Admin Dashboard,
# calling the API endpoints above (update_email, update_setting, get_current_user_for_persona).
# The Persona registration button logic will also be in tbjs, calling TB.user.registerWebAuthnForCurrentUser.
