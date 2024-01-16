from toolboxv2 import Result, get_app, App, MainTool
from toolboxv2.utils.types import ToolBoxError, ApiResult, ToolBoxInterfaces

Name = "email_wating_list"
version = '*.*.*'
export = get_app("email_waiting_list.email_waiting_list.EXPORT").tb


@export(mod_name=Name, api=True, interface=ToolBoxInterfaces.api, state=True)
def add(app:App, email: str) -> ApiResult:
    if app is None:
        app = get_app("email_waiting_list")
    # if "db" not in list(app.MOD_LIST.keys()):
    #    return "Server has no database module"
    tb_token_jwt = app.run_any('DB', 'append_on_set', query="email_waiting_list", data=[email], get_results=True)

    # Default response for internal error
    error_type = ToolBoxError.internal_error
    out = "My apologies, unfortunately, you could not be added to the Waiting list."

    # Check if the email was successfully added to the waiting list
    if not tb_token_jwt.is_error():
        out = "You will receive an invitation email in a few days"
        error_type = ToolBoxError.none
    elif not tb_token_jwt.is_data():
        out = "an error accused "
        tb_token_jwt.print()
        error_type = ToolBoxError.custom_error

    # Check if the email is already in the waiting list
    elif "already in list" in tb_token_jwt.get():
        out = "You are already in the list, please do not try to add yourself more than once."
        error_type = ToolBoxError.custom_error

    # Use the return_result function to create and return the Result object
    return MainTool.return_result(
        error=error_type,
        exec_code=0,  # Assuming exec_code 0 for success, modify as needed
        help_text=out,
        data_info="email",
        data={"message": out}
    )


@get_app("email_waiting_list.send_email_to_all.EXPORT").tb()
def send_email_to_all():
    pass
