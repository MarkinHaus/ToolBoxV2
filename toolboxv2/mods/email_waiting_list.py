from toolboxv2 import Result, get_app, App, MainTool
from toolboxv2.utils.system.types import ToolBoxError, ApiResult, ToolBoxInterfaces
from mailjet_rest import Client
import os

Name = "email_waiting_list"
version = '0.0.0'
export = get_app("email_waiting_list.email_waiting_list.EXPORT").tb
s_export = export(mod_name=Name, version=version, state=False, test=False)
api_key = os.environ.get('MJ_APIKEY_PUBLIC')
api_secret = os.environ.get('MJ_APIKEY_PRIVATE')
mailjet = Client(auth=(api_key, api_secret), version='v3.1')


@export(mod_name=Name, api=True, interface=ToolBoxInterfaces.api, state=True, test=False)
def add(app: App, email: str) -> ApiResult:
    if app is None:
        app = get_app("email_waiting_list")
    # if "db" not in list(app.MOD_LIST.keys()):
    #    return "Server has no database module"
    tb_token_jwt = app.run_any('DB', 'append_on_set', query="email_waiting_list", data=[email], get_results=True)

    # Default response for internal error
    error_type = ToolBoxError.internal_error
    out = "My apologies, unfortunately, you could not be added to the Waiting list."
    tb_token_jwt.print()
    # Check if the email was successfully added to the waiting list
    if not tb_token_jwt.is_error():
        out = "You will receive an invitation email in a few days"
        error_type = ToolBoxError.none
    elif tb_token_jwt.info.exec_code == -4 or tb_token_jwt.info.exec_code == -3:

        app.run_any('DB', 'set', query="email_waiting_list", data=[email], get_results=True)
        out = "You will receive an invitation email in a few days NICE you ar the first on in the list"
        tb_token_jwt.print()
        error_type = ToolBoxError.none

    # Check if the email is already in the waiting list
    elif tb_token_jwt.info.exec_code == -5:
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


@s_export
def send_email(data):
    result = mailjet.send.create(data=data)
    if result.status_code != 0:
        return Result.default_internal_error(exec_code=result.status_code, data=result.json())
    return Result.custom_error(exec_code=result.status_code, data=result.json())


@s_export
def crate_sing_in_email(user_email, user_name):
    return {
        'Messages': [
            {
                "From": {
                    "Email": "MarkinHausDev@gmail.com",
                    "Name": "Me"
                },
                "To": [
                    {
                        "Email": user_email,
                        "Name": user_name
                    }
                ],
                "Subject": "Welcome to SimpleCore!",
                "TextPart": f"Hi {user_name}",
                "HTMLPart": "<h3>Dear passenger 1, welcome to <a href=\"https://www.mailjet.com/\">Mailjet</a>!</h3><br />May the delivery force be with you!"
            }
        ]
    }


@s_export
def crate_magick_lick_device_email(user_email, user_name, link_id, nl=-1):
    return {
        'Messages': [
            {
                "From": {
                    "Email": "MarkinHausDev@gmail.com",
                    "Name": "Me"
                },
                "To": [
                    {
                        "Email": user_email,
                        "Name": user_name
                    }
                ],
                "Subject": "Welcome to SimpleCore!",
                "TextPart": f"Hi {user_name}",
                "HTMLPart": f"<h3>Dear passenger 1, welcome to <a href=\"https://simplecore.app/web/assets/m_log_in.html?key={link_id}&nl={nl}\">Magick Log in link</a> don not chair!</h3><br />Must enter ur user name on the next page to log in"
            }
        ]
    }
