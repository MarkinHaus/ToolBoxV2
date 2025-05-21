import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Gmail-Konfiguration
GMAIL_EMAIL = os.environ.get("GOOGLE_APP_EMAIL")
GMAIL_PASSWORD = os.environ.get("GOOGLE_APP_PASSWORD")
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465  # SSL-Port

from toolboxv2 import App, MainTool, Result, get_app, get_logger
from toolboxv2.utils.system.types import ApiResult, ToolBoxError, ToolBoxInterfaces

Name = "email_waiting_list"
version = '0.0.1'
export = get_app("email_waiting_list.email_waiting_list.EXPORT").tb
s_export = export(mod_name=Name, version=version, state=False, test=False)



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
def send_email(subject, body, recipient_emails):
    """
    Sendet eine E-Mail über Gmail SMTP.

    Args:
        data (tuple):
            subject (str): Betreff der E-Mail.
            body (str): Inhalt der E-Mail.
            recipient_emails (list): Liste der Empfänger-E-Mail-Adressen.
    """
    if not isinstance(subject, str):
        if isinstance(subject, tuple) or isinstance(subject, list):
            subject, *_ = subject
            subject += ' '.join(_)
    get_logger().info(f"{subject=}, {recipient_emails=}")
    try:
        msg = MIMEMultipart()
        msg['From'] = GMAIL_EMAIL
        msg['To'] = ', '.join(recipient_emails) if isinstance(recipient_emails, list) else recipient_emails
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(GMAIL_EMAIL, GMAIL_PASSWORD)
            server.sendmail(GMAIL_EMAIL, recipient_emails, msg.as_string())

        return Result.ok(info="E-Mail erfolgreich gesendet. ")
    except Exception as e:
        get_logger().error(f"Fehler beim Senden der E-Mail: {e}")
        return Result.default_internal_error(info=f"Fehler beim Senden der E-Mail {e}")


@s_export
def create_welcome_email(user_email, user_name):
    subject = "Willkommen bei SimpleCore!"
    body = f"Hallo {user_name},\n\nWillkommen bei SimpleCore! Wir freuen uns, dich an Bord zu haben."
    return subject, body, [user_email]


@s_export
def crate_magic_lick_device_email(user_email, user_name, link_id, nl=-1):
    subject = "Welcome to SimpleCore!",
    body = f"<h3>Log in with : <a href=\"https://simplecore.app/web/assets/m_log_in.html?key={link_id}&nl={nl}\">Magic link</a> don't chair!</h3><br/>Must enter ur user name on the next page to log in."
    return subject, body, [user_email]
