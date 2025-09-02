import datetime
import os
import smtplib
import uuid
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from urllib.parse import quote  # For URL encoding parameters

from jinja2 import BaseLoader, Environment

# Assuming Code is available from your toolboxv2 installation
try:
    from toolboxv2.utils.security.cryp import Code
except ImportError:
    # Placeholder if Code is not found, adapt as needed
    class Code:
        @staticmethod
        def one_way_hash(data, salt, pepper):
            import hashlib
            hashed = hashlib.sha256((str(data) + str(salt) + str(pepper)).encode()).hexdigest()
            return hashed


    print("Warning: toolboxv2.utils.security.cryp.Code not found, using placeholder.")

from toolboxv2 import App, Result, get_app, get_logger  # MainTool not used directly here
from toolboxv2.utils.system.types import (
    ApiResult,
    ToolBoxInterfaces,  # ToolBoxError, ToolBoxInterfaces not used directly
)

# --- Configuration ---
Name = "CloudM.email_services"  # Renamed to reflect broader scope
version = '0.1.0'

# Gmail Configuration from environment variables
GMAIL_EMAIL = os.getenv("GMAIL_EMAIL")
GMAIL_PASSWORD = os.getenv("GMAIL_PASSWORD")  # Gmail App Password if 2FA is on # https://myaccount.google.com/apppasswords.
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 465))  # 465 for SSL, 587 for STARTTLS

# App specific details from environment variables
APP_NAME = os.getenv("APP_NAME", "SimpleCore")
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:5000")  # Your application's base URL
SENDER_EMAIL_ADDRESS = os.getenv("SENDER_EMAIL", GMAIL_EMAIL)  # Email address to send from

# Visuals (optional, provide fallback)
EMAIL_BG_LIGHT_URL =APP_BASE_URL+ os.getenv("EMAIL_BG_LIGHT_URL", "")  # e.g., "https://example.com/bg_light.png"
EMAIL_BG_DARK_URL =APP_BASE_URL+ os.getenv("EMAIL_BG_DARK_URL", "")  # e.g., "https://example.com/bg_dark.png"
EMAIL_LOGO_URL = APP_BASE_URL+ os.getenv("EMAIL_LOGO_URL", "")  # e.g., "https://example.com/logo.png"

# Toolbox App and Export Setup
# Assuming the app instance might be passed in or a default is fetched
_app_instance_for_export = get_app(f"{Name}.EXPORT_SETUP")
export = _app_instance_for_export.tb
s_export = export(mod_name=Name, version=version, state=False, test=False, interface=ToolBoxInterfaces.native, api=False)

# --- HTML Templates (Inline for single file) ---
# Base template structure
BASE_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ subject }}</title>
    <style>
        body { margin: 0; padding: 0; width: 100% !important; -webkit-font-smoothing: antialiased; font-family: Arial, sans-serif; background-color: #f4f4f7; color: #333333; }
        .email-wrapper { background-color: #f4f4f7; {% if email_bg_url %}background-image: url('{{ email_bg_url }}'); background-size: cover; background-position: center;{% endif %} padding: 20px 0; }
        .email-container { background-color: #ffffff; width: 90%; max-width: 600px; margin: 0 auto; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); overflow: hidden; }
        .email-header { background-color: #4a90e2; padding: 20px; text-align: center; }
        .email-header img { max-width: 150px; height: auto; }
        .email-header h1 { color: #ffffff; margin: 10px 0 0 0; font-size: 24px; }
        .email-body { padding: 30px; line-height: 1.6; font-size: 16px; }
        .email-body h2 { color: #4a90e2; margin-top: 0; font-size: 20px; }
        .email-body p { margin-bottom: 15px; }
        .button { display: inline-block; background-color: #4a90e2; color: #ffffff !important; padding: 12px 25px; text-decoration: none !important; border-radius: 5px; font-weight: bold; margin-top: 10px; margin-bottom: 10px; }
        .link-in-text { color: #4a90e2; text-decoration: underline; }
        .email-footer { background-color: #eeeeee; padding: 20px; text-align: center; font-size: 12px; color: #777777; }
        .email-footer a { color: #777777; text-decoration: underline; }
        .preview-text { display: none; font-size: 1px; color: #f4f4f7; line-height: 1px; max-height: 0px; max-width: 0px; opacity: 0; overflow: hidden; mso-hide:all; }
    </style>
</head>
<body>
    <div class="preview-text">{{ preview_text }}</div>
    <table width="100%" border="0" cellpadding="0" cellspacing="0" class="email-wrapper">
        <tr>
            <td align="center">
                <table border="0" cellpadding="0" cellspacing="0" class="email-container">
                    {% if email_logo_url or app_name_for_header %}
                    <tr>
                        <td class="email-header">
                            {% if email_logo_url %}<img src="{{ email_logo_url }}" alt="{{ app_name_for_header }} Logo">{% endif %}
                            <h1>{{ app_name_for_header }}</h1>
                        </td>
                    </tr>
                    {% endif %}
                    <tr>
                        <td class="email-body">
                            {{ content | safe }}
                        </td>
                    </tr>
                    <tr>
                        <td class="email-footer">
                            © {{ current_year }} {{ app_name_for_header }}. All rights reserved.<br>
                            {% if show_unsubscribe_link and recipient_email_for_unsubscribe %}
                            <a href="{{ app_base_url }}/unsubscribe?email={{ recipient_email_for_unsubscribe | urlencode }}">Unsubscribe</a>
                            {% endif %}
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""

# Jinja2 Environment for inline templates
jinja_env = Environment(loader=BaseLoader())
base_template_jinja = jinja_env.from_string(BASE_HTML_TEMPLATE)


class EmailSender:
    def __init__(self, app_context: App = None):
        self.app = app_context if app_context else get_app(Name)  # Get default app if not provided
        self.logger = get_logger()

        if not all([GMAIL_EMAIL, GMAIL_PASSWORD, SMTP_SERVER, SMTP_PORT, SENDER_EMAIL_ADDRESS]):
            self.logger.error("SMTP configuration is incomplete. Set GMAIL_EMAIL, GMAIL_PASSWORD, SENDER_EMAIL.")
            self.is_configured = False
        else:
            self.is_configured = True

    def _render_html(self, subject, content_html, preview_text, recipient_email_for_unsubscribe=None,
                     show_unsubscribe_link=False):
        return base_template_jinja.render(
            subject=subject,
            content=content_html,
            preview_text=preview_text or subject,
            app_name_for_header=APP_NAME,
            email_logo_url=EMAIL_LOGO_URL,
            email_bg_url=EMAIL_BG_LIGHT_URL,
            current_year=datetime.datetime.now().year,
            app_base_url=APP_BASE_URL,
            recipient_email_for_unsubscribe=recipient_email_for_unsubscribe,
            show_unsubscribe_link=show_unsubscribe_link
        )

    def send_html_email(self, recipient_emails, subject, content_html, preview_text="",
                        recipient_email_for_unsubscribe=None, show_unsubscribe_link=False):
        if not self.is_configured:
            return Result.default_internal_error(info="Email service not configured.")

        if isinstance(recipient_emails, str):
            recipient_emails = [recipient_emails]

        html_body = self._render_html(subject, content_html, preview_text, recipient_email_for_unsubscribe,
                                      show_unsubscribe_link)

        msg = MIMEMultipart('alternative')
        msg['From'] = f"{APP_NAME} <{SENDER_EMAIL_ADDRESS}>"
        msg['To'] = ', '.join(recipient_emails)
        msg['Subject'] = subject
        msg.attach(MIMEText(html_body, 'html', 'utf-8'))

        try:
            if SMTP_PORT == 465:
                with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, timeout=10) as server:
                    server.login(GMAIL_EMAIL, GMAIL_PASSWORD)
                    server.sendmail(SENDER_EMAIL_ADDRESS, recipient_emails, msg.as_string())
            elif SMTP_PORT == 587:
                with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10) as server:
                    server.starttls()
                    server.login(GMAIL_EMAIL, GMAIL_PASSWORD)
                    server.sendmail(SENDER_EMAIL_ADDRESS, recipient_emails, msg.as_string())
            else:
                err_msg = f"Unsupported SMTP port: {SMTP_PORT}. Use 465 (SSL) or 587 (STARTTLS)."
                self.logger.error(err_msg)
                return Result.default_internal_error(info=err_msg)

            self.logger.info(f"Email sent to {', '.join(recipient_emails)} with subject: {subject}")
            return Result.ok(info=f"Email successfully sent to {', '.join([email[:4]+'...'+email[-12:] for email in recipient_emails])}.")
        except smtplib.SMTPAuthenticationError as e:
            self.logger.error(f"SMTP Authentication Error for user {GMAIL_EMAIL}: {e}")
            return Result.default_internal_error(info=f"SMTP Authentication Error: {e}")
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}", exc_info=True)
            return Result.default_internal_error(info=f"Failed to send email: {e}")


@s_export  # Changed to native, api=False as it's a backend function
def send_welcome_email(app: App, user_email: str, username: str, welcome_action_url: str = None):
    """Sends a welcome email to a new user."""
    sender = EmailSender(app)
    subject = f"Welcome to {APP_NAME}, {username}!"
    preview_text = f"We're thrilled to have you, {username}!"
    action_url = welcome_action_url or f"{APP_BASE_URL}/dashboard"  # Default to dashboard

    content_html = f"""
        <h2>Welcome Aboard, {username}!</h2>
        <p>Thank you for signing up for {APP_NAME}. We're excited to have you join our community!</p>
        <p>Here are a few things you might want to do next:</p>
        <ul>
            <li>Explore your new account features.</li>
            <li>Customize your profile.</li>
        </ul>
        <p>Click the button below to get started:</p>
        <a href="{action_url}" class="button">Go to Your Dashboard</a>
        <p>If the button doesn't work, copy and paste this link into your browser:<br><span class="link-in-text">{action_url}</span></p>
        <p>Best regards,<br>The {APP_NAME} Team</p>
    """
    return sender.send_html_email(user_email, subject, content_html, preview_text,
                                  recipient_email_for_unsubscribe=user_email, show_unsubscribe_link=True)


@s_export
def send_magic_link_email(app: App, user_email: str, magic_link_url: str, username: str = None):
    """Sends a magic link email for login."""
    sender = EmailSender(app)
    greeting_name = f", {username}" if username else ""
    subject = f"Your Magic Login Link for {APP_NAME}"
    preview_text = "Securely access your account with this one-time link."

    content_html = f"""
        <h2>Hello{greeting_name}!</h2>
        <p>You requested a magic link to sign in to your {APP_NAME} account.</p>
        <p>Click the button below to log in. This link is temporary and will expire shortly.</p>
        <a href="{magic_link_url}" class="button">Log In Securely</a>
        <p> Invitation key: {magic_link_url.split('?key=')[1].split('&name=')[0].replace('%23', '#')}</p>
        <p>If you did not request this link, please ignore this email. Your account is safe.</p>
        <p>If the button doesn't work, copy and paste this link into your browser:<br><span class="link-in-text">{magic_link_url}</span></p>
        <p>Thanks,<br>The {APP_NAME} Team</p>
    """
    return sender.send_html_email(user_email, subject, content_html, preview_text)


@s_export
def send_email_verification_email(app: App, user_email: str, username: str, verification_url: str):
    """Sends an email verification link to the user."""
    sender = EmailSender(app)
    subject = f"Verify Your Email for {APP_NAME}"
    preview_text = f"Almost there, {username}! Just one more step to activate your account."

    content_html = f"""
        <h2>Hi {username},</h2>
        <p>Thanks for signing up for {APP_NAME}! To complete your registration, please verify your email address by clicking the button below.</p>
        <a href="{verification_url}" class="button">Verify Email Address</a>
        <p>If you didn't create an account with {APP_NAME}, you can safely ignore this email.</p>
        <p>If the button doesn't work, copy and paste this link into your browser:<br><span class="link-in-text">{verification_url}</span></p>
        <p>Sincerely,<br>The {APP_NAME} Team</p>
    """
    return sender.send_html_email(user_email, subject, content_html, preview_text)


@s_export
def send_waiting_list_confirmation_email(app: App, user_email: str):
    """Sends a confirmation email for joining the waiting list."""
    sender = EmailSender(app)
    subject = f"You're on the Waiting List for {APP_NAME}!"
    preview_text = "Thanks for your interest! We'll keep you updated."

    content_html = f"""
        <h2>You're In!</h2>
        <p>Thank you for joining the waiting list for {APP_NAME}. We're working hard to get things ready and appreciate your interest.</p>
        <p>We'll notify you as soon as we have updates or when access becomes available.</p>
        <p>In the meantime, you can follow our progress or learn more at <a href="{APP_BASE_URL}" class="link-in-text">{APP_BASE_URL}</a>.</p>
        <p>Stay tuned,<br>The {APP_NAME} Team</p>
    """
    return sender.send_html_email(user_email, subject, content_html, preview_text,
                                  recipient_email_for_unsubscribe=user_email, show_unsubscribe_link=True)


@s_export
def send_signup_invitation_email(app: App, invited_user_email: str, invited_username: str,
                                 inviter_username: str = None):
    """Generates an invitation link and sends it via email."""
    sender = EmailSender(app)

    # Generate invitation code as specified in the prompt
    # This uses the Code class, assuming TB_R_KEY is set in the environment
    invitation_code = Code.one_way_hash(invited_username, "00#", os.getenv("TB_R_KEY", "pepper123"))[:12] + str(
        uuid.uuid4())[:6]

    # Construct the signup link URL (adjust your frontend signup path as needed)
    signup_link_url = f"{APP_BASE_URL}/web/assets/signup.html?invitation={quote(invitation_code)}&email={quote(invited_user_email)}&username={quote(invited_username)}"

    subject = f"You're Invited to Join {APP_NAME}!"
    preview_text = f"{inviter_username or 'A friend'} has invited you to {APP_NAME}!"
    inviter_line = f"<p>{inviter_username} has invited you to join.</p>" if inviter_username else "<p>You've been invited to join.</p>"

    content_html = f"""
        <h2>Hello {invited_username},</h2>
        {inviter_line}
        <p>{APP_NAME} is an exciting platform, and we'd love for you to be a part of it!</p>
        <p>Click the button below to accept the invitation and create your account:</p>
        <a href="{signup_link_url}" class="button">Accept Invitation & Sign Up</a>
        <p>This invitation is unique to you : {invitation_code}</p>
        <p>If the button doesn't work, copy and paste this link into your browser:<br><span class="link-in-text">{signup_link_url}</span></p>
        <p>We look forward to seeing you there!<br>The {APP_NAME} Team</p>
    """
    return sender.send_html_email(invited_user_email, subject, content_html, preview_text)


@export(mod_name=Name, api=True, interface=ToolBoxInterfaces.api, state=True, test=False)
def add(app: App, email: str) -> ApiResult:
    if app is None:
        app = get_app("email_waiting_list")
    # if "db" not in list(app.MOD_LIST.keys()):
    #    return "Server has no database module"
    tb_token_jwt = app.run_any('DB', 'append_on_set', query="email_waiting_list", data=[email], get_results=True)

    # Default response for internal error
    out = "My apologies, unfortunately, you could not be added to the Waiting list."
    result = Result.default_user_error(info=out, data={"message": out})
    flag = False
    if tb_token_jwt.info.exec_code == -4:
        app.run_any('DB', 'set', query="email_waiting_list", data={"set":[email]}, get_results=True)
    # Check if the email is already in the waiting list
    elif tb_token_jwt.info.exec_code == -5:
        out = "You are already in the list, please do not try to add yourself more than once."
        result = Result.default_user_error(info=out, data={"message": out})
        flag = True

    elif tb_token_jwt.is_error():
        if tb_token_jwt.info.help_text == "":
            tb_token_jwt.info.help_text = out
        result = tb_token_jwt
        flag = True

    if flag:
        return result

    out = "You will receive an invitation email in a few days"
    sending_result = send_waiting_list_confirmation_email(app, email)
    if sending_result.is_error():
        out = "You are in the list, but there was an error sending the confirmation email. Please try again later."
        result = Result.default_internal_error(info=out, data={"message": out})
    else:
        result = Result.ok(info=out, data_info="email", data={"message": out})

    return result

# --- Example Usage (for testing, typically called from other modules) ---
if __name__ == "__main__":
    # This is for local testing. Ensure environment variables are set.
    # You would need a mock or real 'App' instance from toolboxv2
    # For simplicity, this example won't run without a proper App context
    # or if GMAIL_EMAIL/PASSWORD are not set.

    print("To test, ensure GMAIL_EMAIL, GMAIL_PASSWORD, and SENDER_EMAIL are set.")
    print(f"Using SMTP Server: {SMTP_SERVER}:{SMTP_PORT}")
    print(f"App Name: {APP_NAME}, Base URL: {APP_BASE_URL}")
    print(f"Optional: EMAIL_BG_LIGHT_URL='{EMAIL_BG_LIGHT_URL}', EMAIL_LOGO_URL='{EMAIL_LOGO_URL}'")

    if GMAIL_EMAIL and GMAIL_PASSWORD:
        test_app = get_app("TestEmailApp")  # Create a dummy app for testing
        test_logger = get_logger()
        test_logger.info("Starting email tests...")

        test_recipient = "your_test_email@example.com"  # <--- !!! REPLACE THIS !!!
        if test_recipient == "your_test_email@example.com":
            print(
                "\n!!! PLEASE REPLACE 'your_test_email@example.com' with a real test email address to run tests. !!!\n")
        else:
            # Test Welcome Email
            # welcome_result = send_welcome_email(test_app, test_recipient, "Test User", f"{APP_BASE_URL}/welcome-test")
            # test_logger.info(f"Welcome Email Result: {welcome_result.info.help_text if welcome_result.info else welcome_result.print(show=False)}")

            # Test Magic Link Email
            # magic_link = f"{APP_BASE_URL}/auth/magic?token={uuid.uuid4().hex}"
            # magic_link_result = send_magic_link_email(test_app, test_recipient, magic_link, "Test User")
            # test_logger.info(f"Magic Link Email Result: {magic_link_result.info.help_text if magic_link_result.info else magic_link_result.print(show=False)}")

            # Test Email Verification
            # verification_link = f"{APP_BASE_URL}/auth/verify?token={uuid.uuid4().hex}"
            # verification_result = send_email_verification_email(test_app, test_recipient, "Test User", verification_link)
            # test_logger.info(f"Email Verification Result: {verification_result.info.help_text if verification_result.info else verification_result.print(show=False)}")

            # Test Waiting List Confirmation
            # waiting_list_result = send_waiting_list_confirmation_email(test_app, test_recipient)
            # test_logger.info(f"Waiting List Email Result: {waiting_list_result.info.help_text if waiting_list_result.info else waiting_list_result.print(show=False)}")

            # Test Signup Invitation
            # Requires TB_R_KEY to be set for Code.one_way_hash
            # if os.getenv("TB_R_KEY"):
            #    invitation_result = send_signup_invitation_email(test_app, test_recipient, "Invited Friend", "Generous Inviter")
            #    test_logger.info(f"Signup Invitation Email Result: {invitation_result.info.help_text if invitation_result.info else invitation_result.print(show=False)}")
            # else:
            #    test_logger.warning("TB_R_KEY not set, skipping signup invitation email test.")

            print(f"\nTest emails (if uncommented) would be sent to {test_recipient}.")
    else:
        print("Gmail credentials not found in environment variables. Skipping tests.")
        # email_waiting_list
