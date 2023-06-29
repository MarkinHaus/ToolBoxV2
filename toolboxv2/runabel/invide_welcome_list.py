import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
 # https://mailtrap.io/blog/python-send-email-gmail/
def send_html_email(subject, html_content, to_email, from_email, from_password, smtp_server='smtp.gmail.com', smtp_port=587):
    # Erstellen Sie die E-Mail-Nachricht
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email

    # Fügen Sie den HTML-Inhalt hinzu
    html_part = MIMEText(html_content, 'html')
    msg.attach(html_part)

    # Verbinden Sie sich mit dem SMTP-Server und senden Sie die E-Mail
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(from_email, from_password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
        print('E-Mail erfolgreich gesendet!')
    except Exception as e:
        print(f'Fehler beim Senden der E-Mail: {e}')


def run(app, _):
    # Beispiel für die Verwendung der Funktion
    subject = 'Betreff der E-Mail'
    html_content = '''
    <html>
      <body>
        <h1>Willkommen!</h1>
        <p>Dies ist eine benutzerdefinierte HTML-E-Mail.</p>
      </body>
    </html>
    '''
    to_email = 'MarkinHausmanns@gmail.com'
    from_email = 'drrking883@gmail.com'
    from_password = '3170mm3170M2!'

    send_html_email(subject, html_content, to_email, from_email, from_password)
