import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
import os

load_dotenv()

# Email credentials
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "465"))
SMTP_USE_TLS = os.environ.get("SMTP_USE_TLS", "False").lower() in ("true", "1", "yes")

def send_email(receiver_email, subject, body, is_html=False):
    if not SENDER_EMAIL or not EMAIL_PASSWORD:
        print("Email sending skipped: SENDER_EMAIL or EMAIL_PASSWORD not set in environment.")
        return False

    # Setup the MIME
    message = MIMEMultipart()
    message["From"] = SENDER_EMAIL
    message["To"] = receiver_email
    message["Subject"] = subject

    # Attach the body text to the email
    message.attach(MIMEText(body, "html" if is_html else "plain"))

    server = None
    # SMTP server setup
    try:
        if SMTP_PORT == 465 and not SMTP_USE_TLS:
            server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
        else:
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            if SMTP_USE_TLS or SMTP_PORT == 587:
                server.starttls()
                
        server.login(SENDER_EMAIL, EMAIL_PASSWORD)
        
        # Convert the message to a string and send it
        server.sendmail(SENDER_EMAIL, receiver_email, message.as_string())
        print(f"Email sent successfully to {receiver_email}!")
        return True
        
    except Exception as e:
        print(f"Failed to send email to {receiver_email}. Error: {e}")
        return False
    finally:
        if server:
            try:
                server.quit()
            except Exception:
                pass