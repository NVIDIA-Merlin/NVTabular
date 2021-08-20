import smtplib
from email.message import EmailMessage


class Mail:
    def __init__(self, address, port1, port2, username, password):
        self.server = smtplib.SMTP(address, port1)
        self.server.connect(address, port2)
        self.server.ehlo()
        self.server.starttls()
        self.server.login(username, password)

    def send_email(self, subject, sender, receiver, body):
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = receiver
        msg.set_content(body)
        self.server.send_message(msg)

    def close(self):
        self.server.quit()
