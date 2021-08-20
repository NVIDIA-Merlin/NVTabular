import argparse
import smtplib
from email.message import EmailMessage

from tabulate import tabulate

# Get parameters
parser = argparse.ArgumentParser(description="Pushes nightly nvbug release")
parser.add_argument("--cont", type=str, help="Merlin container")
parser.add_argument("--qa", type=str, help="Container tests results")
parser.add_argument("--server", type=str, help="Server to send email")
parser.add_argument("--port1", type=str, help="Port to send email")
parser.add_argument("--port2", type=str, help="Port to send email")
parser.add_argument("--username", type=str, help="Username to send email")
parser.add_argument("--password", type=str, help="Password to send email")
parser.add_argument("--sender", type=str, help="Email sender")
parser.add_argument("--receiver", type=str, help="Email receiver")
args = parser.parse_args()

# Set Table
mydata = [
    ["1", "NGC catalog URL of the container", args.cont],
    ["2", "Version published to the NGC catalog", "nightly"],
    ["3", "Expected publishing date and time", "COMPLETED"],
    ["4", "QA results", args.qa],
    ["5", "Security scan results", "??"],
    ["6", "SWIPAT Approval", "??"],
    ["7", "NGC Release Date", "COMPLETED"],
    ["8", "AWS Federation", "No"],
    ["9", "AWS Federation Google form completed", "No"],
]

# Compose email
msg = EmailMessage()
msg["Subject"] = "NGC Release"
msg["From"] = args.sender
msg["To"] = args.receiver
msg.set_content(tabulate(mydata, tablefmt="grid"))

# Send email
server = smtplib.SMTP(args.server, args.port1)
server.connect(args.server, args.port2)
server.ehlo()
server.starttls()
server.login(args.username, args.password)
server.send_message(msg)
server.quit()
