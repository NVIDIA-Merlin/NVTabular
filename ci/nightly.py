from tabulate import tabulate
import argparse
import smtplib

# Get parameters
parser = argparse.ArgumentParser(description='Pushes nightly nvbug release')
parser.add_argument('--cont', type=str, help='Merlin container')
parser.add_argument('--qa', type=str, help='Container tests results')
parser.add_argument('--server', type=str, help='Server to send email')
parser.add_argument('--port', type=str, help='Port to send email')
parser.add_argument('--username', type=str, help='Username to send email')
parser.add_argument('--password', type=str, help='Password to send email')
parser.add_argument('--sender', type=str, help='Email sender')
parser.add_argument('--receiver', type=str, help='Email receiver')
args = parser.parse_args()

# Set date
mydata = [["1", "NGC catalog URL of the container", args.cont],
          ["2", "Version published to the NGC catalog", "nightly"], 
          ["3", "Expected publishing date and time", "COMPLETED"], 
          ["4", "QA results", args.qa], 
          ["5", "Security scan results", "??"], 
          ["6", "SWIPAT Approval", "??"], 
          ["7", "NGC Release Date", "COMPLETED"], 
          ["8", "AWS Federation", "No"], 
          ["9", "AWS Federation Google form completed", "No"]]
  
# create table
body = tabulate(mydata, tablefmt="grid")

# Send email
server = smtplib.SMTP(args.server, args.port)
server.login(args.username, args.password)

header = "/n"
server.sendmail(args.sender, args.receiver, header+body)
