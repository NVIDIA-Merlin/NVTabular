import argparse
import json

import urllib3
from prettytable import PrettyTable

# Get parameters
parser = argparse.ArgumentParser(description="Pushes nightly nvbug release")
parser.add_argument("--cont", type=str, help="Merlin container")
parser.add_argument("--qa", type=str, help="Container tests results")
parser.add_argument("--username", type=str, help="Username for auth")
parser.add_argument("--password", type=str, help="Password for auth")
args = parser.parse_args()

# Set Table
x = PrettyTable(["Index", "Description", "Information"])
x.add_row(["1", "NGC catalog URL of the container", args.cont])
x.add_row(["2", "Version published to the NGC catalog", "nightly"])
x.add_row(["3", "Expected publishing date and time", "COMPLETED"])
x.add_row(["4", "QA results", args.qa])
x.add_row(["5", "Security scan results", "??"])
x.add_row(
    ["6", "SWIPAT Approval", "https://nvbugswb.nvidia.com/NvBugs5/SWBug.aspx?bugid=3267056&cmtNo="]
)
x.add_row(["7", "NGC Release Date", "COMPLETED"])
x.add_row(["8", "AWS Federation", "No"])
x.add_row(["9", "AWS Federation Google form completed", "No"])

# Create NGC Bug
ngc_config = {
    "IsSendNotification": True,
    "BugId": 0,
    "Synopsis": "[NGC Catalog Publishing] NGC Catalog Release Notification for [" + args.cont + "]",
    "Description": x.get_html_string(),
    "BugAction": {"Value": "QA - Closed - Verified"},
    "Disposition": {"Value": "Bug - Fixed"},
    "IsRestrictedAccess": 0,
    "ApplicationDivisionID": 1,
    "BugTypeID": 6,
    "BugType": "Software",
    "Priority": {"Value": "Unprioritized"},
    "Severity": {"Value": "6-Enhancement"},
    "Engineer": {"Value": "Alberto Alvarez Aldea"},
    "QAEngineer": {"Value": "albertoa@nvidia.com"},
    "ARB": [{"Value": "albertoa"}, {"Value": "jperez@nvidia.com"}],
    "CCUsers": [{"Value": "Julio Perez"}, {"Value": "Alberto Alvarez Aldea"}],
}

http = urllib3.PoolManager()
headers = urllib3.util.make_headers(basic_auth=args.username + ":" + args.password)
headers["Content-Type"] = "application/json"
result = http.request(
    "POST",
    url="https://nvbugsapi.nvidia.com/nvbugswebserviceapi/api/bug/SaveBug",
    headers=headers,
    body=json.dumps(ngc_config),
)

print(result.status)
print(json.loads(result.data.decode("utf-8")))
