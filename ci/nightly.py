from tabulate import tabulate
import argparse

# Get parameters
parser = argparse.ArgumentParser(description='Pushes nightly nvbug release')
parser.add_argument('--cont', type=str, help='Merlin container')
parser.add_argument('--qa', type=str, help='Container tests results')
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
print(tabulate(mydata, tablefmt="grid"))
