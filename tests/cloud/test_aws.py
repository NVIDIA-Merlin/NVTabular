#
# Copyright (c) 2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import time

import boto3
import paramiko


def wait_until_checks(instances):
    while True:
        response = client.describe_instance_status(InstanceIds=instances)
        try:
            if response["InstanceStatuses"][0]["InstanceStatus"]["Status"] == "ok":
                if response["InstanceStatuses"][0]["SystemStatus"]["Status"] == "ok":
                    break
        except Exception:
            pass
        finally:
            time.sleep(10)


client = boto3.client("ec2")
resource = boto3.resource("ec2")

# Create EC2 key pair
print("[+] Creating EC2 KeyPar")
keypair_id = "ec2-keypair"
keypair_file = "ec2-keypair.pem"
outfile = open(keypair_file, "w")
key_pair = client.create_key_pair(KeyName=keypair_id)
outfile.write(str(key_pair["KeyMaterial"]))
outfile.close()
os.chmod(keypair_file, 0o400)

# Create EC2 instance
print("[+] Creating instances")
instances = resource.create_instances(
    ImageId="ami-0f899ff8474ea45a9",  # Deep Learning AMI (Amazon Linux 2) Version 36.0
    BlockDeviceMappings=[{"DeviceName": "/dev/xvda", "Ebs": {"VolumeSize": 1000}}],  # 1TB Storage
    MinCount=1,
    MaxCount=1,
    InstanceType="p3dn.24xlarge",  # 8xV100
    KeyName="ec2-keypair",
    SecurityGroupIds=["launch-wizard-3"],
)
instances = [ins.id for ins in instances]

# Start EC2 instances make sure it is ready
print("[+] Starting instances")
print(instances)
client.start_instances(InstanceIds=instances)
wait_until_checks(instances)
current_instance = list(resource.instances.filter(InstanceIds=instances))
print(current_instance)
ip_address = current_instance[0].public_ip_address
print(ip_address)

# Run NVTabular Tests
print("[+] Running tests")
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
privkey = paramiko.RSAKey.from_private_key_file(keypair_file)
ssh.connect(hostname=ip_address, username="ec2-user", pkey=privkey)
command = (
    "docker run --runtime=nvidia --ipc=host --name aws_test nvcr.io/nvidia/nvtabular:0.2 "
    '/bin/bash -c "source activate rapids && pytest /nvtabular/tests"'
)
stdin, stdout, stderr = ssh.exec_command(command)
print("stdout:", stdout.read())
print("stderr:", stderr.read())

# Stop EC2 instances
print("[+] Stopping instances")
client.stop_instances(InstanceIds=instances)

# Remove EC2 instances
print("[+] Removing instances")
client.terminate_instances(InstanceIds=instances)

# Delete EC2 key pair
print("[+] Deleting KeyPar")
client.delete_key_pair(KeyName=keypair_id)
os.remove(keypair_file)
