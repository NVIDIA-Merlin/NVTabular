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

import boto3

client = boto3.client("ec2")
resource = boto3.resource("ec2")

# Create EC2 key pair
keypair_id = "ec2-keypair"
keypair_file = "ec2-keypair.pem"
outfile = open(keypair_file, "w")
key_pair = client.create_key_pair(KeyName=keypair_id)
KeyPairOut = str(key_pair["KeyMaterial"])
print(KeyPairOut)
outfile.write(KeyPairOut)

# Create EC2 instances
instances = resource.create_instances(
    ImageId="ami-0f899ff8474ea45a9",  # Deep Learning AMI (Amazon Linux 2) Version 36.0
    MinCount=1,
    MaxCount=1,
    InstanceType="p3dn.24xlarge",  # 8xV100
    KeyName="ec2-keypair",
    SecurityGroupIds=[
        "launch-wizard-3",
    ],
)
instances = [ins.id for ins in instances]
print(instances)

# Start EC2 instances
client.start_instances(InstanceIds=instances)

# Run NVTabular Tests
ssm = boto3.client("ssm")
commands = [
    "docker run --runtime=nvidia --ipc=host --name aws_test nvcr.io/nvidia/nvtabular:0.2 "
    '/bin/bash -c "source activate rapids && pytest /nvtabular/tests"'
]
result = ssm.send_command(
    DocumentName="AWS-RunShellScript",
    Parameters={"commands": commands},
    InstanceIds=instances,
)
print(result)

# Stop EC2 instances
client.stop_instances(InstanceIds=instances)

# Remove EC2 instances
resource.terminate_instances(InstanceIds=instances)

# Delete EC2 key pair
client.delete_key_pair(KeyName=keypair_id)
os.remove(keypair_file)
