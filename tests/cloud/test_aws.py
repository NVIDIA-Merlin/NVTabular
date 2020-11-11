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

import boto3

ec2 = boto3.resource('ec2')

# Create EC2 key pair
outfile = open('ec2-keypair.pem','w')
key_pair = ec2.create_key_pair(KeyName='ec2-keypair')
KeyPairOut = str(key_pair.key_material)
print(KeyPairOut)
outfile.write(KeyPairOut)

# Create EC2 instances
instances = ec2.create_instances(
     ImageId='ami-063585f0e06d22308', # Deep Learning AMI (Ubuntu 18.04) Version 36.0 
     MinCount=1,
     MaxCount=1,
     InstanceType='p4d.24xlarge', # 8xA100
     KeyName='ec2-keypair'
 )

# Start EC2 instances
ec2.start_instances(InstanceIds=instances)

# Run NVTabular Tests
ssm = boto3.client('ssm')
commands = ['docker run --runtime=nvidia --ipc=host --name aws_test nvcr.io/nvidia/nvtabular:0.2 /bin/bash -c "source activate rapids && pytest /nvtabular/tests"']
result = ssm.send_command(
        DocumentName="AWS-RunShellScript", # One of AWS' preconfigured documents
        Parameters={'commands': commands},
        InstanceIds=instance_ids,
    )
print{result)

# Stop EC2 instances
ec2.stop_instances(InstanceIds=instances)

# Remove EC2 instances
ec2.terminate_instances(InstanceIds=instances)

# Delete EC2 key pair
ec2.delete_key_pair(KeyName='KEY_PAIR_NAME')