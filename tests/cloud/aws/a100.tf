terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 3.0"
    }
  }
}

provider "aws" {
  profile = "default"
  region  = "us-east-1"
}

resource "aws_instance" "v100" {
  ami           = "ami-0f899ff8474ea45a9"
  instance_type = "p4d.24xlarge"
  root_block_device {
        volume_size           = 1000
    }
}
