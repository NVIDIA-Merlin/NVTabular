terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 3.0"
    }
  }
}

variable "image_id" {
  default = "p3dn.24xlarge"
}

variable "script" {
  default = "run.sh"
}

provider "aws" {
  profile = "default"
  region  = "us-east-1"
}

resource "aws_instance" "dgx-100" {
  # Machine
  ami           = "ami-0f899ff8474ea45a9"
  instance_type = var.image_id
  
  # Storage
  root_block_device {
        volume_size = 1000
  }

  # Access to AWS
  connection {
    type        = "ssh"
    user        = "ec2-user"
    password    = ""
    private_key = "~/.aws/terraform.pem"
    host        = self.public_ip
  }

  # Run script after init
  provisioner "file" {
    source      = "${var.script}"
    destination = "/tmp/${var.script}"
  }

  provisioner "remote-exec" {
    inline = [
      "chmod +x /tmp/${var.script}",
      "/tmp/${var.script} -c AWS -g 0",
    ]
  }
}