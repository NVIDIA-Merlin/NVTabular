terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "3.5.0"
    }
  }
}

variable "gce_ssh_user" {
  default = "albertoa"
}

variable "script" {
  default = "run.py"
}

variable "zone" {
  default = "us-west1-a"
}

provider "google" {
  project     = "merlin-295819"
  zone      = var.zone
}

resource "google_compute_network" "vpc_network" {
  name = "terraform-network"
}

resource "google_compute_instance" "v100" {
  name         = "v100-gpu"
  machine_type = "n1-standard-96"

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-10"
      size = 1000
      type = "pd-ssd"
    }
  }

  network_interface {
    network = "default"
    access_config {
    }
  }

  guest_accelerator {
    type  = "nvidia-tesla-v100" // Type GPU
    count = 8                   // Num GPU
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
  }

  connection {
    type = "ssh"
    host = self.network_interface[0].access_config[0].nat_ip
    user = var.gce_ssh_user
    private_key = file("~/.ssh/id_rsa.pub")
  }

  # Run script after init
  provisioner "file" {
    source      = "../${var.script}"
    destination = "/tmp/${var.script}"
  }

  provisioner "remote-exec" {
    inline = [
      "chmod +x /tmp/${var.script}",
      "python /tmp/${var.script} -c gcp",
    ]
  }
  metadata = {
    ssh-keys = "${var.gce_ssh_user}:${file("~/.ssh/id_rsa.pub")}"
  }
}