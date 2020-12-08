terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
      version = "3.5.0"
    }
  }
}

variable "region" {
  default = "us-west1-a"
}

provider "google" {

  credentials = file("<NAME>.json")
  project = "<PROJECT_ID>"
  region  = "${var.region}"
}

resource "google_compute_network" "vpc_network" {
  name = "terraform-network"
}

resource "google_compute_instance" "vm_instance" {
  name         = "terraform-instance"
  machine_type = "f1-micro"

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-10"
    }
  }

  guest_accelerator{
    type = "nvidia-tesla-v100" // Type GPU
    count = 1 // Num GPU
  }

  scheduling{
    on_host_maintenance = "TERMINATE"
  }

  network_interface {
    network = "default"
    access_config {
    }
  }
}
