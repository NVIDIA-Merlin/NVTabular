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
  credentials = file("terraform.json")
  project = "merlin-295819"
  region  = "${var.region}"
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
    }
  }

  network_interface {
    network = "default"
    access_config {
    }
  }

  guest_accelerator{
    type = "nvidia-tesla-v100" // Type GPU
    count = 8 // Num GPU
  }

  scheduling{
    on_host_maintenance = "TERMINATE"
  }

  metadata_startup_script = "${file("run.sh")}"
}