# terraform/main.tf

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Enable required APIs
resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "storage.googleapis.com",
    "monitoring.googleapis.com",
    "secretmanager.googleapis.com",
    "iam.googleapis.com",
  ])

  service            = each.value
  disable_on_destroy = false
}

# Artifact Registry for Docker images
resource "google_artifact_registry_repository" "docker" {
  location      = var.region
  repository_id = "${var.app_name}-docker"
  format        = "DOCKER"
  description   = "Docker images for sentiment MLOps pipeline"

  depends_on = [google_project_service.apis]
}

# Cloud Storage buckets
resource "google_storage_bucket" "data" {
  name                        = "${var.project_id}-${var.app_name}-data"
  location                    = var.region
  force_destroy               = true
  uniform_bucket_level_access = true
}

resource "google_storage_bucket" "models" {
  name                        = "${var.project_id}-${var.app_name}-models"
  location                    = var.region
  force_destroy               = true
  uniform_bucket_level_access = true
}

resource "google_storage_bucket" "mlflow" {
  name                        = "${var.project_id}-${var.app_name}-mlflow"
  location                    = var.region
  force_destroy               = true
  uniform_bucket_level_access = true
}

# Service account for Cloud Run
resource "google_service_account" "cloud_run" {
  account_id   = "${var.app_name}-run"
  display_name = "Sentiment MLOps Cloud Run SA"
}

# Grant Cloud Run SA access to model bucket
resource "google_storage_bucket_iam_member" "run_models_reader" {
  bucket = google_storage_bucket.models.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.cloud_run.email}"
}

# Grant Cloud Run SA access to MLflow bucket
resource "google_storage_bucket_iam_member" "run_mlflow_reader" {
  bucket = google_storage_bucket.mlflow.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.cloud_run.email}"
}

# Grant Cloud Run SA monitoring metric writer
resource "google_project_iam_member" "run_monitoring" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.cloud_run.email}"
}

# Cloud Run service
resource "google_cloud_run_v2_service" "inference" {
  name     = "${var.app_name}-inference"
  location = var.region

  template {
    scaling {
      min_instance_count = var.cloud_run_min_instances
      max_instance_count = var.cloud_run_max_instances
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker.repository_id}/sentiment-serve:latest"

      resources {
        limits = {
          memory = var.cloud_run_memory
          cpu    = var.cloud_run_cpu
        }
      }

      ports {
        container_port = 8080
      }

      env {
        name  = "MODEL_DIR"
        value = "/app/models/sentiment"
      }

      startup_probe {
        http_get {
          path = "/health"
        }
        initial_delay_seconds = 10
        period_seconds        = 5
        failure_threshold     = 3
      }

      liveness_probe {
        http_get {
          path = "/health"
        }
        period_seconds = 30
      }
    }

    service_account = google_service_account.cloud_run.email
  }

  depends_on = [google_project_service.apis]
}

# Allow unauthenticated access (for demo/portfolio purposes)
resource "google_cloud_run_v2_service_iam_member" "public" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.inference.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Cloud Monitoring alert policy for error rate
resource "google_monitoring_alert_policy" "error_rate" {
  display_name = "${var.app_name} - High Error Rate"
  combiner     = "OR"

  conditions {
    display_name = "Cloud Run 5xx error rate"
    condition_threshold {
      filter          = "resource.type = \"cloud_run_revision\" AND resource.labels.service_name = \"${var.app_name}-inference\" AND metric.type = \"run.googleapis.com/request_count\" AND metric.labels.response_code_class = \"5xx\""
      duration        = "60s"
      comparison      = "COMPARISON_GT"
      threshold_value = 5

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }

  notification_channels = []  # Add notification channel IDs here
}
