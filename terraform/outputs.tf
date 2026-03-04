# terraform/outputs.tf

output "cloud_run_url" {
  description = "URL of the deployed Cloud Run inference service"
  value       = google_cloud_run_v2_service.inference.uri
}

output "artifact_registry_url" {
  description = "URL of the Artifact Registry Docker repository"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker.repository_id}"
}

output "data_bucket" {
  description = "GCS bucket for data"
  value       = google_storage_bucket.data.name
}

output "models_bucket" {
  description = "GCS bucket for model artifacts"
  value       = google_storage_bucket.models.name
}

output "mlflow_bucket" {
  description = "GCS bucket for MLflow artifacts"
  value       = google_storage_bucket.mlflow.name
}

output "service_account_email" {
  description = "Cloud Run service account email"
  value       = google_service_account.cloud_run.email
}
