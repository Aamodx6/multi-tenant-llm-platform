variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "cluster_name" {
  type    = string
  default = "llm-prod-eks"
}

variable "model_bucket_name" {
  type    = string
  default = "llm-model-weights"
}
