output "vpc_id" {
  value = aws_vpc.this.id
}

output "private_subnet_ids" {
  value = aws_subnet.private[*].id
}

output "bucket_names" {
  value = { for k, b in aws_s3_bucket.this : k => b.bucket }
}

output "ecr_repository_url" {
  value = aws_ecr_repository.scorer.repository_url
}

output "kms_key_arn" {
  value = aws_kms_key.this.arn
}
