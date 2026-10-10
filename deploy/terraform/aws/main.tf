# Data perimeter for running Forewarn inside the bank's own AWS account.
#
# What this builds
#   * a VPC with private subnets only: no internet gateway, no NAT gateway, so
#     there is no route out of the account
#   * VPC endpoints for the AWS services the workload needs (S3, ECR, CloudWatch
#     Logs, STS, KMS)
#   * three encrypted S3 buckets (features in, decisions out, audit log) that
#     refuse any request not arriving through the S3 endpoint of this VPC
#   * an audit bucket with Object Lock, so decision records cannot be changed
#     or deleted during the retention period
#   * a private ECR repository with immutable tags
#
# What it does not build: the cluster. Attach the bank's own EKS (or ECS)
# cluster to these subnets and apply deploy/k8s.
#
# Status: reference configuration. It was written without access to Terraform
# in the build environment and has NOT been run through `terraform validate`
# or `plan`. Treat it as a starting point for the bank's platform team.

data "aws_region" "current" {}
data "aws_caller_identity" "current" {}

locals {
  buckets = toset(["features", "decisions", "audit"])
  interface_services = toset(["ecr.api", "ecr.dkr", "logs", "sts", "kms"])
}

# ------------------------------------------------------------------ network
resource "aws_vpc" "this" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags                 = { Name = var.name }
}

resource "aws_subnet" "private" {
  count             = length(var.availability_zones)
  vpc_id            = aws_vpc.this.id
  availability_zone = var.availability_zones[count.index]
  cidr_block        = cidrsubnet(var.vpc_cidr, 2, count.index)
  tags              = { Name = "${var.name}-private-${count.index}" }
}

# One route table, no default route: nothing here can reach the internet.
resource "aws_route_table" "private" {
  vpc_id = aws_vpc.this.id
  tags   = { Name = "${var.name}-private" }
}

resource "aws_route_table_association" "private" {
  count          = length(aws_subnet.private)
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}

resource "aws_security_group" "endpoints" {
  name_prefix = "${var.name}-endpoints-"
  description = "HTTPS from inside the VPC to interface endpoints"
  vpc_id      = aws_vpc.this.id

  ingress {
    description = "HTTPS from the VPC"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
}

resource "aws_vpc_endpoint" "s3" {
  vpc_id            = aws_vpc.this.id
  service_name      = "com.amazonaws.${data.aws_region.current.name}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = [aws_route_table.private.id]
}

resource "aws_vpc_endpoint" "interface" {
  for_each            = local.interface_services
  vpc_id              = aws_vpc.this.id
  service_name        = "com.amazonaws.${data.aws_region.current.name}.${each.key}"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = aws_subnet.private[*].id
  security_group_ids  = [aws_security_group.endpoints.id]
  private_dns_enabled = true
}

# ------------------------------------------------------------------ encryption
resource "aws_kms_key" "this" {
  description             = "${var.name} data at rest"
  enable_key_rotation     = true
  deletion_window_in_days = 30
}

resource "aws_kms_alias" "this" {
  name          = "alias/${var.name}"
  target_key_id = aws_kms_key.this.key_id
}

# ------------------------------------------------------------------ storage
resource "aws_s3_bucket" "this" {
  for_each            = local.buckets
  bucket              = "${var.name}-${each.key}-${data.aws_caller_identity.current.account_id}"
  object_lock_enabled = each.key == "audit"
}

resource "aws_s3_bucket_public_access_block" "this" {
  for_each                = local.buckets
  bucket                  = aws_s3_bucket.this[each.key].id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "this" {
  for_each = local.buckets
  bucket   = aws_s3_bucket.this[each.key].id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "this" {
  for_each = local.buckets
  bucket   = aws_s3_bucket.this[each.key].id
  rule {
    bucket_key_enabled = true
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.this.arn
    }
  }
}

resource "aws_s3_bucket_object_lock_configuration" "audit" {
  bucket     = aws_s3_bucket.this["audit"].id
  depends_on = [aws_s3_bucket_versioning.this]
  rule {
    default_retention {
      mode  = "COMPLIANCE"
      years = var.audit_retention_years
    }
  }
}

# Requests must be encrypted in transit and must arrive through this VPC's S3
# endpoint. Note that the second statement also locks out console access from
# outside the VPC, which is the intent; break-glass access needs its own
# documented exception.
data "aws_iam_policy_document" "bucket" {
  for_each = local.buckets

  statement {
    sid       = "DenyInsecureTransport"
    effect    = "Deny"
    actions   = ["s3:*"]
    resources = [aws_s3_bucket.this[each.key].arn, "${aws_s3_bucket.this[each.key].arn}/*"]
    principals {
      type        = "*"
      identifiers = ["*"]
    }
    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }

  statement {
    sid       = "DenyOutsideVpcEndpoint"
    effect    = "Deny"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"]
    resources = [aws_s3_bucket.this[each.key].arn, "${aws_s3_bucket.this[each.key].arn}/*"]
    principals {
      type        = "*"
      identifiers = ["*"]
    }
    condition {
      test     = "StringNotEquals"
      variable = "aws:sourceVpce"
      values   = [aws_vpc_endpoint.s3.id]
    }
  }
}

resource "aws_s3_bucket_policy" "this" {
  for_each = local.buckets
  bucket   = aws_s3_bucket.this[each.key].id
  policy   = data.aws_iam_policy_document.bucket[each.key].json
}

# ------------------------------------------------------------------ registry
resource "aws_ecr_repository" "scorer" {
  name                 = "${var.name}-scorer"
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "KMS"
    kms_key         = aws_kms_key.this.arn
  }
}
