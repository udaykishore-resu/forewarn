variable "name" {
  description = "Prefix for every resource."
  type        = string
  default     = "forewarn"
}

variable "vpc_cidr" {
  description = "CIDR of the isolated VPC."
  type        = string
  default     = "10.80.0.0/20"
}

variable "availability_zones" {
  description = "Zones for the private subnets (two or more)."
  type        = list(string)
}

variable "audit_retention_years" {
  description = "Object Lock retention for the audit bucket, in compliance mode."
  type        = number
  default     = 7
}
