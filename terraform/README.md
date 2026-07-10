# Terraform - Infrastructure as Code

This directory contains Terraform configuration for the core Azure infrastructure
used in this project: the Data Lake Storage Gen2 account and Key Vault.

## Approach: codifying an existing environment

This infrastructure was originally provisioned manually through the Azure Portal
during initial development (see ../ENGINEERING_LOG.md for the real debugging
story). This Terraform configuration codifies that same infrastructure as-code,
following a common real-world pattern: teams often start with manual provisioning
during early prototyping, then formalize into IaC once the design stabilizes.

## What is included

- Storage Account (azurerm_storage_account) - Data Lake Storage Gen2, LRS redundancy
- Storage Container (azurerm_storage_container) - the data container for Delta tables
- Key Vault (azurerm_key_vault) - RBAC-authorization mode

## What is intentionally excluded

- Role assignments (Key Vault Secrets Officer/User) - tied to tenant-specific
  object IDs, not portable across environments. See commented example in main.tf
  showing the production approach using a dynamic service principal lookup.
- Databricks workspace and cluster - out of scope for this exercise.

## Usage

terraform init
terraform validate
terraform plan -var="storage_account_name=<name>" -var="tenant_id=<id>"

## Why this matters

Even without applying against a live environment, writing and validating this
configuration demonstrates the core IaC skill: infrastructure defined as
versioned, reviewable code rather than manual clicks.
