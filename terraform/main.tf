terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.90"
    }
  }
  required_version = ">= 1.5.0"
}

provider "azurerm" {
  features {
    key_vault {
      purge_soft_delete_on_destroy    = false
      recover_soft_deleted_key_vaults = true
    }
  }
}

variable "resource_group_name" {
  description = "Name of the existing resource group for this project"
  type        = string
  default     = "financial-rag-rg"
}

variable "location" {
  description = "Azure region for all resources"
  type        = string
  default     = "eastus"
}

variable "storage_account_name" {
  description = "Globally unique name for the storage account"
  type        = string
}

variable "key_vault_name" {
  description = "Name of the Key Vault used for storing API secrets"
  type        = string
  default     = "financial-rag-rg"
}

variable "tenant_id" {
  description = "Azure AD tenant ID"
  type        = string
}

data "azurerm_resource_group" "main" {
  name = var.resource_group_name
}

resource "azurerm_storage_account" "data_lake" {
  name                     = var.storage_account_name
  resource_group_name      = data.azurerm_resource_group.main.name
  location                 = data.azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  account_kind             = "StorageV2"
  is_hns_enabled           = true

  tags = {
    project     = "financial-rag"
    environment = "development"
    managed_by  = "terraform"
  }
}

resource "azurerm_storage_container" "data_container" {
  name                  = "financial-rag-data"
  storage_account_name  = azurerm_storage_account.data_lake.name
  container_access_type = "private"
}

resource "azurerm_key_vault" "main" {
  name                       = var.key_vault_name
  resource_group_name        = data.azurerm_resource_group.main.name
  location                   = data.azurerm_resource_group.main.location
  tenant_id                  = var.tenant_id
  sku_name                   = "standard"
  soft_delete_retention_days = 90
  purge_protection_enabled   = false
  enable_rbac_authorization  = true

  network_acls {
    default_action = "Allow"
    bypass         = "AzureServices"
  }

  tags = {
    project     = "financial-rag"
    environment = "development"
    managed_by  = "terraform"
  }
}

output "storage_account_name" {
  value = azurerm_storage_account.data_lake.name
}

output "storage_account_primary_endpoint" {
  value = azurerm_storage_account.data_lake.primary_dfs_endpoint
}

output "key_vault_uri" {
  value = azurerm_key_vault.main.vault_uri
}

output "key_vault_id" {
  value = azurerm_key_vault.main.id
}
