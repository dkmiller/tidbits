provider "azurerm" {
  version = "~>2.0"
  features {}
}

resource "azurerm_resource_group" "rg" {
        name = "QuickstartTerraformTest-rg"
        location = "westus2"
}
