# https://github.com/PowerShell/PowerShell-Docker
FROM mcr.microsoft.com/powershell:latest

RUN apt-get update
RUN apt-get install -y curl sudo unzip wget

# https://docs.microsoft.com/en-us/cli/azure/install-azure-cli-apt
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

ARG terraform_url="https://releases.hashicorp.com/terraform/0.12.29/terraform_0.12.29_linux_amd64.zip"

# https://www.slashroot.in/terraform-tutorial-how-install-terraform
RUN wget $terraform_url
RUN unzip terra*
RUN mv terraform /usr/local/bin/
