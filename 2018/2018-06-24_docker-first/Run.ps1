<#
This is a Makefile proxy which runs some code from different
frameworks.
#>

Write-Host 'Running some C#'
# This is needed to copy any local files.
docker build --tag net net
docker run net dotnet run arg0-foo arg1-baz

Write-Host 'Running some Python'
# This is needed to copy any local files.
docker build --tag py py
docker run py python app.py arg0-foo arg1-baz arg2-bar

Write-Host 'Using Java + Gradle'
docker build --tag java java
