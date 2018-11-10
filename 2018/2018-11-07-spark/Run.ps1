<#
Prepare and start a container running a local Spark cluster with only one
master and worker. The logs for that cluster will be in a new PowerShell
window.
#>

Start-Process -WorkingDirectory $PSScriptRoot powershell {
    # Ovewrite the Git submodule's Docker compose file with a fixed one.
    Copy-Item docker-compose.yml docker-spark

    # Start the Spark instance.
    Set-Location docker-spark
    docker system prune --force
    docker-compose up
}
