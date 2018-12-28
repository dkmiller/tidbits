<#
Prepare and start a container running a local Spark cluster with only one
master and worker. The logs for that cluster will be in a new PowerShell
window.
#>

Start-Process -WorkingDirectory $PSScriptRoot -WindowStyle Minimized powershell {
    # Ovewrite the Git submodule's Docker compose file with a fixed one.
    Copy-Item docker-compose.yml docker-spark

    # Start the Spark instance.
    Set-Location docker-spark
    docker system prune --force
    docker-compose up
}

# Follow: https://stackoverflow.com/a/22430636 .
do {
    Write-Host 'Waiting for Spark cluster to initialize...'
    Start-Sleep 5
} until ((Test-NetConnection localhost -Port 8080).TcpTestSucceeded)

# Enter the master's shell.
docker exec --interactive --tty docker-spark_master_1 /bin/bash