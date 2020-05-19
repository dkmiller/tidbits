<#
Build a Docker image unless configured otherwise, then run it, exposing the
appropriate port.
#>

param(
    [string]$Image = 'streamlit-app',
    [int]$Port = 8501,
    [switch]$SkipBuild
)

if (!$SkipBuild) {
    docker build --tag $Image $PSScriptRoot
}

try {
    $Container = docker run -d -p "${$Port}:${$Port}" $Image
    Write-Host "Running Streamlit in $Container"
    docker logs -f $Container
} finally {
    # This code will run even if the user executes CTRL+C to stop the script:
    # https://stackoverflow.com/a/15788979
    Write-Host "Shutting down $Container"
    docker kill $Container
}
