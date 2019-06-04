<#
Lightweight speed-test script.

Uses: https://pypi.org/project/speedtest-cli/ .
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$Output,
    [int]$SleepSeconds,
    [switch]$Summarize
)

if ($Summarize) {
    Get-Content $Output |
        # TODO: figure out a way to also parse CSV.
        ConvertFrom-Json |
        Select-Object -Expand Download |
        Measure-Object -Average -Maximum -Minimum -StandardDeviation
    # https://stackoverflow.com/a/2022469
    exit
}

# Install if necessary.
pip install speedtest-cli

do {
    Write-Host 'Speed test...'
    speedtest-cli --json --secure | Add-Content $Output
    Start-Sleep -Seconds $SleepSeconds
} while ($true)
