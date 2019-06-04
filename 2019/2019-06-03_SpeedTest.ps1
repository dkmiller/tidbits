<#
Lightweight speed-test script.

Uses: https://pypi.org/project/speedtest-cli/ .
#>

param(
    [string]$Output,
    [int]$SleepSeconds
)

if($Output -match '\.csv$') {
    Write-Host 'Writing to CSV file...'
    $Format = '--csv'
} elseif ($Output -match '\.ndjson') {
    Write-Host 'Writing to line-delimited JSON file...'
    $Format = '--json'
} else {
    throw 'Invalid output extension provided.'
}

# Install if necessary.
pip install speedtest-cli

do {
    Write-Host 'Speed test...'
    speedtest-cli $Format --secure | Add-Content $Output
    Start-Sleep -Seconds $SleepSeconds
} while ($true)
