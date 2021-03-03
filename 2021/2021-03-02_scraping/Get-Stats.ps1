param(
    $File,
    $Nonce,
    $Domain,
    [int]$Delay = 1
)

$Data = Get-Content $File | ConvertFrom-Json
$StatsFile = "$File-stats.csv"

foreach ($Line in $Data) {
    $Id = $Line.short_url.Split("=")[-1]
    Write-Host "Processing page with ID: $Id"
    try {
        $Body = "action=load_results&postID=$Id&nonce=$Nonce"
        $Url = "https://$($Domain).com/wp-admin/admin-ajax.php"
        Write-Host "Calling $Url with body $Body"
        $Response = Invoke-WebRequest $Url -Method Post -Body $Body
        $Json = $Response.Content | ConvertFrom-Json
        Write-Host "Received: $Json"
        $Line | Add-Member -NotePropertyName VoteCount -NotePropertyValue $Json.voteCount
        $Line | Add-Member -NotePropertyName AvgRating -NotePropertyValue $Json.avgRating

        $Line | Export-Csv $StatsFile -Append
    }
    catch {
        Write-Warning "failed :("
    }

    Write-Host "Sleeping $Delay seconds..."
    Start-Sleep -Seconds $Delay
}
