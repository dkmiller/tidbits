param(
    $File,
    [int]$Nonce
)

$Data = import-csv $File

# https://stackoverflow.com/a/33156229
$NewData = New-Object System.Collections.Generic.List[System.Object]

foreach ($Line in $Data) {
    $Id = $Line.ShortUrl.Split("=")[-1]
    write-host $Id
    try {
        $Resp = iwr "https://marriageheat.com/wp-admin/admin-ajax.php" -Method Post -Body "action=load_results&postID=$Id&nonce=$Nonce"
        $Json = $Resp.Content | convertfrom-json
        write-host $Json
        $Line | Add-Member -NotePropertyName VoteCount -NotePropertyValue $Json.voteCount
        $Line | Add-Member -NotePropertyName AvgRating -NotePropertyValue $Json.avgRating
        $NewData.Add($Line)
    
    } catch {
        Write-Warning "failed :("
    }
}

$NewData | Export-Csv new-data.csv
