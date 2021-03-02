param(
    $File
)

$Data = import-csv $File

# https://stackoverflow.com/a/33156229
$NewData = New-Object System.Collections.Generic.List[System.Object]

foreach ($Line in $Data) {
    $Id = $Line.ShortUrl.Split("=")[-1]
    write-host $Id
    $Resp = iwr "https://marriageheat.com/wp-admin/admin-ajax.php" -Method Post -Body "action=load_results&postID=$Id&nonce=41908004f0"
    $Json = $Resp.Content | convertfrom-json
    write-host $Json
    $Line | Add-Member -NotePropertyName VoteCount -NotePropertyValue $Json.voteCount
    $Line | Add-Member -NotePropertyName AvgRating -NotePropertyValue $Json.avgRating
    $NewData.Add($Line)
}

$NewData | Export-Csv new-data.csv
