Write-Host $PSScriptRoot

$Current = $PWD

Get-ChildItem "$PSScriptRoot/components" | ForEach-Object {
    $ImageName = "ghcr.io/dkmiller/tidbits/$($_.Name):latest"
    Write-Host "Building $ImageName ... "
    Set-Location $_

    docker build -t $ImageName .

    Write-Host "Pushing $ImageName ..."
    docker push $ImageName
}

Set-Location $Current
