param(
    [ValidateSet("job", "sweep")]
    [string]$File = "job",
    [switch]$Local,
    [string]$LocalDataLocation = "/src/tmp/data"
)


git clean -xdf

if ($Local) {
    $Data = Get-ChildItem $LocalDataLocation
    if ($Data) {
        python train.py data.root=$LocalDataLocation data.download=false
    }
    else {
        python train.py data.root=$LocalDataLocation
    }
}

az ml job create --file "$($File).yml" --query metadata.interaction_endpoints.studio --out tsv
