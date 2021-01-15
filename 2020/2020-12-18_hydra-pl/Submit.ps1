param(
    [ValidateSet("job", "sweep")]
    [string]$File = "job",
    [switch]$Local,
    [string]$LocalDataLocation = "/src/tmp/data",
    [string]$Subscription = "48bbc269-ce89-4f6f-9a12-c6f91fcb772d",
    [string]$ResourceGroup = "aml1p-rg",
    [string]$Workspace = "aml1p-ml-wus2"
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

az ml job create `
    --file "$($File).yml" `
    --subscription $Subscription `
    --workspace-name $Workspace `
    --resource-group $ResourceGroup `
    --query metadata.interaction_endpoints.studio `
    -o tsv
