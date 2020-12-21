param(
    [ValidateSet("job", "sweep")]
    [string]$File = "job"
)


git clean -xdf

az ml job create --file "$($File).yml" --query metadata.interaction_endpoints.studio --out tsv
