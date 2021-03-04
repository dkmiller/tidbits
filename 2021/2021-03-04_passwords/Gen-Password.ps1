<#
Generate GUID-based password with both lower AND upper case letters.
#>

$Guid = [guid]::NewGuid().ToString()

$Result = ""

foreach ($c in [char[]]$guid) {
  $b = ($true, $false) | Get-Random;
  if ($b) {
    $Result += ([string]$c).toupper()
  } else {
    $Result += $c
  }
}

Write-Host $Result
