<#
Simple PowerShell wrapper for the Scala Build Tool (SBT).
#>

docker run -v "$($PWD):/src" -w /src bigtruedata/sbt sbt $args