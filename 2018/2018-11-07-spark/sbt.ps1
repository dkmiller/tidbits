<#
Simple PowerShell wrapper for the Scala Build Tool (SBT).
#>

docker run --volume "$($PWD):/src" --workdir /src bigtruedata/sbt sbt $args