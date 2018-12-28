<#
Simple wrapper around scala.exe.
#>

docker run --volume "$($PWD):/src" --workdir /src bigtruedata/scala scala $args