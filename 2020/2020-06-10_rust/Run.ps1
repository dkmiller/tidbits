<#
Run the code using Docker.
#>

param(
    [ValidateSet('all', 'cargo')]
    $Command = 'cargo'
)

docker run --volume "$($PWD):/src" --workdir /src rust make $Command
