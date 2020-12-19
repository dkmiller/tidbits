param(
    $DockerImage='az-cli-terraform'
)

$Current = $PWD

try {
    docker build -t $DockerImage .
    docker run --volume ${PWD}:/src -w /src $DockerImage pwsh -File Foo.ps1
} finally {
    Set-Location $Current
}
