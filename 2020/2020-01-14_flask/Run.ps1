<#
Run the Flask app, either using the local Python installation,
or using Docker.
#>

param (
    [switch]$SkipBuild,
    [switch]$Local
)

if (!($SkipBuild -or $Local)) {
    .\Build.ps1
}

if ($Local) {
    $env:FLASK_ENV = 'development'
    $env:FLASK_APP = 'src/app.py'
    flask run
}
else {
    docker run -p 5000:5000 danmill/flask
}
