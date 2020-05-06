{ name = "my-project"
, dependencies = [ "console", "effect", "math", "psci-support" ]
, packages = ./packages.dhall
, sources = [ "src/**/*.purs", "test/**/*.purs" ]
}
