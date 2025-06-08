# Lean

https://code.visualstudio.com/docs/devcontainers/create-dev-container#_dockerfile

``` bash
docker build . -t lean

# If you want to create a new project.
lake +leanprover/lean4:v4.20.1 new foo__

lake build

.lake/build/bin/src
```

## Links

- https://xenaproject.wordpress.com/2020/12/05/liquid-tensor-experiment/
- https://adam.math.hhu.de/#/g/leanprover-community/nng4/world/Tutorial/level/3
- https://lean-lang.org/theorem_proving_in_lean4/title_page.html
- https://leanprover-community.github.io/install/project.html
