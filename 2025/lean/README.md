# Lean

Open this folder in the configured VS Code Dev Container, then type
`⌘ + SHIFT + B` or explicitly run the command below to build and run the Lean program
here.

``` bash
./run.sh
```

## Links

- [Create a dev container &gt; Dockerfile](https://code.visualstudio.com/docs/devcontainers/create-dev-container#_dockerfile)
- [Liquid tensor experiment](https://xenaproject.wordpress.com/2020/12/05/liquid-tensor-experiment/)
- [Natural number game](https://adam.math.hhu.de/#/g/leanprover-community/nng4)
- [Theorem proving in Lean 4](https://lean-lang.org/theorem_proving_in_lean4/title_page.html)
- [Lean projects](https://leanprover-community.github.io/install/project.html)
- [How to add the custom compile commands in VS Code?](https://stackoverflow.com/a/71747570)

## Misc commands

``` bash
# Manually build the Docker image.
docker build . -t lean

# If you want to create a new project.
lake +leanprover/lean4:v4.20.1 new ProjectName
```
