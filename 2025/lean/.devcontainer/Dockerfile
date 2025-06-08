# https://github.com/devcontainers/images/tree/main/src/base-ubuntu
FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

USER vscode

# https://github.com/leanprover/elan
# https://stackoverflow.com/a/53605439
RUN curl https://elan.lean-lang.org/elan-init.sh -sSf | sh -s -- -v -y

ENV PATH="$PATH:/home/vscode/.elan/bin"

# https://docs.lean-lang.org/lean4/doc/setup.html#setting-up-lean
RUN elan default leanprover/lean4:stable

# Trigger a download
RUN lean --version
