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


# RUN mkdir -p /root/.elan \
#   && chown -R 1000:1000 /root/.elan \
#   && usermod -a -G {group-name} username
#   && chmod -R g+rwx vscode /root/.elan

# USER 1000
# RUN touch /root/.elan/foo



# ENV PATH="$PATH:/root/.elan/bin"
# RUN chown -R 1000:1000 /root/.elan

# https://stackoverflow.com/a/45973108
# RUN mkdir -p /root/.elan \
#   && chown -R vscode /root/.elan


# # USER vscode
# # RUN touch /root/.elan/foo


# # TODO: give user "vscode" access to Elan / Lean installation.

# USER vscode
# # RUN which lean

# # RUN elan default leanprover/lean4:stable
