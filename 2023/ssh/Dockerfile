# Variant of the OpenSSH server:
# https://github.com/linuxserver/docker-openssh-server
# with boilerplate configuration set and port forwarding enabled.
FROM linuxserver/openssh-server:version-9.3_p2-r0

RUN apk add --no-cache --upgrade screen

ENV PUID=1000
ENV PGID=1000
ENV TZ=Etc/UTC
ENV LOG_STDOUT=true

# Configs to tweak: https://github.com/caprover/caprover/issues/960#issuecomment-1101508239
# Edit files: https://stackoverflow.com/a/27714071/
RUN sed -i "s|AllowTcpForwarding no|AllowTcpForwarding yes|g" /etc/ssh/sshd_config \
  && sed -i "s|GatewayPorts no|GatewayPorts yes|g" /etc/ssh/sshd_config
