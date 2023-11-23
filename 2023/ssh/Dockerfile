# Variant of the OpenSSH server:
# https://github.com/linuxserver/docker-openssh-server
# with boilerplate configuration set and port forwarding enabled.
FROM linuxserver/openssh-server:version-9.3_p2-r0

ENV PUID=1000
ENV PGID=1000
ENV TZ=Etc/UTC
ENV LOG_STDOUT=true

# Configs to tweak: https://github.com/caprover/caprover/issues/960#issuecomment-1101508239
# Edit files: https://stackoverflow.com/a/27714071/
RUN sed -i "s|AllowTcpForwarding no|AllowTcpForwarding yes|g" /etc/ssh/sshd_config \
  && sed -i "s|GatewayPorts no|GatewayPorts yes|g" /etc/ssh/sshd_config

# ===== Build SSH server =====
# docker build -t ssh .

# ===== Start SSH server =====
# docker run -e PUBLIC_KEY="$(cat ~/.ssh/id_rsa_16505ade1dbd42f38623fd2aef236a27.pub)" -e USER_NAME=$(whoami) -p 2222:2222 -p 24464:24464 ssh

# ===== Clean up known_hosts =====
# sed -i '' "/localhost/d" ~/.ssh/known_hosts

# ===== SSH into server to ensure it's up =====
# ssh -i ~/.ssh/id_rsa_16505ade1dbd42f38623fd2aef236a27 -p 2222 -o StrictHostKeyChecking=accept-new dan@localhost

# ===== Start port forwarding =====
# ssh -i ~/.ssh/id_rsa_16505ade1dbd42f38623fd2aef236a27 -p 2222 -o StrictHostKeyChecking=accept-new -fN -L 63752:localhost:24464 dan@localhost


# ===== From inside SSH session (https://stackoverflow.com/a/19139134/) =====
# ===== Content type: https://stackoverflow.com/a/1414329/
# ===== date: https://stackoverflow.com/a/1092643/
# Headers reverse-engineered from https://httpbin.org/#/Dynamic_data/get_uuid
# echo -e "HTTP/1.1 200 OK\nContent-Type: text/plain\nConnection: close\ncontent-length: 57\n\nHi from ssh server 50ca245d-a96e-47b1-a042-180dd751ee77\n" | nc -l localhost 24464


# ==== Escaping is a b***
# ssh -i ~/.ssh/id_rsa_16505ade1dbd42f38623fd2aef236a27 -p 2222 -o StrictHostKeyChecking=accept-new dan@localhost echo 'echo -e ""HTTP/1.1'

# 'echo -e ""HTTP/1.1 200 OK\\nContent-Type: text/plain\nConnection: close\\ncontent-length: 57\\n\\nHi from ssh server 50ca245d-a96e-47b1-a042-180dd751ee77\\n"" \| nc -l localhost 24464'

# ===== Remote machine has `GNU bash, version 5.2.15(1)-release (aarch64-alpine-linux-musl)`.

# ==== Output as same as bare `which`
# ssh -i ~/.ssh/id_rsa_16505ade1dbd42f38623fd2aef236a27 -p 2222 -o StrictHostKeyChecking=accept-new dan@localhost /bin/bash -c 'which ls'

# ===== Works as expected!
# ssh -i ~/.ssh/id_rsa_16505ade1dbd42f38623fd2aef236a27 -p 2222 -o StrictHostKeyChecking=accept-new dan@localhost /bin/bash -c 'which\ ls'

# ===== Also works as expected (https://stackoverflow.com/a/28120935/)
# subprocess.check_output(["ssh", "-i", "~/.ssh/id_rsa_16505ade1dbd42f38623fd2aef236a27", "-p", "2222", "-o", "StrictHostKeyChecking=accept-new", "dan@localhost", "/bin/bash", "-c", shlex.quote("which ls")]


# import subprocess
# import shlex
# subprocess.check_output(["ssh", "-i", "~/.ssh/id_rsa_16505ade1dbd42f38623fd2aef236a27", "-p", "2222", "-o", "StrictHostKeyChecking=accept-new", "dan@localhost", "/bin/bash", "-c", shlex.quote("which ls")])
# ssh -i ~/.ssh/id_rsa_16505ade1dbd42f38623fd2aef236a27 -p 2222 -o StrictHostKeyChecking=accept-new dan@localhost /bin/bash -c 'echo\ -e\ hi'

# ssh -i ~/.ssh/id_rsa_16505ade1dbd42f38623fd2aef236a27 -p 2222 -o StrictHostKeyChecking=accept-new dan@localhost /bin/bash -c 'echo\ -e\ ""HTTP/1.1 200 OK\\nContent-Type: text/plain\nConnection: close\\ncontent-length: 57\\n\\nHi from ssh server 50ca245d-a96e-47b1-a042-180dd751ee77\\n"" \| nc -l localhost 24464'


# ssh -i ~/.ssh/id_rsa_16505ade1dbd42f38623fd2aef236a27 -p 2222 -o StrictHostKeyChecking=accept-new dan@localhost bash -c 'echo -e "HTTP/1.1 200 OK\\nContent-Type: text/plain\\nConnection: close\\ncontent-length: 57\\n\\nHi from ssh server 1eb00d67-c06b-416f-aaa2-2d10d14bd64c\\n" | nc -l localhost 24464'
# # ssh -i ~/.ssh/id_rsa_16505ade1dbd42f38623fd2aef236a27 -p 2222 -o StrictHostKeyChecking=accept-new dan@localhost bash -c 'echo "HTTP/1.1 200 OK\\\nContent-Type: text/plain\\\nConnection: close\\\ncontent-length: 57\\\n\\\nHi from ssh server 1eb00d67-c06b-416f-aaa2-2d10d14bd64c\\\n" | nc -l localhost 24464'

# ===== From outside SSH session, works! but
# ===== no chunk, no close, no size. Assume close to signal end
# curl -v http://localhost:63752/foo
# python -c 'import requests; print(requests.get("http://localhost:63752/foo", timeout=2).text)'

# while true ; do nc -l -p 1500 -e /path/to/yourprogram ; done
# ssh -i ~/.ssh/id_rsa_16505ade1dbd42f38623fd2aef236a27 -p 2222 -o StrictHostKeyChecking=no dan@localhost

# ===== Who is listening on remote port
# lsof -i :24464
# ===== Who is listening on local port
# lsof -i :63752
