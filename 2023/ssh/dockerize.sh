## This file runs a simple end-to-end example with non-Python client and server.
## Only the RSA public/private key pair is generated using Python code.

# https://coderwall.com/p/s8n9qa/default-parameter-value-in-bash
# https://stackoverflow.com/a/33419280/
PORT="${1:-2222}"

echo "Port $PORT"

export HEX=$(gen-rsa)
echo "Hex $HEX"

echo "ssh -i ~/.ssh/id_rsa_$HEX -p $PORT $(whoami)@localhost"

# https://hub.docker.com/r/linuxserver/openssh-server
docker run \
  -e PUID=1000 \
  -e PGID=1000 \
  -e TZ=Etc/UTC \
  -e PUBLIC_KEY="$(cat ~/.ssh/id_rsa_$HEX.pub)" \
  -e USER_NAME=$(whoami) \
  -e LOG_STDOUT=true \
  -p $PORT:$PORT \
  -p 24464:24464 \
  linuxserver/openssh-server:version-9.3_p2-r0

sleep 3



# ssh -i ~/.ssh/id_rsa_16505ade1dbd42f38623fd2aef236a27 -p 2222 -o StrictHostKeyChecking=accept-new dan@localhost
# echo "Hi" | nc -l -p 24464
# ssh -i ~/.ssh/id_rsa_ae8421c6652f4568bf53bcb18011698b -p 2222 -o StrictHostKeyChecking=accept-new -fN -L 24464:localhost:24464 dan@localhost

# ERROR:
# refused local port forward: originator

# TODO: https://serverfault.com/questions/899848/remote-port-forwarding-inside-docker-containers

# docker stop $(docker ps -a -q)

# # https://stackoverflow.com/a/62309999/
# sed -i '' "/localhost/d" ~/.ssh/known_hosts

# TODO: edit /config/ssh_host_keys/sshd_config in the Docker container to
# set
# AllowTcpForwarding yes
# GatewayPorts yes
# (https://github.com/caprover/caprover/issues/960#issuecomment-1101508239)
