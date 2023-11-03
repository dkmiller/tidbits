## This file runs a simple end-to-end example with non-Python client and server.
## Only the RSA public/private key pair is generated using Python code.

# https://coderwall.com/p/s8n9qa/default-parameter-value-in-bash
# https://stackoverflow.com/a/33419280/
PORT="${1:-2222}"

echo "Port $PORT"

export HEX=$(gen-rsa)
echo "Hex $HEX"

# https://hub.docker.com/r/linuxserver/openssh-server
docker run \
  -e PUID=1000 \
  -e PGID=1000 \
  -e TZ=Etc/UTC \
  -e PUBLIC_KEY="$(cat ~/.ssh/id_rsa_$HEX.pub)" \
  -e USER_NAME=$(whoami) \
  -e LOG_STDOUT=true \
  -p $PORT:$PORT \
  linuxserver/openssh-server:version-9.3_p2-r0

sleep 3

ssh -i ~/.ssh/id_rsa_$HEX -p $PORT $(whoami)@localhost

docker stop $(docker ps -a -q)

# https://stackoverflow.com/a/62309999/
sed -i '' "/localhost/d" ~/.ssh/known_hosts
