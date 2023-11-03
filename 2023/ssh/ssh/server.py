import docker

from ssh.models import SshHost


def run_dockerized_server(host_config: SshHost, public_key: str):
    assert (
        host_config.port == 2222
    ), "https://github.com/linuxserver/docker-openssh-server/issues/30"
    client = docker.from_env()
    container = client.containers.run(
        "linuxserver/openssh-server:version-9.3_p2-r0",
        environment={
            "PUID": 1000,
            "PGID": 1000,
            "TZ": "Etc/UTC",
            "PUBLIC_KEY": public_key,
            "USER_NAME": host_config.user,
            # https://github.com/linuxserver/docker-openssh-server/issues/30#issuecomment-1525103465
            "LISTEN_PORT": host_config.port,
            "LOG_STDOUT": True,
        },
        # 0.0.0.0:2222->2222/tcp
        # A tuple of (address, port) if you want to specify the host interface. For example, {'1111/tcp': ('127.0.0.1', 1111)}.
        # TODO: host??
        ports={host_config.port: host_config.port},
        hostname=host_config.host,
        detach=True,
    )

    return container


# # https://hub.docker.com/r/linuxserver/openssh-server
# docker run \
#   -e PUID=1000 \
#   -e PGID=1000 \
#   -e TZ=Etc/UTC \
#   -e PUBLIC_KEY="$(cat ~/.ssh/id_rsa_$HEX.pub)" \
#   -e USER_NAME=$(whoami) \
#   -e LOG_STDOUT=true \
#   -p $PORT:$PORT \
#   linuxserver/openssh-server:version-9.3_p2-r0
