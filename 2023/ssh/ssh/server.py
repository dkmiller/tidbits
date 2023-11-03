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
        ports={host_config.port: host_config.port},
        hostname=host_config.host,
        detach=True,
    )

    return container
