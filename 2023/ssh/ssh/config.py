from pathlib import Path

from ssh.models import SshHost


def host_config(host: SshHost, private_key: Path) -> str:
    """
    Section of `~/.ssh/config` corresponding to the provided host.

    https://phoenixnap.com/kb/ssh-config
    """
    return f"""
    Host {host.host}
        HostName {host.host}
        User {host.user}
        IdentityFile {str(private_key)}
        StrictHostKeyChecking accept-new
    """
