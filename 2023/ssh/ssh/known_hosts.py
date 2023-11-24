import logging
from subprocess import PIPE, check_output

from ssh.models import SshHost

log = logging.getLogger(__name__)


class KnownHostsClient:
    """
    Wrap necessary functionality for managing the standard "known hosts" file at
    `~/.ssh/known_hosts`.
    """

    def reset(self, host: SshHost) -> None:
        """
        In unit and integration testing, it's necessary to clean up known hosts configuration
        between tests giving the same localhost reference different public/private key pairs.

        Replaces the simple shell call

        ```bash
        sed -i '' "/localhost/d" ~/.ssh/known_hosts
        ```
        """
        # https://superuser.com/a/30089
        output = check_output(
            ["ssh-keygen", "-R", f"[{host.host}]:{host.port}"], stderr=PIPE
        )
        log.info(output)
