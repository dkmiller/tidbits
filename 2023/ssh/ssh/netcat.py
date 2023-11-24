import logging
import shlex

log = logging.getLogger(__name__)

# https://unix.stackexchange.com/q/289364
# https://superuser.com/a/115556
# https://stackoverflow.com/a/19139134/


class NetcatClient:
    """
    The OpenSSH Docker image is very limited, with the only natural HTTP host being the
    `nc` executable (https://unix.stackexchange.com/a/715981).
    """

    def http_response(self, body: str) -> str:
        """
        Construct the raw text for a successful plaintext HTTP response containing the specified
        body.
        """
        rv = f"""HTTP/1.1 200 OK
Content-Type: text/plain
Connection: close
content-length: {len(body)}
{body}"""
        log.info("Raw HTTP Response:\n%s", rv)
        return rv

    def remote_bash_command(self, http_response: str, remote_port: int) -> str:
        """
        It's surprisingly tricky to "stuff" a netcat session with pre-configured HTTP response into
        a remote SSH host. This returns a command usable inside a remote bash session.
        """
        raw_command = f'echo -e "{http_response}" | nc -l localhost {remote_port}'
        # Escaping newlines: https://stackoverflow.com/a/15392758/
        rv = raw_command.encode("unicode_escape").decode()
        log.info("Remote bash command: %s", rv)
        return rv

    def ssh_exec(self, body: str, remote_port: int) -> list[str]:
        http_response = self.http_response(body)
        remote_command = self.remote_bash_command(http_response, remote_port)
        quoted_command = shlex.quote(remote_command)
        return ["/bin/bash", "-c", quoted_command]
