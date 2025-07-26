import logging
from typing import Optional, Union

log = logging.getLogger(__name__)


class ScreenClient:
    def session(
        self, command: Union[str, list[str]], name: str, logfile: Optional[str] = None
    ) -> tuple[str, ...]:
        if isinstance(command, list):
            command = " ".join(command)

        log.info("Running %s", command)
        args = ("screen", "-S", name)
        if logfile:
            # Send screen session logs to the specified file:
            # https://stackoverflow.com/a/50651839/
            args += ("-L", "-Logfile", logfile)
        return args + ("-m", "-d", command)
