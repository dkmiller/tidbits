import logging
from dataclasses import dataclass
from subprocess import Popen

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Result:
    stderr: str
    stdout: str
    status: int


def wait(process: Popen) -> Result:
    """
    Wait until a process finishes, then return decoded standard output/error with status code.
    """
    # https://stackoverflow.com/a/38956698/
    (output, error) = process.communicate()

    status = process.wait()
    return Result(error.decode(), output.decode(), status)


def kill(process: Popen) -> None:
    # https://stackoverflow.com/a/43276598/
    poll = process.poll()
    pid = process.pid
    if poll:
        log.info("Process %s already terminated with status %s", pid, poll)
    else:
        log.info("Killing %s", pid)
        process.kill()
