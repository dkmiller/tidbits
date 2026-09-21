"""
Process-related utilities.
"""
import logging
import shlex
from subprocess import PIPE, CompletedProcess, Popen, run

from ssh.abstractions import Result

log = logging.getLogger(__name__)


def popen(args: tuple[str, ...]) -> Popen:
    # https://stackoverflow.com/a/31867499/
    rv = Popen(args, stderr=PIPE, stdout=PIPE)
    # Make logged commands copy/pasteable.
    copyable_args = " ".join(map(shlex.quote, args))
    log.info("Spawned process %s: %s", rv.pid, copyable_args)
    return rv


def run_pipe(*args, **kwargs) -> CompletedProcess[bytes]:
    """
    Invoke `subprocess.run` with the provided args and piped standard error + output.
    """
    return run(*args, stderr=PIPE, stdout=PIPE, **kwargs)


def wait(process: Popen) -> Result:
    """
    Wait until a process finishes, then return decoded standard output/error with status code.
    """
    # https://stackoverflow.com/a/38956698/
    (output, error) = process.communicate()

    status = process.wait()
    return Result(error.decode(), output.decode(), status)


def kill(process: Popen) -> Result:
    # https://stackoverflow.com/a/43276598/
    poll = process.poll()
    pid = process.pid
    if poll:
        log.info("Process %s already terminated with status %s", pid, poll)
    else:
        log.info("Killing %s", pid)
        process.kill()

    return wait(process)
