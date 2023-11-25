import asyncio
import logging
from dataclasses import dataclass
from subprocess import Popen

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Result:
    stderr: str
    stdout: str
    status: int


async def aexec(*args: str):
    process = await asyncio.create_subprocess_exec(
        args[0],
        *args[1:],
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
    )
    (stdout, stderr) = await process.communicate()
    return_code = await process.wait()
    from ssh.process import Result

    return Result((stderr or b"").decode(), stdout.decode(), return_code)


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
