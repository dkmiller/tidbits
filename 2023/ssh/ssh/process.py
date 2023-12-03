"""
https://superfastpython.com/asyncio-create_subprocess_shell/

https://docs.python.org/3/library/asyncio-subprocess.html

?? https://superfastpython.com/asyncio-background-task/
"""
import asyncio
import logging
import shlex
from dataclasses import dataclass
from subprocess import Popen

from ssh.abstractions import Result

log = logging.getLogger(__name__)


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


async def spawn(args: tuple[str, ...]):
    """
    Spawn an asynchronous process and return a reference to the spawned process.
    """
    process = await asyncio.create_subprocess_exec(
        args[0],
        *args[1:],
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
    )
    copyable_args = " ".join(map(shlex.quote, args))
    log.info("Spawned process %s: %s", process.pid, copyable_args)
    return process


async def spawn_and_wait(args: tuple[str, ...]):
    """
    Spawn an asynchronous process, wait for it to finish, then return an implementation-agnostic
    "wrapper" of the standard output, error, and return code.
    """
    process = await spawn(args)
    (stdout, stderr) = await process.communicate()
    return_code = await process.wait()

    if stdout is not None:
        stdout = stdout.decode()

    if stderr is not None:
        stderr = stderr.decode()

    return Result(stderr, stdout, return_code)
