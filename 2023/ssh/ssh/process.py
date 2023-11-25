"""
https://superfastpython.com/asyncio-create_subprocess_shell/

https://docs.python.org/3/library/asyncio-subprocess.html

?? https://superfastpython.com/asyncio-background-task/
"""
import asyncio
import logging
import shlex
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Result:
    stderr: str
    stdout: str
    status: int


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
