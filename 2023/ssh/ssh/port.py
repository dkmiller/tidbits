import subprocess

import pytest


def process_holding(port: int):
    # Get only the port from lsof: https://stackoverflow.com/a/62453482/
    return subprocess.check_output(["lsof", "-t", "-i", f":{port}"])


def assert_free(port: int):
    with pytest.raises(subprocess.CalledProcessError):
        process_holding(port)
