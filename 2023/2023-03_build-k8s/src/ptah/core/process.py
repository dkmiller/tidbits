import base64
import json
import os
import shlex
import sys
from subprocess import Popen
from typing import List

import psutil


class ProcessClient:
    def run(self, args: List[str]) -> None:
        # https://stackoverflow.com/a/847800
        command = " ".join(map(shlex.quote, args))
        print(f"Running {command}")
        # TODO: retry logic.
        # TODO: propagate "kill" logic: https://stackoverflow.com/a/43323376
        # lsof -n -i :8001
        val = os.system(command)
        print(f"Return code: {val}")

    def encode(self, input: List[str]) -> str:
        serialized = json.dumps(input)
        input_b = serialized.encode()
        input_b64 = base64.b64encode(input_b)
        rv = input_b64.decode()
        print(f"Base 64 + JSON encode: {input} --> {rv}")
        return rv

    def decode(self, input: str) -> List[str]:
        input_b = input.encode()
        input_b64 = base64.b64decode(input_b)
        decoded = input_b64.decode()
        rv = json.loads(decoded)
        print(f"Base 64 + JSON decode: {input} --> {rv}")
        return rv

    def decode_and_run(self, arg: str):
        decoded = self.decode(arg)
        # args = json.loads(decoded)
        # command = self.command(args)
        self.run(decoded)

    def spawn(self, args: List[str]):
        # https://stackoverflow.com/a/27625288/
        # serialized = json.dumps(args)
        encoded = self.encode(args)
        print(f"Running {sys.executable} -m {__name__} {encoded}")
        p = Popen([sys.executable, "-m", __name__, encoded])
        print(f"Kicked off {p.pid}")

    def find(self, args: List[str]) -> List[psutil.Process]:
        #         [p for p in psutil.process_iter(["cmdline"]) if "ptah" in str(p.info["cmdline"])][0].info["cmdline"]
        # ['/Users/dan/miniforge3/bin/python3.9', '-m', 'ptah.core.process', 'WyJrdWJlY3RsIiwgInByb3h5Il0=']
        encoded = self.encode(args)
        rv = []
        for p in psutil.process_iter(["cmdline"]):
            try:
                args = p.info["cmdline"]
                if (
                    args[0] == sys.executable
                    and args[1] == "-m"
                    and args[2] == __name__
                    and args[3] == encoded
                ):
                    rv.append(p)
            except:
                # Don't log anything, there will be a surprising number of errors.
                pass
        return rv

    def kill(self, args: List[str]):
        processes = self.find(args)
        for process in processes:
            process.terminate()  # .kill()
        else:
            print(f"Found no processes running `{args}`")

    def ensure(self, args: List[str]):
        processes = self.find(args)
        if processes:
            print(f"Processes running `{args}` already found: {[p.pid for p in processes]}")
        else:
            self.spawn(args)


# list(psutil.process_iter(["cmdline"]))[-1].info["cmdline"]

#     def kill(self, args: List[str]):
#         # /Users/dan/miniforge3/bin/python3.9 -m ptah.core.process WyJrdWJlY3RsIiwgInByb3h5Il0=
#         impo


# import subprocess

# # https://stackoverflow.com/a/27625288/2543689
# subprocess.Popen([""])
# python -c "import psutil; print(list(psutil.process_iter([]))[0])"
# python -c "import psutil; print([x.name() for x in psutil.process_iter([])])"

# [p.info["cmdline"] for p in psutil.process_iter([])]

# [p for p in psutil.process_iter([]) if "ptah" in str(p.info["cmdline"])][0].info["cmdline"]

# os.system("kubectl port-forward deployment/api-deployment 8000:8000")

# >>> [p for p in psutil.process_iter([]) if "kubectl" in str(p.info["cmdline"])][0].info["cmdline"]
# ['kubectl', 'port-forward', 'deployment/api-deployment', '8000:8000']

if __name__ == "__main__":
    import sys

    ProcessClient().decode_and_run(sys.argv[-1])
