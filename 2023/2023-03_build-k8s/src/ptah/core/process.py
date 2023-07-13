import base64
import json
import signal
import sys
from subprocess import Popen
from typing import List

import psutil


def _sigterm_handler(_signo, _stack_frame):
    """
    Convert the Posix SIGTERM into a Python exception which may be caught and handled:
    https://stackoverflow.com/a/24574672
    """
    sys.exit(0)


class ProcessClient:
    def run(self, args: List[str]) -> None:
        """
        Run the process `args`, propagating terminate signals. You can debug for processes on a
        port with

        ```bash
        lsof -n -i :8001
        ```
        """
        print(f"Running {args}")

        process = Popen(args)

        try:
            # https://stackoverflow.com/a/15108096
            return_code = process.wait()
            print(f"Return code: {return_code}")
        finally:
            # https://stackoverflow.com/a/43323376
            print(f"Killing {process.pid}")
            process.kill()

    def encode(self, input: List[str]) -> str:
        """
        JSON-serialize, then Base64-encode, a list of strings.
        """
        serialized = json.dumps(input)
        input_b = serialized.encode()
        input_b64 = base64.b64encode(input_b)
        rv = input_b64.decode()
        return rv

    def decode(self, encoded: str) -> List[str]:
        """
        Base64-decode, then JSON-deserialize, a string into a list of strings.
        """
        encoded_b = encoded.encode()
        encoded_b64 = base64.b64decode(encoded_b)
        decoded = encoded_b64.decode()
        rv = json.loads(decoded)
        return rv

    def spawn(self, args: List[str]) -> None:
        # https://stackoverflow.com/a/27625288/
        encoded = self.encode(args)
        print(f"Running {sys.executable} -m {__name__} {encoded}")
        p = Popen([sys.executable, "-m", __name__, encoded])
        print(f"Kicked off {p.pid}")

    def find(self, args: List[str]) -> List[psutil.Process]:
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

    def terminate(self, args: List[str]):
        processes = self.find(args)
        for process in processes:
            print(f"Terminating {process.pid}")
            process.terminate()
        if not processes:
            print(f"Found no processes running `{args}`")

    def ensure(self, args: List[str]):
        processes = self.find(args)
        if processes:
            print(f"Processes running `{args}` already found: {[p.pid for p in processes]}")
        else:
            self.spawn(args)


if __name__ == "__main__":
    pc = ProcessClient()
    decoded = pc.decode(sys.argv[-1])
    signal.signal(signal.SIGTERM, _sigterm_handler)
    pc.run(decoded)
