"""
Imitate: https://stackoverflow.com/a/44712152 .
"""

import gzip
from pathlib import Path
import shutil
import sys


print(f"Arguments: {sys.argv}")


in_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])


gz_files = in_dir.glob("*.gz")


for in_path in gz_files:
    out_path = out_dir / in_path.name.replace(".gz", "")
    print(f"{in_path} -> {out_path}")
    with gzip.open(str(in_path), "rb") as f_in:
        with open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
