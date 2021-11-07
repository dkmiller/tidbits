"""
Imitate: https://stackoverflow.com/a/44712152 .
"""

import idx2numpy
import numpy as np
import os
from pathlib import Path
import sys


print(f"Arguments: {sys.argv}")


in_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])


files = in_dir.glob("*")


for in_path in files:
    out_path = out_dir / (in_path.name + ".npy")
    if os.path.isfile(in_path):
        print(f"{in_path} -> {out_path}")
        arr = idx2numpy.convert_from_file(str(in_path))
        np.save(str(out_path), arr)
