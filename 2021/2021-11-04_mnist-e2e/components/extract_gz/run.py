"""
Imitate: https://stackoverflow.com/a/44712152 .
"""

import gzip
import shutil
import sys


with gzip.open('file.txt.gz', 'rb') as f_in:
    with open('file.txt', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)