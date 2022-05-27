from argparse_dataclass import ArgumentParser
from bs4 import BeautifulSoup
from dataclasses import dataclass
import logging
import requests
import shutil
import tarfile
import tempfile
import zstandard


log = logging.getLogger(__name__)


@dataclass
class Args:
    train: str
    test: str
    val: str
    root: str = "https://mystic.the-eye.eu/public/AI/pile"


# https://stackoverflow.com/a/39217788/2543689
def download_file(url):
    local_filename = url.split("/")[-1]
    log.info(f"Downloading {url} to {local_filename}")
    with requests.get(url, stream=True) as r:
        with open(local_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    return local_filename


def extract_zst(source: str, target: str):
    log.info(f"Extract ZStandard from {source} to {target}")
    decompressor = zstandard.ZstdDecompressor()
    with tempfile.TemporaryFile(suffix=".tar") as temp:
        log.info(f"Extracting decompressed stream to {temp}")
        with open(source, "rb") as source_f:
            decompressor.copy_stream(source_f, temp)
        temp.seek(0)
        log.info(f"Extracting tarfile to {target}")
        with tarfile.open(fileobj=temp) as z:
            from pathlib import Path
            target = Path(target).expanduser().resolve()
            print(target)
            z.extractall(target)


def main(args: Args):
    pass

# download_file(_url)


# https://www.scivision.dev/python-extract-zstd/

#https://github.com/akshayrajeev/Index-Of-Downloader/blob/master/IndexDownloader.py

if __name__ == "__main__":
    parser = ArgumentParser(Args)
    args = parser.parse_args()
    logging.basicConfig(level="INFO")
    main(args)

extract_zst("/home/azureuser/tidbits/2021/2021-08-19_huggingface-reddit/00.jsonl.zst", "/home/azureuser/tmp3/")

##

from pywebcopy import Crawler, config
url = "https://mystic.the-eye.eu/public/AI/pile"
project_folder = '/home/azureuser/tmp4/'
config.setup_config(url, project_folder, "eleutherai")

wp = Crawler()

# If you want to you can use `requests` to fetch the pages
>>> wp.get(url, **{'auth': ('username', 'password')})

# Then you can access several methods like
>>> wp.crawl()