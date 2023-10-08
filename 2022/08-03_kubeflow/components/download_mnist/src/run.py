from aiohttp import ClientSession
from argparse_dataclass import dataclass
import asyncio
import idx2numpy
import io
import gzip
import numpy as np
from pathlib import Path


@dataclass
class Args:
    train: str
    test: str
    images_file: str = "images.npy"
    labels_file: str = "labels.npy"
    train_images: str = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
    train_labels: str = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
    test_images: str = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
    test_labels: str = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"


async def extract_and_convert(session: ClientSession, url: str) -> np.ndarray:
    """
    Download, extract, and convert a URL for a Gzip-style compressed `.idx`
    file, returning the resulting array.
    """
    print(f"Downloading {url}")
    async with session.get(url) as r:
        _bytes = await r.content.read()
        print(f"Downloaded compressed file of {len(_bytes)} bytes")
        # https://linuxhint.com/python-gzip-decompress-function/
        decompressed = gzip.decompress(_bytes)
        arr = idx2numpy.convert_from_file(io.BytesIO(decompressed))
        print(f"Decompressed array has shape {arr.shape}")
        return arr


def numpy_save(arr: np.ndarray, base: str, name: str):
    """
    Seems like Kubeflow doesn't create the parent directory.
    """
    base_path = Path(base)
    base_path.mkdir(parents=True, exist_ok=True)
    np.save(base_path / name, arr)


async def main(args: Args):
    print(f"Parsed args: {args}")
    async with ClientSession() as session:
        couroutines = [
            extract_and_convert(session, args.train_images),
            extract_and_convert(session, args.train_labels),
            extract_and_convert(session, args.test_images),
            extract_and_convert(session, args.test_labels),
        ]

        train_x, train_y, test_x, test_y = await asyncio.gather(*couroutines)

        numpy_save(train_x, args.train, args.images_file)
        numpy_save(train_y, args.train, args.labels_file)
        numpy_save(test_x, args.test, args.images_file)
        numpy_save(test_y, args.test, args.labels_file)


if __name__ == "__main__":
    args = Args.parse_args()  # type: ignore
    asyncio.run(main(args))
