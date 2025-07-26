import re
from pathlib import Path
import pandas as pd


def date(path: Path):
    # https://stackoverflow.com/a/15474798/
    path_s = str(path.absolute().resolve())
    m = re.search(r"(\d{4})[\-\./](\d{1,2})[\-\./](?:\d{1,2})?", path_s)
    if m:
        try:
            suffix = m.group(3)
        except:
            suffix = 1
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(suffix):02d}"
    else:
        return pd.NA



def load_raw(root: str, glob: str) -> pd.DataFrame:
    paths = list(Path(root).glob(glob))
    df = pd.DataFrame(data=paths, columns=["path"])
    df["date"] = df["path"].apply(date)
    df["text"] = [p.read_text() for p in paths]
    # https://stackoverflow.com/a/22006514/
    df["path"] = df["path"].astype(str)
    return df
