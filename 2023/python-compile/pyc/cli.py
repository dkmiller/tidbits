import sys
from omegaconf import OmegaConf


if __name__ == "__main__":
    arg = sys.argv[1]
    print(f"Hi from pyc: loading {arg}")
    conf = OmegaConf.load(arg)
    print(conf)
