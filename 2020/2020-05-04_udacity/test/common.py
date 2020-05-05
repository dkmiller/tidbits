import os
import pathlib

def _data_directory():
    pwd = pathlib.Path(__file__).parent.absolute()
    return os.path.join(pwd, '..', '.data')

def titanic_csv():
    pass
