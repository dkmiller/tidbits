import os
import pathlib
import requests

def data_directory():
    '''
    Common directory for all testing data.
    '''
    pwd = pathlib.Path(__file__).parent.absolute()
    return os.path.join(pwd, '..', '.data')

def path_in_data_directory(filename, url):
    '''
    Download if not already, and if already exists return a working path to
    the file version of the URL.
    '''
    directory = data_directory()
    result = os.path.join(directory, filename)
    if not os.path.exists(result):
        os.makedirs(directory, exist_ok=True)
        r = requests.get(url)
        with open(result, 'wb') as f:
            f.write(r.content)
    return result
