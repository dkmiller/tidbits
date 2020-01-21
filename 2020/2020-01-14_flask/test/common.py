from urllib import request

def get_bytes(url):
    '''
    Get the bytes from a URL.
    '''
    response = request.urlopen(url)
    return response.read()
