'''
Functional programming 
'''


def identity(x):
    '''
    Identity function, defined here because the Python team has explicitly
    rejected the proposal to include one in the standard library:
    https://bugs.python.org/issue1673203 .
    '''
    return x


def compose(*args):
    '''
    Compose functions f1, f2, f3, ... as x -> ...f3(f2(f1(x))). Defined here
    because the Python team has explicitly rejected the proposal to include one
    in the standard library: https://bugs.python.org/issue1506122 .
    '''

    def result(x):
        for f in reversed(args):
            x = f(x)
        return x
    return result
