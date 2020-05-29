# import sys
import importlib

# print(sys.modules.keys())

import pkgutil

# https://stackoverflow.com/a/9194180
modules = list(pkgutil.iter_modules())

def __evil_(module_name: str):
    result = []
    stack = [module_name]

    while len(stack) > 0:
        current = stack.pop()
        result.append(current)
        try:
            i = importlib.import_module(current)
            d = dir(i)
            stack.extend([f'{current}.{n}' for n in d])
        except ModuleNotFoundError:
            # Found a method or class.
            pass
    return result


# print(__evil_('torch'))

# print(1/0)

counter = 1

for m in modules:
    print(f'{m.name} {counter}')
    counter += 1
    # https://stackoverflow.com/a/8719100
    try:
        if m.name != 'win32traceutil':
            # print(m.name)
            ms = __evil_(m.name)
            print(f'\t{len(ms)}')
        # if m.name == 'torch':
        #     i = importlib.import_module(m.name)
        #     print(f'\t{dir(i)}')
    except Exception:
        print('\t----')

#print(modules[1])

# print(dir(modules[1].name))
