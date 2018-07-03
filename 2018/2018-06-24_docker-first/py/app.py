'''
Simple Python application with a "non-standard" set of imports.
'''

import jinja2
import os
import redis
import sys

t = jinja2.Template('- {{hi}} -')

result = t.render(hi = '|haha|')

print('Hello from Python!')
print(result)

envvar = os.getenv('NAME')

print('Environment variable:')
print(envvar)

print('Command line arguments:')
print(', '.join(sys.argv))
