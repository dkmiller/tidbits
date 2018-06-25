'''
Simple Flask application. From:
https://docs.docker.com/get-started/part2/#apppy
'''

import jinja2
import os

t = jinja2.Template('- {{hi}} -')

result = t.render(hi = '|haha|')

print('Hello from Python!')
print(result)

envvar = os.getenv('NAME')

print('Environment variable:')
print(envvar)
