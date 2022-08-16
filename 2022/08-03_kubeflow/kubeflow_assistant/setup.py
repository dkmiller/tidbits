# Imitate:
# https://python-packaging.readthedocs.io/en/latest/minimal.html

from setuptools import setup

# TODO: imitate
# - https://git.musta.ch/airbnb/data/blob/master/airflow/teams/experimentation/projects/jitneyclient/setup.py

setup(
    name="kubeflow-assistant",
    version="0.0",
    description="Assist submission of local Kubeflow pipelines",
    url="https://github.com/dkmiller/tidbits/",
    author="Dan Miller",
    author_email="dan.miller@airbnb.com",
    license="MIT",
    packages=["kubeflow_assistant"],
    zip_safe=False,
)
