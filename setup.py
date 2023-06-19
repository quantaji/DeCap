import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name='decap',
    py_modules=['decap'],
    version='0.1',
    description='A third-party minimal realization of DeCap.',
    author='',
    packages=find_packages(),
    install_requires=[str(r) for r in pkg_resources.parse_requirements(open(os.path.join(
        os.path.dirname(__file__),
        'requirements.txt',
    )))],
    extras_require={'dev': ['pytest']},
)
