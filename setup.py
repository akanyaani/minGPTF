from setuptools import setup, find_packages
import os

with open('./requirements.txt') as f:
    required = f.read().splitlines()
    print(required)

setup(
    name='minGPTF',
    version="0.0.1",
    url="https://github.com/akanyaani/minGPTF.git",
    packages=find_packages(),
    include_package_data=True,
    package_data={'': ['data/*']},
    install_requires=required,
)
