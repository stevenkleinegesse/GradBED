from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="GradBED",
    author="Steven Kleinegesse",
    description=("Gradient-Based BED for Implicit Models using MI Lower Bounds"),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/stevenkleinegesse",
    packages=['GradBED']
)
