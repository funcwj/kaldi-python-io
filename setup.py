#!/usr/bin/env python

from setuptools import setup, find_packages

VERSION = "1.2.0"

with open("README.md", "r") as f:
    long_description = f.read()

setup(name="kaldi-python-io",
      version=VERSION,
      description="A pure python IO interface for data accessing in kaldi",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/funcwj/kaldi-python-io",
      author="Jian Wu",
      author_email="funcwj@foxmail.com",
      packages=["kaldi_python_io"],
      install_requires=["numpy>=1.14"],
      python_requires=">=3.6",
      license="Apache V2.0")
