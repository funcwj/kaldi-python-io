#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="kaldi-python-io",
    version="0.1",
    description="A pure python IO interface for data accessing in kaldi",
    url="https://github.com/funcwj/kaldi-python-io",
    author="Jian Wu",
    author_email="funcwj@foxmail.com",
    packages=["kaldi_python_io"],
    install_requires=["numpy>=1.14"],
    license="Apache V2.0")
