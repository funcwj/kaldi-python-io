#!/usr/bin/env python

# wujian@2019

import numpy as np

from kaldi_python_io import ScriptReader, ArchiveReader
from kaldi_python_io import AlignArchiveReader, Nnet3EgsReader
from kaldi_python_io import ArchiveWriter


def test_archive_writer(ark, scp):
    # for matrix
    with ArchiveWriter(ark, scp) as writer:
        for i in range(10):
            mat = np.random.rand(100, 20)
            writer.write("mat-{:d}".format(i), mat)
    scp_reader = ScriptReader(scp)
    for key, mat in scp_reader:
        print("{0}: {1}".format(key, mat.shape))
    # for vector
    with ArchiveWriter(ark, scp, matrix=False) as writer:
        for i in range(10):
            vec = np.random.rand(100)
            print(vec)
            writer.write("vec-{:d}".format(i), vec)
    scp_reader = ScriptReader(scp, matrix=False)
    for key, vec in scp_reader:
        print(vec)
        print("{0}: {1}".format(key, vec.size))
    print("TEST *test_archieve_writer* DONE!")


def test_archive_reader(ark_or_pipe, matrix=True):
    ark_reader = ArchiveReader(ark_or_pipe, matrix=matrix)
    for key, obj in ark_reader:
        print("{0}: {1}".format(key, obj.shape))
    print("TEST *test_archive_reader* DONE!")


def test_script_reader(scp, matrix=True):
    scp_reader = ScriptReader(scp, matrix=matrix)
    for key, obj in scp_reader:
        print("{0}: {1}".format(key, obj.shape))
    print("TEST *test_script_reader* DONE!")


def test_align_archive_reader(ark_or_pipe):
    ali_reader = AlignArchiveReader(ark_or_pipe)
    for key, vec in ali_reader:
        print("{0}: {1}".format(key, vec.shape))
    print("TEST *test_align_archive_reader* DONE!")


def test_nnet3egs_reader(egs):
    egs_reader = Nnet3EgsReader(egs)
    for key, _ in egs_reader:
        print("{}".format(key))
    print("TEST *test_nnet3egs_reader* DONE!")


if __name__ == "__main__":
    test_archive_writer("asset/foo.ark", "asset/foo.scp")
    # archive_reader
    test_archive_reader("asset/6.mat.ark", matrix=True)
    test_archive_reader("asset/6.vec.ark", matrix=False)
    test_archive_reader("cat asset/6.mat.ark |")
    # script_reader
    test_script_reader("asset/6.mat.scp", matrix=True)
    test_script_reader("asset/6.vec.scp", matrix=False)
    test_script_reader("shuf asset/6.mat.scp | head -n 2 |")
    # align_archive_reader
    test_align_archive_reader("gunzip -c asset/10.ali.gz |")
    # nnet3egs_reader
    test_nnet3egs_reader("asset/10.egs")
