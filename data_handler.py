#!/usr/bin/env python
# wujian@2018
"""
    Simple wrapper for iobase.py
    - ArchiveReader
    - ScriptReader
    - ArchiveWriter
    - AlignmentReader
"""

import os
import glob
import warnings
import numpy as np
import iobase as io


def parse_scps(scp_path, addr_processor=lambda x: x):
    assert os.path.exists(scp_path)
    scp_dict = dict()
    with open(scp_path, 'r') as f:
        for scp in f:
            scp_tokens = scp.strip().split()
            if len(scp_tokens) != 2:
                raise RuntimeError(
                    "Error format of context \'{}\'".format(scp))
            key, addr = scp_tokens
            if key in scp_dict:
                raise ValueError("Duplicate key \'{}\' exists!".format(key))
            scp_dict[key] = addr_processor(addr)
    return scp_dict


class Reader(object):
    """
        Base class, to be implemented
    """

    def __init__(self, scp_path, addr_processor=lambda x: x):
        if not os.path.exists(scp_path):
            raise FileNotFoundError("Could not find file {}".format(scp_path))
        self.index_dict = parse_scps(scp_path, addr_processor=addr_processor)
        self.index_keys = [key for key in self.index_dict.keys()]

    def _load(self, key):
        raise NotImplementedError

    # number of utterance
    def __len__(self):
        return len(self.index_dict)

    # avoid key error
    def __contains__(self, key):
        return key in self.index_dict

    # sequential index
    def __iter__(self):
        for key in self.index_keys:
            yield key, self._load(key)

    # random index, support str/int as index
    def __getitem__(self, index):
        if type(index) == int:
            num_utts = len(self.index_keys)
            if index >= num_utts or index < 0:
                raise KeyError("Interger index out of range, {} vs {}".format(
                    index, num_utts))
            key = self.index_keys[index]
            return self._load(key)
        elif type(index) is str:
            if index not in self.index_dict:
                raise KeyError("Missing utterance {}!".format(index))
            return self._load(index)
        else:
            raise IndexError("Unsupported index type: {}".format(type(index)))


class ScriptReader(Reader):
    """
        Reader for kaldi's scripts(for BaseFloat matrix)
    """

    def __init__(self, ark_scp):
        def addr_processor(addr):
            addr_token = addr.split(":")
            if len(addr_token) == 1:
                raise ValueError("Unsupported scripts address format")
            path, offset = ":".join(addr_token[0:-1]), int(addr_token[-1])
            return (path, offset)

        super(ScriptReader, self).__init__(
            ark_scp, addr_processor=addr_processor)

    def _load(self, key):
        path, offset = self.index_dict[key]
        with open(path, 'rb') as f:
            f.seek(offset)
            io.expect_binary(f)
            ark = io.read_general_mat(f)
        return ark

class Writer(object):
    """
        Base class, to be implemented
    """

    def __init__(self, ark_path, scp_path=None):
        self.scp_path = scp_path
        self.ark_path = ark_path

    def __enter__(self):
        self.scp_file = None if self.scp_path is None else open(
            self.scp_path, "w")
        self.ark_file = open(self.ark_path, "wb")
        return self

    def __exit__(self, type, value, trace):
        if self.scp_file:
            self.scp_file.close()
        self.ark_file.close()

    def write(self, key, value):
        raise NotImplementedError

class ArchiveReader(object):
    """
        Sequential Reader for .ark object
    """
    def __init__(self, ark_path):
        if not os.path.exists(ark_path):
            raise FileNotFoundError("Could not find {}".format(ark_path))
        self.ark_path = ark_path
    
    def __iter__(self):
        with open(self.ark_path, "rb") as fd:
            for key, mat in io.read_ark(fd):
                yield key, mat

class AlignmentReader(ScriptReader):
    """
        Reader for kaldi's scripts(for int32 vector, such as alignments)
    """

    def __init__(self, ark_scp):
        super(AlignmentReader, self).__init__(ark_scp)

    def _load(self, key):
        path, offset = self.index_dict[key]
        with open(path, 'rb') as f:
            f.seek(offset)
            io.expect_binary(f)
            ark = io.read_common_int_vec(f)
        return ark

class ArchiveWriter(Writer):
    """
        Writer for kaldi's archive && scripts(for BaseFloat matrix)
    """

    def __init__(self, ark_path, scp_path=None):
        super(ArchiveWriter, self).__init__(ark_path, scp_path)

    def write(self, key, matrix):
        io.write_token(self.ark_file, key)
        offset = self.ark_file.tell()
        # binary symbol
        io.write_binary_symbol(self.ark_file)
        io.write_common_mat(self.ark_file, matrix)
        abs_path = os.path.abspath(self.ark_path)
        if self.scp_file:
            self.scp_file.write("{}\t{}:{:d}\n".format(key, abs_path, offset))


def test_archive_writer(ark, scp):
    with ArchiveWriter(ark, scp) as writer:
        for i in range(10):
            mat = np.random.rand(100, 20)
            writer.write("mat-{:d}".format(i), mat)
    print("TEST *test_archieve_writer* DONE!")


def test_script_reader(egs):
    scp_reader = ScriptReader(egs)
    for key, mat in scp_reader:
        print("{}: {}".format(key, mat.shape))
    print("TEST *test_script_reader* DONE!")


def test_alignment_reader(egs):
    ali_reader = AlignmentReader(egs)
    for key, vec in ali_reader:
        print("{}: {}".format(key, vec.shape))
    print("TEST *test_alignment_reader* DONE!")


if __name__ == "__main__":
    test_archive_writer("egs.ark", "egs.scp")
    test_script_reader("egs.scp")
    # test_alignment_reader("egs.scp")
