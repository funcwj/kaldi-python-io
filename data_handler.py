#!/usr/bin/env python
# wujian@2018
"""
    Simple wrapper for iobase.py
    - ArchiveReader
    - ScriptReader
    - ArchiveWriter
    - AlignmentReader
    - Nnet3EgsReader
"""

import os
import glob
import random
import warnings
import numpy as np
import iobase as io


def ext_fopen(fname, mode):
    """
    Extend file open function, support "-", which means std-input/output
    """
    if mode not in ["w", "r", "wb", "rb"]:
        raise ValueError("Unknown open mode: {mode}".format(mode=mode))
    if not fname:
        return None
    if fname == "-":
        if mode in ["w", "wb"]:
            return sys.stdout.buffer if mode == "wb" else sys.stdout
        else:
            return sys.stdin.buffer if mode == "rb" else sys.stdin
    else:
        return open(fname, mode)


def ext_fclose(fname, fd):
    """
    Extend file close function, support "-", which means std-input/output
    """
    if fname != "-" and fd:
        fd.close()


def parse_scps(scp_path, addr_processor=lambda x: x):
    """
    Parse kaldi's script(.scp) file with supported for stdin
    """
    scp_dict = dict()
    f = ext_fopen(scp_path, 'r')
    for scp in f:
        scp_tokens = scp.strip().split()
        if len(scp_tokens) != 2:
            raise RuntimeError("Error format of context \'{}\'".format(scp))
        key, addr = scp_tokens
        if key in scp_dict:
            raise ValueError("Duplicate key \'{}\' exists!".format(key))
        scp_dict[key] = addr_processor(addr)
    ext_fclose(scp_path, f)
    return scp_dict


class Reader(object):
    """
        Base class for sequential/random accessing, to be implemented
    """

    def __init__(self, scp_path, addr_processor=lambda x: x):
        self.index_dict = parse_scps(scp_path, addr_processor=addr_processor)
        self.index_keys = [key for key in self.index_dict.keys()]

    def _load(self, key):
        raise NotImplementedError

    def shuf(self):
        random.shuffle(self.index_keys)

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
        if type(index) not in [int, str]:
            raise IndexError("Unsupported index type: {}".format(type(index)))
        if type(index) == int:
            # from int index to key
            num_utts = len(self.index_keys)
            if index >= num_utts or index < 0:
                raise KeyError(
                    "Interger index out of range, {:d} vs {:d}".format(
                        index, num_utts))
            index = self.index_keys[index]
        if index not in self.index_dict:
            raise KeyError("Missing utterance {}!".format(index))
        return self._load(index)


class SequentialReader(object):
    """
        Base class for sequential reader(only for .ark/.egs)
    """

    def __init__(self, ark_path):
        if not os.path.exists(ark_path):
            raise FileNotFoundError("Could not find {}".format(ark_path))
        self.ark_path = ark_path

    def __iter__(self):
        raise NotImplementedError


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
        # if dump ark to output, then ignore scp
        if ark_path == "-" and scp_path:
            warnings.warn(
                "Ignore .scp output discriptor cause dump archives to stdout")
            self.scp_path = None

    def __enter__(self):
        # "wb" is important
        self.ark_file = ext_fopen(self.ark_path, "wb")
        self.scp_file = ext_fopen(self.scp_path, "w")
        return self

    def __exit__(self, type, value, trace):
        ext_fclose(self.ark_path, self.ark_file)
        ext_fclose(self.scp_path, self.scp_file)

    def write(self, key, value):
        raise NotImplementedError


class ArchiveReader(SequentialReader):
    """
        Sequential Reader for .ark object
    """

    def __init__(self, ark_path):
        super(ArchiveReader, self).__init__(ark_path)

    def __iter__(self):
        with open(self.ark_path, "rb") as fd:
            for key, mat in io.read_ark(fd):
                yield key, mat


class Nnet3EgsReader(SequentialReader):
    """
        Sequential Reader for .egs object
    """

    def __init__(self, ark_path):
        super(Nnet3EgsReader, self).__init__(ark_path)

    def __iter__(self):
        with open(self.ark_path, "rb") as fd:
            for key, egs in io.read_nnet3_egs_ark(fd):
                yield key, egs


class AlignmentReader(ScriptReader):
    """
        Reader for kaldi's scripts(for int32 vector, such as alignments)
    """

    def __init__(self, ark_scp):
        super(AlignmentReader, self).__init__(ark_scp)

    def _load(self, key):
        path, offset = self.index_dict[key]
        with open(path, "rb") as f:
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


def test_nnet3egs_reader(egs):
    egs_reader = Nnet3EgsReader(egs)
    for key, _ in egs_reader:
        print("{}".format(key))
    print("TEST *test_nnet3egs_reader* DONE!")


if __name__ == "__main__":
    test_archive_writer("egs.ark", "egs.scp")
    test_script_reader("egs.scp")
    # test_alignment_reader("egs.scp")
    # test_nnet3egs_reader("10.egs")
