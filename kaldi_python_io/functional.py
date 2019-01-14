# wujian@2019

from ._io_kernel import read_float_mat, read_float_vec

def read_kaldi_mat(file):
    with open(file, "rb") as f:
        mat = read_float_mat(f, direct_access=True)
    return mat

def read_kaldi_vec(file):
    with open(file, "rb") as f:
        mat = read_float_vec(f, direct_access=True)
    return mat