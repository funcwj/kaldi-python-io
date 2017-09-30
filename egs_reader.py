#!/usr/bin/env python
# coding=utf-8
# wujian@17.9.19


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import struct
import numpy as np

debug = False


def print_info(info):
    if debug:
        print(info)

def throw_on_error(ok, info=''):
    if not ok:
        raise SystemExit(info)

def peek_char(fd):
    """ read a char and seek the point back
    """
    peek_c = fd.read(1)
    fd.seek(-1, 1)
    return peek_c 

def expect_space(fd):
    """ generally, there is a space following the string token
    we need to consume it
    """
    space = fd.read(1)
    throw_on_error(space == ' ', 'Expect space, but gets {}'.format(space))


def expect_binary(fd):
    """ read the binary flags in kaldi, the scripts only
    support reading egs in binary format
    """
    flags = fd.read(2)
    throw_on_error(flags == '\0B', 'Expect binary flags \'B\', but gets {}'.format(flags))

def write_binary_symbol(fd):
    fd.write('\0B')

def read_token(fd):
    """ read {token + ' '} from the file
    this function also consume the space
    """
    key = ''
    while True:
        c = fd.read(1)
        if c == ' ' or c == '':
            break
        key += c
    return None if key == '' else key.strip()

def write_token(fd, token):
    fd.write(token)
    fd.write(' ')

def expect_token(fd, ref):
    """ check weather the token read equals to the reference
    """
    token = read_token(fd)
    throw_on_error(token == ref, 'Expect token \'{}\', but gets {}'.format(ref, token))

def read_key(fd):
    """ read the binary flags following the key
        key might be None
    """
    key = read_token(fd)
    if key: 
        expect_binary(fd)
    return key

def write_key(fd, key):
    write_token(fd, key)
    write_binary_symbol(fd)
    

def read_int32(fd):
    """ read a value in type 'int32' in kaldi setup
    """
    int_size = fd.read(1)
    throw_on_error(int_size == '\04')
    int_str = fd.read(4)
    int_val = struct.unpack('i', int_str)
    return int_val[0]

def write_int32(fd, int32):
    fd.write('\04')
    int_pack = struct.pack('i', int32)
    fd.write(int_pack)

def read_float32(fd):
    """ read a value in type 'BaseFloat' in kaldi setup
    """
    float_size = fd.read(1)
    throw_on_error(float_size == '\04')
    float_str = fd.read(4)
    float_val = struct.unpack('f', float_str)
    return float_val

def read_index_tuple(fd):
    """ read the member in struct Index in nnet3/nnet-common.h  
        return a tuple (n, t, x)
    """
    n = read_int32(fd)
    t = read_int32(fd)
    x = read_int32(fd)
    return (n, t, x)

def read_index(fd, index, cur_set):
    """ wapper to handle struct Index reading task
        see: nnet3/nnet-common.cc
        static void ReadIndexVectorElementBinary(std::istream &is, 
            int32 i, std::vector<Index> *vec)
        return a tuple(n, t, x)
    """
    c = struct.unpack('b', fd.read(1))[0]
    if index == 0:
        if abs(c) < 125:
            return (0, c, 0)
        else:
            if c != 127:
               throw_on_error(false, 'Unexpected character {} encountered while reading Index vector.'.format(c))
            return read_index_tuple(fd)
    else:
        prev_index = cur_set[index - 1]
        if abs(c) < 125:
            return (prev_index[0], prev_index[1] + c, prev_index[2])
        else:
            if c != 127:
               throw_on_error(false, 'Unexpected character {} encountered while reading Index vector.'.format(c))
            return read_index_tuple(fd)


def read_index_vec(fd):
    """ read several Index and return as a list of index:
        [(n_1, t_1, x_1), ..., (n_m, t_m, x_m)]
    """
    expect_token(fd, '<I1V>')
    size = read_int32(fd)
    print_info('\tSize of index vector: {}'.format(size))
    index = []
    for i in range(size):
        cur_index = read_index(fd, i, index)
        index.append(cur_index)
    return index


def read_common_mat(fd):
    """ read common matrix(for class Matrix in kaldi setup), see
        void Matrix<Real>::Read(std::istream & is, bool binary, bool add)
        in matrix/kaldi-matrix.cc
        return a numpy ndarray object
    """
    mat_type = read_token(fd)
    print_info('\tType of the common matrix:{}'.format(mat_type))
    float_size = 4 if mat_type == 'FM' else 8
    float_type = np.float32 if mat_type == 'FM' else np.float64
    num_rows = read_int32(fd)
    num_cols = read_int32(fd)
    print_info('\tSize of the common matrix: {} x {}'.format(num_rows, num_cols))
    mat_data = fd.read(float_size * num_cols * num_rows)
    mat = np.fromstring(mat_data, dtype=float_type)
    return mat.reshape(num_rows, num_cols)

def write_common_mat(fd, mat):
    assert mat.dtype == np.float32 or mat.dtype == np.float64
    mat_type = 'FM' if mat.dtype == np.float32 else 'DM'
    write_token(fd, mat_type)
    num_rows, num_cols = mat.shape
    write_int32(fd, num_rows)
    write_int32(fd, num_cols)
    mat.tofile(fd, sep='')


def read_common_int_vec(fd):
    vec_size = read_int32(fd)
    vec = np.zeros(vec_size, dtype=int)
    for i in range(vec_size):
        value = read_int32(fd)
        vec[i] = value
    return vec

def read_sparse_vec(fd):
    """ reference to function Read in SparseVector
        return a list of key-value pair:
        [(I1, V1), ..., (In, Vn)]
    """
    expect_token(fd, 'SV')
    dim = read_int32(fd)
    num_elems = read_int32(fd)
    print_info('\tRead sparse vector(dim = {}, row = {})'.format(dim, num_elems))
    sparse_vec = []
    for i in range(num_elems):
        index = read_int32(fd)
        value = read_float32(fd)
        sparse_vec.append((index, value))
    return sparse_vec 

def read_sparse_mat(fd):
    """ reference to function Read in SparseMatrix
        A sparse matrix contains couples of sparse vector
    """
    mat_type = read_token(fd)
    print_info('\tFollowing matrix type: {}'.format(mat_type))
    num_rows = read_int32(fd)
    sparse_mat = []
    for i in range(num_rows):
        sparse_mat.append(read_sparse_vec(fd))
    return sparse_mat 

def uint16_to_floats(min_value, prange, pchead):
    """ uncompress type unsigned int16
    see matrix/compressed-matrix.cc
    inline float CompressedMatrix::Uint16ToFloat(
        const GlobalHeader &global_header, uint16 value)
    """
    p = []
    for value in pchead:
        p.append(float(min_value + prange * 1.52590218966964e-05 * value))
    return p

def uint8_to_float(char, pchead):
    """ uncompress unsigned int8
    see matrix/compressed-matrix.cc
    inline float CompressedMatrix::CharToFloat(
        float p0, float p25, float p75, float p100, uint8 value)
    """
    if char <= 64:
        return float(pchead[0] + (pchead[1] - pchead[0]) * char * (1 / 64.0))
    elif char <= 192:
        return float(pchead[1] + (pchead[2] - pchead[1]) * (char - 64) * (1 / 128.0))
    else:
        return float(pchead[2] + (pchead[3] - pchead[2]) * (char - 192) * (1 / 63.0))

def uncompress(compress_data, cps_type, head):
    """ In format CM(kOneByteWithColHeaders):
        PerColHeader, ...(x C), ... uint8 sequence ...
            first: get each PerColHeader pch for a single column
            then : using pch to uncompress each float in the column
        We load it seperately at a time 
        In format CM2(kTwoByte):
        ...uint16 sequence...
        In format CM3(kOneByte):
        ...uint8 sequence...
    """
    min_value, prange, num_rows, num_cols = head
    mat = np.zeros([num_rows, num_cols])
    print_info('\tUncompress to matrix {} X {}'.format(num_rows, num_cols))
    if cps_type == 'CM':
        # checking compressed data size, 8 is the sizeof PerColHeader
        assert len(compress_data) == num_cols * (8 + num_rows)
        # type uint16
        phead_seq = struct.unpack('{}H'.format(4 * num_cols), compress_data[: 8 * num_cols])
        # type uint8
        uint8_seq = struct.unpack('{}B'.format(num_rows * num_cols), compress_data[8 * num_cols: ])
        for i in range(num_cols):
            pchead = uint16_to_floats(min_value, prange, phead_seq[i * 4: i * 4 + 4])
            for j in range(num_rows):
                mat[j, i] = uint8_to_float(uint8_seq[i * num_rows + j], pchead)
    elif cps_type == 'CM2':
        inc = float(prange * (1.0 / 65535.0))
        uint16_seq = struct.unpack('{}H'.format(num_rows * num_cols), compress_data)
        for i in range(num_rows):
            for j in range(num_cols):
                mat[i, j] = float(min_value + uint16_seq[i * num_rows + j] * inc)
    else:
        inc = float(prange * (1.0 / 255.0))
        uint8_seq = struct.unpack('{}B'.format(num_rows * num_cols), compress_data)
        for i in range(num_rows):
            for j in range(num_cols):
                mat[i, j] = float(min_value + uint8_seq[i * num_rows + j] * inc)

    return mat

def read_compress_mat(fd):
    """ reference to function Read in CompressMatrix
        return a numpy ndarray object
    """
    cps_type = read_token(fd)
    print_info('\tFollowing matrix type: {}'.format(cps_type))
    head = struct.unpack('ffii', fd.read(16))
    print_info('\tCompress matrix header: {}'.format(head))
    # 8: sizeof PerColHeader
    # head: {min_value, range, num_rows, num_cols}
    num_rows, num_cols = head[2], head[3]
    if cps_type == 'CM':
        remain_size = num_cols * (8 + num_rows)
    elif cps_type == 'CM2':
        remain_size = 2 * num_rows * num_cols
    elif cps_type == 'CM3':
        remain_size = num_rows * num_cols
    else: 
        throw_on_error(false, 'Unknown matrix compressing type: {}'.format(cps_type))
    # now uncompress it
    compress_data = fd.read(remain_size)
    mat = uncompress(compress_data, cps_type, head)
    return mat


def read_general_mat(fd):
    """ reference to function Read in class GeneralMatrix
        return compress_mat/sparse_mat/common_mat
    """
    peek_mat_type = peek_char(fd)
    if peek_mat_type == 'C':
        return read_compress_mat(fd)
    elif peek_mat_type == 'S':
        return read_sparse_mat(fd)
    else:
        return read_common_mat(fd)

def read_nnet_io(fd):
    """ reference to function Read in class NnetIo
        each NnetIo contains three member: string, Index, GeneralMatrix
        we store them in the dict:{'name': ..., 'index': ..., 'matrix': ...}
    """
    expect_token(fd, '<NnetIo>')
    nnet_io = {}

    name = read_token(fd)
    nnet_io['name'] = name
    print_info('\tName of NnetIo: {}'.format(name))

    index = read_index_vec(fd)
    nnet_io['index'] = index
    print_info(index)

    mat = read_general_mat(fd)
    nnet_io['matrix'] = mat
    print_info(mat)
    expect_token(fd, '</NnetIo>')
    return nnet_io

def read_nnet3eg(fd):
    """ reference to function Read in class NnetExample
        return a list of dict, each dict represent a NnetIo object
        a NnetExample contains several NnetIo
    """
    expect_token(fd, '<Nnet3Eg>')
    expect_token(fd, '<NumIo>')
    # num of the NnetIo
    num_io = read_int32(fd)
    egs = []
    for i in range(num_io):
        egs.append(read_nnet_io(fd))
    expect_token(fd, '</Nnet3Eg>')
    return egs

def read_ark(fd):
    """ usage:
    for key, mat in read_ark(ark):
        print(key)
        ...
    """
    while True:
        key = read_key(fd)
        if not key:
            break
        mat = read_common_mat(fd)
        yield key, mat

def read_egs(fd):
    """ usage:
    for key, eg in read_egs(ark):
        print(key)
        ...
    """
    while True:
        key = read_key(fd)
        if not key:
            break
        eg = read_nnet3eg(fd)
        yield key, eg

def read_ali(fd):
    while True:
        key = read_key(fd)
        if not key:
            break
        ali = read_common_int_vec(fd)
        yield key, ali

# -----------------test part-------------------
def _test_ali():
    with open('10.ali', 'rb') as fd:
        for key, vec in read_ali(fd):
            print(key)

def _test_egs():
    with open('10.egs', 'rb') as egs:
        while True:
            key = read_key(egs)
            if not key:
                break
            print('Egs key: {}'.format(key))
            print(read_nnet3eg(egs))

def _test_write_ark():
    with open('10.ark', 'rb') as ark, open('10.ark.new', 'wb') as dst:
        for key, mat in read_ark(ark):
            write_key(dst, key)
            write_common_mat(dst, mat)

def _test_read_ark():
    with open('10.ark.new', 'rb') as ark:
        for key, mat in read_ark(ark):
            print(key)

if __name__ == '__main__':
    _test_write_ark()
    _test_read_ark()
    # _test_egs()
    # _test_ali()
