#!/usr/bin/env python
# coding=utf-8
# wujian@17.9.19

import struct
import numpy as np


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

def expect_token(fd, ref):
    """ check weather the token read equals to the reference
    """
    token = read_token(fd)
    throw_on_error(token == ref, 'Expect token \'{}\', but gets {}'.format(ref, token))

def read_int32(fd):
    """ read a value in type 'int32' in kaldi setup
    """
    int_size = fd.read(1)
    throw_on_error(int_size == '\4')
    int_str = fd.read(4)
    int_val = struct.unpack('i', int_str)
    return int_val[0]

def read_float32(fd):
    """ read a value in type 'BaseFloat' in kaldi setup
    """
    float_size = fd.read(1)
    throw_on_error(float_size == '\4')
    float_str = fd.read(4)
    float_val = struct.unpack('f', float_str)
    return float_val

def read_index_tuple(fd):
    """ read the member in struct Index in nnet3/nnet-common.h  
    """
    n = read_int32(fd)
    t = read_int32(fd)
    x = read_int32(fd)
    return (n, t, x)

def read_index(fd, index, cur_set):
    """ read struct Index
        see:
        static void ReadIndexVectorElementBinary(std::istream &is, 
            int32 i, std::vector<Index> *vec)
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
    """ read several Index and return as a list
    """
    expect_token(fd, '<I1V>')
    size = read_int32(fd)
    print '\tSize of index vector: {}'.format(size)
    index = []
    for i in range(size):
        cur_index = read_index(fd, i, index)
        index.append(cur_index)
    return index


def read_common_mat(fd):
    pass


def read_sparse_vec(fd):
    """ reference to function Read in SparseVector
    """
    expect_token(fd, 'SV')
    dim = read_int32(fd)
    num_elems = read_int32(fd)
    print '\tRead sparse vector(dim = {}, row = {})'.format(dim, num_elems)
    sparse_vec = []
    for i in range(num_elems):
        index = read_int32(fd)
        value = read_float32(fd)
        sparse_vec.append((index, value))
    return sparse_vec 

def read_sparse_mat(fd):
    """ reference to function Read in SparseMatrix
    """
    mat_type = read_token(fd)
    print '\tFollowing matrix type: {}'.format(mat_type)
    num_rows = read_int32(fd)
    sparse_mat = []
    for i in range(num_rows):
        sparse_mat.append(read_sparse_vec(fd))
    print sparse_mat
    return sparse_mat 

def uint16_to_floats(min_value, prange, pchead):
    """
    see matrix/compressed-matrix.cc
    inline float CompressedMatrix::Uint16ToFloat(
        const GlobalHeader &global_header, uint16 value)
    """
    p = []
    for value in pchead:
        p.append(float(min_value + prange * 1.52590218966964e-05 * value))
    return p

def uint8_to_float(char, pchead):
    """
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
    min_value, prange, num_rows, num_cols = head
    mat = np.zeros([num_rows, num_cols])
    print '\tUncompress to matrix {} X {}'.format(num_rows, num_cols)
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
    waiting for implement uncompress operation
    """
    cps_type = read_token(fd)
    print '\tFollowing matrix type: {}'.format(cps_type)
    head = struct.unpack('ffii', fd.read(16))
    print '\tCompress matrix header: ', head
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
    print uncompress(compress_data, cps_type, head)


def read_general_mat(fd):
    """ reference to function Read in class GeneralMatrix
    """
    peek_mat_type = peek_char(fd)
    if peek_mat_type == 'C':
        read_compress_mat(fd)
    elif peek_mat_type == 'S':
        read_sparse_mat(fd)
    else:
        read_common_mat(fd)

def read_nnet_io(fd):
    """ reference to function Read in class NnetIo
    """
    expect_token(fd, '<NnetIo>')
    print '\tName: {}'.format(read_token(fd))
    print read_index_vec(fd)
    read_general_mat(fd)
    expect_token(fd, '</NnetIo>')

def read_nnet3eg(fd):
    """ reference to function Read in class NnetExample
    """
    expect_binary(fd)
    expect_token(fd, '<Nnet3Eg>')
    expect_token(fd, '<NumIo>')
    num_io = read_int32(fd)
    for i in range(num_io):
        read_nnet_io(fd)
    expect_token(fd, '</Nnet3Eg>')


def test():
    with open('10.egs', 'rb') as egs:
        while True:
            key = read_token(egs)
            if not key:
                break
            print 'Egs key: {}'.format(key)
            read_nnet3eg(egs)

if __name__ == '__main__':
    test()
