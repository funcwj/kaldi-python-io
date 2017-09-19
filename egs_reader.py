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
        if c == ' ':
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
    reference static void ReadIndexVectorElementBinary(std::istream &is, 
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

def uncompress_data(data):
    pass

# waiting for implement 
def read_compress_mat(fd):
    """ reference to function Read in CompressMatrix
    waiting for implement uncompress operation
    """
    mat_type = read_token(fd)
    print '\tFollowing matrix type: {}'.format(mat_type)
    head = struct.unpack('ffii', fd.read(16))
    print '\tCompress matrix header: ', head
    if mat_type == 'CM':
        remain_size = head[3] * (8 + head[2])
    elif mat_type == 'CM2':
        remain_size = 2 * head[2] * head[3]
    elif mat_type == 'CM3':
        remain_size = head[2] + head[3]
    else: 
        throw_on_error(false, 'Unknown matrix type: {}'.format(mat_type))
    cpdata = fd.read(remain_size)
    uncompress_data(cpdata)

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
