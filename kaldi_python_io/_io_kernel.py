#!/usr/bin/env python
# coding=utf-8
# wujian@17.9.19
"""
    Kaldi IO function implement(for binary format), test pass in Python 3.6.0
"""

import struct
import numpy as np

debug = False


def print_info(info):
    if debug:
        print(info)


def throw_on_error(ok, info=''):
    if not ok:
        raise RuntimeError(info)


def peek_char(fd, num_chars=1):
    """ 
        Read a char and seek the point back
    """
    peek_c = fd.peek(num_chars)[:num_chars]
    return bytes.decode(peek_c)


def expect_space(fd):
    """ 
        Generally, there is a space following the string token, we need to consume it
    """
    space = bytes.decode(fd.read(1))
    throw_on_error(space == ' ', f'Expect space, but gets {space}')


def expect_binary(fd):
    """ 
        Read the binary flags in kaldi, the scripts only support reading egs in binary format
    """
    flags = bytes.decode(fd.read(2))
    throw_on_error(flags == '\0B', f'Expect binary flag, but gets {flags}')


def read_token(fd):
    """ 
        Read {token + ' '} from the file(this function also consume the space)
    """
    key = ''
    while True:
        c = bytes.decode(fd.read(1))
        if c == ' ' or c == '':
            break
        key += c
    return None if key == '' else key.strip()


def write_token(fd, token):
    """
        Write a string token, following a space symbol
    """
    fd.write(str.encode(token + " "))


def expect_token(fd, ref):
    """ 
        Check weather the token read equals to the reference
    """
    token = read_token(fd)
    throw_on_error(token == ref, f'Expect token \'{ref}\', but gets {token}')


def read_key(fd):
    """ 
        Read the binary flags following the key(key might be None)
    """
    key = read_token(fd)
    if key:
        expect_binary(fd)
    return key


def write_binary_symbol(fd):
    """ 
        Write a binary symbol
    """
    fd.write(str.encode('\0B'))


def read_int32(fd):
    """ 
        Read a value in type 'int32' in kaldi setup
    """
    int_size = bytes.decode(fd.read(1))
    throw_on_error(int_size == '\04', f'Expect \'\\04\', but gets {int_size}')
    int_str = fd.read(4)
    int_val = struct.unpack('i', int_str)
    return int_val[0]


def write_int32(fd, int32):
    """
        Write a int32 number
    """
    fd.write(str.encode('\04'))
    int_pack = struct.pack('i', int32)
    fd.write(int_pack)


def read_float32(fd):
    """ 
        Read a value in type 'BaseFloat' in kaldi setup
    """
    float_size = bytes.decode(fd.read(1))
    throw_on_error(float_size == '\04',
                   f'Expect \'\\04\', but gets {float_size}')
    float_str = fd.read(4)
    float_val = struct.unpack('f', float_str)
    return float_val


def read_common_mat(fd):
    """ 
        Read common matrix(for class Matrix in kaldi setup)
        see matrix/kaldi-matrix.cc::
            void Matrix<Real>::Read(std::istream & is, bool binary, bool add)
        Return a numpy ndarray object
    """
    mat_type = read_token(fd)
    print_info(f'\tType of the common matrix: {mat_type}')
    if mat_type not in ["FM", "DM"]:
        raise RuntimeError(f"Unknown matrix type in kaldi: {mat_type}")
    float_size = 4 if mat_type == 'FM' else 8
    float_type = np.float32 if mat_type == 'FM' else np.float64
    num_rows = read_int32(fd)
    num_cols = read_int32(fd)
    print_info(f'\tSize of the common matrix: {num_rows} x {num_cols}')
    mat_data = fd.read(float_size * num_cols * num_rows)
    mat = np.fromstring(mat_data, dtype=float_type)
    return mat.reshape(num_rows, num_cols)


def write_common_mat(fd, mat):
    """
        Write a common matrix
    """
    if mat.dtype not in [np.float32, np.float64]:
        raise RuntimeError(f"Unsupported numpy dtype: {mat.dtype}")
    mat_type = 'FM' if mat.dtype == np.float32 else 'DM'
    write_token(fd, mat_type)
    num_rows, num_cols = mat.shape
    write_int32(fd, num_rows)
    write_int32(fd, num_cols)
    fd.write(mat.tobytes())


def read_float_vec(fd, direct_access=False):
    """
        Read float vector(for class Vector in kaldi setup)
        see matrix/kaldi-vector.cc
    """
    if direct_access:
        expect_binary(fd)
    vec_type = read_token(fd)
    print_info(f'\tType of the common vector: {vec_type}')
    if vec_type not in ["FV", "DV"]:
        raise RuntimeError(f"Unknown matrix type in kaldi: {vec_type}")
    float_size = 4 if vec_type == 'FV' else 8
    float_type = np.float32 if vec_type == 'FV' else np.float64
    dim = read_int32(fd)
    print_info(f'\tDim of the common vector: {dim}')
    vec_data = fd.read(float_size * dim)
    return np.fromstring(vec_data, dtype=float_type)


def write_float_vec(fd, vec):
    """
        Write a float vector
    """
    if vec.dtype not in [np.float32, np.float64]:
        raise RuntimeError(f"Unsupported numpy dtype: {vec.dtype}")
    vec_type = 'FV' if vec.dtype == np.float32 else 'DV'
    write_token(fd, vec_type)
    if vec.ndim != 1:
        raise RuntimeError("write_float_vec accept 1D-vector only")
    dim = vec.size
    write_int32(fd, dim)
    fd.write(vec.tobytes())


def read_int32_vec(fd, direct_access=False):
    """
        Read int32 vector (alignments)
    """
    if direct_access:
        expect_binary(fd)
    vec_size = read_int32(fd)
    vec = np.array([read_int32(fd) for _ in range(vec_size)], dtype=np.int32)
    return vec


def read_sparse_vec(fd):
    """ 
        Reference to function Read in SparseVector
        Return a list of key-value pair:
            [(I1, V1), ..., (In, Vn)]
    """
    expect_token(fd, 'SV')
    dim = read_int32(fd)
    num_elems = read_int32(fd)
    print_info(f'\tRead sparse vector(dim = {dim}, row = {num_elems})')
    sparse_vec = []
    for _ in range(num_elems):
        index = read_int32(fd)
        value = read_float32(fd)
        sparse_vec.append((index, value))
    return sparse_vec


def read_sparse_mat(fd):
    """ 
        Reference to function Read in SparseMatrix
        A sparse matrix contains couples of sparse vector
    """
    mat_type = read_token(fd)
    print_info(f'\tFollowing matrix type: {mat_type}')
    num_rows = read_int32(fd)
    sparse_mat = []
    for _ in range(num_rows):
        sparse_mat.append(read_sparse_vec(fd))
    return sparse_mat


# TODO: optimize speed here, original IO 200x slower than uncompressed matrix
#       speed up 5x, now only 50x slower than uncompressed one
def uncompress(cdata, cps_type, head):
    """ 
        In format CM(kOneByteWithColHeaders):
        PerColHeader, ...(x C), ... uint8 sequence ...
            first: get each PerColHeader pch for a single column
            then : using pch to uncompress each float in the column
        We load it seperately at a time 
        In format CM2(kTwoByte):
        ...uint16 sequence...
        In format CM3(kOneByte):
        ...uint8 sequence...
    """
    min_val, prange, num_rows, num_cols = head
    # mat = np.zeros([num_rows, num_cols])
    print_info(f'\tUncompress to matrix {num_rows} X {num_cols}')
    if cps_type == 'CM':
        # checking compressed data size, 8 is the sizeof PerColHeader
        assert len(cdata) == num_cols * (8 + num_rows)
        chead, cmain = cdata[:8 * num_cols], cdata[8 * num_cols:]
        # type uint16
        pch = np.fromstring(chead, dtype=np.uint16).astype(np.float32)
        pch = np.transpose(pch.reshape(num_cols, 4))
        pch = pch * prange / 65535.0 + min_val
        # type uint8
        uint8 = np.fromstring(cmain, dtype=np.uint8).astype(np.float32)
        uint8 = np.transpose(uint8.reshape(num_cols, num_rows))
        # precompute index
        le64_index = uint8 <= 64
        gt92_index = uint8 >= 193
        # le92_index = np.logical_not(np.logical_xor(le64_index, gt92_index))
        return np.where(
            le64_index,
            uint8 * (pch[1] - pch[0]) / 64.0 + pch[0],
            np.where(gt92_index,
                     (uint8 - 192) * (pch[3] - pch[2]) / 63.0 + pch[2],
                     (uint8 - 64) * (pch[2] - pch[1]) / 128.0 + pch[1]))
    else:
        if cps_type == 'CM2':
            inc = float(prange / 65535.0)
            uint_seq = np.fromstring(cdata, dtype=np.uint16).astype(np.float32)
        else:
            inc = float(prange / 255.0)
            uint_seq = np.fromstring(cdata, dtype=np.uint8).astype(np.float32)
        mat = min_val + uint_seq.reshape(num_rows, num_cols) * inc

    return mat


def read_index_tuple(fd):
    """ 
        Read the member in struct Index in nnet3/nnet-common.h  
        Return a tuple (n, t, x)
    """
    n = read_int32(fd)
    t = read_int32(fd)
    x = read_int32(fd)
    return (n, t, x)


def read_index(fd, index, cur_set):
    """ 
        Wapper to handle struct Index reading task(see: nnet3/nnet-common.cc)
            static void ReadIndexVectorElementBinary(std::istream &is, \
                int32 i, std::vector<Index> *vec)
        Return a tuple(n, t, x)
    """
    c = struct.unpack('b', fd.read(1))[0]
    if index == 0:
        if abs(c) < 125:
            return (0, c, 0)
        else:
            if c != 127:
                throw_on_error(
                    False,
                    f'Unexpected character {c} encountered while reading Index vector.'
                )
            return read_index_tuple(fd)
    else:
        prev_index = cur_set[index - 1]
        if abs(c) < 125:
            return (prev_index[0], prev_index[1] + c, prev_index[2])
        else:
            if c != 127:
                throw_on_error(
                    False,
                    f'Unexpected character {c} encountered while reading Index vector.'
                )
            return read_index_tuple(fd)


def read_index_vec(fd):
    """ 
        Read several Index and return as a list of index:
        [(n_1, t_1, x_1), ..., (n_m, t_m, x_m)]
    """
    expect_token(fd, '<I1V>')
    size = read_int32(fd)
    print_info(f'\tSize of index vector: {size}')
    index = []
    for i in range(size):
        cur_index = read_index(fd, i, index)
        index.append(cur_index)
    return index


def read_compress_mat(fd):
    """ 
        Reference to function Read in CompressMatrix
        Return a numpy ndarray object
    """
    cps_type = read_token(fd)
    print_info(f'\tFollowing matrix type: {cps_type}')
    head = struct.unpack('ffii', fd.read(16))
    print_info(f'\tCompress matrix header: {head}')
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
        throw_on_error(False, f'Unknown matrix compressing type: {cps_type}')
    # now uncompress it
    compress_data = fd.read(remain_size)
    mat = uncompress(compress_data, cps_type, head)
    return mat


def read_float_mat(fd, direct_access=False):
    """ 
        Reference to function Read in class GeneralMatrix
        Return compress_mat/sparse_mat/common_mat
    """
    if direct_access:
        expect_binary(fd)
    peek_mat_type = peek_char(fd)
    if peek_mat_type == 'C':
        return read_compress_mat(fd)
    elif peek_mat_type == 'S':
        return read_sparse_mat(fd)
    else:
        return read_common_mat(fd)

def read_float_mat_vec(fd, direct_access=False):
    """
    Read float matrix or vector
    """
    if direct_access:
        expect_binary(fd)
    peek_type = peek_char(fd, num_chars=2)
    # FV DV FM DM
    if peek_type[-1] == "V":
        return read_float_vec(fd, direct_access=False)
    else:
        return read_float_mat(fd, direct_access=False)

def write_float_mat_vec(fd, mat_or_vec):
    """
    Write float matrix or vector
    """
    if isinstance(mat_or_vec, np.ndarray):
        if mat_or_vec.ndim == 2:
            write_common_mat(fd, mat_or_vec)
        else:
            write_float_vec(fd, mat_or_vec)
    else:
        raise TypeError(f"Unsupport type: {type(mat_or_vec)}")

def read_nnet_io(fd):
    """ 
        Reference to function Read in class NnetIo
        each NnetIo contains three member: string, Index, GeneralMatrix
        I store them in the dict:{'name': ..., 'index': ..., 'matrix': ...}
    """
    expect_token(fd, '<NnetIo>')
    nnet_io = {}

    name = read_token(fd)
    nnet_io['name'] = name
    print_info(f'\tName of NnetIo: {name}')

    index = read_index_vec(fd)
    nnet_io['index'] = index
    print_info(index)

    mat = read_float_mat(fd)
    nnet_io['matrix'] = mat
    print_info(mat)
    expect_token(fd, '</NnetIo>')
    return nnet_io


def read_nnet3_egs(fd):
    """ 
        Reference to function Read in class NnetExample
        Return a list of dict, each dict represent a NnetIo object
        a NnetExample contains several NnetIo
    """
    expect_token(fd, '<Nnet3Eg>')
    expect_token(fd, '<NumIo>')
    # num of the NnetIo
    num_io = read_int32(fd)
    egs = []
    for _ in range(num_io):
        egs.append(read_nnet_io(fd))
    expect_token(fd, '</Nnet3Eg>')
    return egs


def read_nnet3_egs_ark(fd):
    """ 
        Usage:
        for key, eg in read_nnet3_egs(ark):
            print(key)
            ...
    """
    while True:
        key = read_key(fd)
        if not key:
            break
        egs = read_nnet3_egs(fd)
        yield key, egs


def read_float_ark(fd):
    """ 
    Read float matrix/vector
        Usage:
        for key, mat in read_ark(ark):
            print(key)
            ...
    """
    while True:
        key = read_key(fd)
        if not key:
            break
        obj = read_float_mat_vec(fd)
        yield key, obj


def read_int32_ali(fd):
    """
    Read int23 vector (alignments)
    """
    while True:
        key = read_key(fd)
        if not key:
            break
        ali = read_int32_vec(fd)
        yield key, ali