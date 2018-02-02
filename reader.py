#!/usr/bin/env python
# coding=utf-8
# wujian@17.12.30

import os
import glob
import logging

import kaldi_io.iobase as io
from kaldi_io.utils import parse_scps

logfmt  = "%(filename)s[%(lineno)d] %(asctime)s %(levelname)s: %(message)s"
datefmt = "%Y-%m-%d %T"
logging.basicConfig(level=logging.INFO, format=logfmt, datefmt=datefmt)

class ScpReader(object):
    def __init__(self, scp_path, model='feature'):
        assert model in ['feature', 'pdfid']
        assert os.path.exists(scp_path), '{} do not exists, please check!'.format(scp_path)
        self.scp_dict = parse_scps(scp_path) 
        self.io_func = io.read_general_mat if model == 'feature' else io.read_common_int_vec
        self.str = '{}: {}({})'.format(self.__class__.__name__, scp_path, model)
        
    def __len__(self):
        return len(self.scp_dict)

    def __str__(self):
        return self.str

    def _load_ark(self, key):
        ark_path, offset = self.scp_dict[key]
        with open(ark_path, 'rb') as f:
            f.seek(offset)
            ark = self.io_func(f, direct_access=True)
        return ark

    def keys(self):
        return self.scp_dict.keys()

    def __iter__(self):
        for key in self.scp_dict:
            yield key, self._load_ark(key)

    # used for direct access like 'scp_reader[key]'
    def __getitem__(self, key):
        ark = self._load_ark(key)
        return ark 

    # used for judge 'assert key in scp_reader'
    def __contains__(self, key):
        return key in self.scp_dict
        
class ArkReader(object):
    def __init__(self, ark_dirp, model='feature'):
        assert model in ['feature', 'pdfid']
        self.io_func = io.read_general_mat if model == 'feature' else io.read_common_int_vec
        self.ark_list = [ark_dirp] if os.path.isfile(ark_dirp) else glob.glob(ark_dirp)
        self.str = '{}: {}({})'.format(self.__class__.__name__, ark_dirp, model)

    # return nums of arks
    def __len__(self):
        return len(self.ark_list)

    def __str__(self):
        return self.str

    def _load_ark(self, ark_path):
        with open(ark_path, 'rb') as fd:
            while True:
                key = io.read_key(fd)
                if not key:
                    break
                pkg = self.io_func(fd)
                yield key, pkg

    def __iter__(self):
        for ark_path in self.ark_list:
            for key, pkg in self._load_ark(ark_path):
                yield key, pkg


def _test_ark_reader():
    ark_reader = ArkReader('../data/pdf/pdf.*.ark', model='pdfid')
    for key, vec in ark_reader:
        print(key)


class DataReader(object):
    def __init__(self, feats_scp, pdfs_scp, time_delay=0):
        self.feats_reader = ScpReader(feats_scp, model='feature')
        self.pdfs_reader  = ScpReader(pdfs_scp, model='pdfid')
        self.time_delay   = time_delay
        assert time_delay >= 0

    def _process_utt(self, feats_mat, pdfs_vec):
        assert pdfs_vec.ndim == 1
        num_frames, dim = feats_mat.shape
        assert num_frames == pdfs_vec.size
        return (feats_mat, pdfs_vec) if not self.time_delay else \
                (feats_mat[: -self.time_delay], pdfs_vec[self.time_delay: ])

    def __str__(self):
        return ('{}:\n\tfeature_reader = {}\n\tpdf_reader = {}' + \
                '\n\ttime_delay = {}').format(self.__class__.__name__, self.feats_reader, \
                self.pdfs_reader, self.time_delay)

    def __iter__(self): 
        num_miss_trans = processed = 0
        tot_frames = 0
        for key, feat in self.feats_reader:
            # may not have alignments
            if key not in self.pdfs_reader:
                num_miss_trans += 1
                continue
            processed += 1
            tot_frames += feat.shape[0]
            if processed % 10000 == 0:
                logging.info('Processed {} utterances'.format(processed))
            pdf = self.pdfs_reader[key]
            feat, pdf = self._process_utt(feat, pdf)
            yield key, feat, pdf
        if num_miss_trans:
            logging.warn('{} utterances missing targets'.format(num_miss_trans))
        logging.info('Processed {} utterances({} frames) in total'.format(processed, tot_frames))

def _test_scp_reader():
    scp_reader = ScpReader('data/dev_feats.scp', model='feature')
    print(scp_reader)
    for key, ark in scp_reader:
        print(ark.shape)
        assert key in scp_reader

    scp_reader = ScpReader('data/dev_pdfs.scp', model='pdfid')
    for key, ark in scp_reader:
        print(ark.shape)
        assert key in scp_reader

def _test_data_reader():
    data_reader = DataReader('feats.scp', 'pdfs.scp')
    for key, feat, pdf in data_reader:
        print(feat.shape)

if __name__ == '__main__':
    # _test_scp_reader()
    _test_ark_reader()
    
