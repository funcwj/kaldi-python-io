#!/usr/bin/env python
# coding=utf-8

import logging
import numpy as np

logfmt = "%(filename)s[%(lineno)d] %(asctime)s %(levelname)s: %(message)s"
datefmt = "%Y-%m-%d %T"
logging.basicConfig(level=logging.INFO, format=logfmt, datefmt=datefmt)


def parse_scps(scp_path):
    with open(scp_path, 'r') as f:
        scps = f.readlines()
    scp_dict = dict()
    num_utts = 0
    for scp in scps:
        scp_tokens = scp.strip().split()
        assert len(scp_tokens) == 2, "error format of \'{}\'".format(scp)
        key, addr = scp_tokens
        ark_tokens = addr.split(':')
        assert len(ark_tokens) == 2, "error format of \'{}\'".format(scp)
        ark, offset = ark_tokens
        assert key not in scp_dict, "duplicate key[{}] exists".format(key)
        scp_dict[key] = (ark, int(offset))
        num_utts += 1
    logging.info('Read {} utterance from {}'.format(num_utts, scp_path))
    return scp_dict


def parse_utt2spk(utt2spk_path):
    with open(utt2spk_path, 'r') as f:
        utt2spk = f.readlines()
    utt2spk_dict = dict()
    for line in utt2spk:
        tokens = line.strip().split()
        assert len(tokens) == 2, 'error format of \'{}\' in utt2spk'.format(
            line)
        utt_id, spk_id = tokens
        utt2spk_dict[utt_id] = spk_id
    return utt2spk_dict


def apply_mvn(feats_mat, norm_means=True, norm_vars=True):
    mean = np.mean(feats_mat, axis=0)
    std = np.std(feats_mat, axis=0)
    if norm_means:
        feats_mat = feats_mat - mean
    if norm_vars:
        feats_mat = feats_mat / std
    return feats_mat
