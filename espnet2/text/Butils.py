from __future__ import division

import argparse
import json
import logging
import math
import os
import random
from copy import deepcopy
from time import time

import editdistance
import numpy as np
import six
import torch

from torch.nn.utils.rnn import pad_sequence

from espnet2.text.build_tokenizer import build_tokenizer
from espnet.lm.lm_utils import make_lexical_tree

random.seed(0)


class BiasProc(object):
    def __init__(self, blist, maxlen, bdrop, bpemodel, charlist):
        with open(blist) as fin:
            self.wordblist = [line.split() for line in fin]
        self.tokenizer = build_tokenizer("bpe", bpemodel)
        self.encodedset = self.encode_blist()
        self.maxlen = maxlen
        self.bdrop = bdrop
        self.chardict = {}
        for i, char in enumerate(charlist):
            self.chardict[char] = i
        self.charlist = charlist

    def encode_blist(self):
        encodedset = set()
        self.encodedlist = []
        for word in self.wordblist:
            bpeword = self.tokenizer.text2tokens(word)
            encodedset.add(tuple(bpeword[0]))
            self.encodedlist.append(tuple(bpeword[0]))
        return encodedset

    def encode_spec_blist(self, blist):
        encoded = []
        for word in blist:
            bpeword = self.tokenizer.text2tokens(word)
            encoded.append(tuple(bpeword))
        return encoded

    def construct_blist(self, bwords):
        if len(bwords) < self.maxlen:
            distractors = random.sample(self.encodedlist, k=self.maxlen - len(bwords))
            sampled_words = []
            for word in distractors:
                if word not in bwords:
                    sampled_words.append(word)
            sampled_words = sampled_words + bwords
        else:
            sampled_words = bwords
        uttKB = sorted(sampled_words)
        worddict = {word: i + 1 for i, word in enumerate(uttKB)}
        lextree = make_lexical_tree(worddict, self.chardict, -1)
        return lextree, worddict

    def select_biasing_words(self, yseqs, suffix=True, cb=False, ret_worddict=False):
        bwords = []
        wordbuffer = []
        yseqs = [[idx for idx in yseq if idx != -1] for yseq in yseqs]
        for i, yseq in enumerate(yseqs):
            for j, wp in enumerate(yseq):
                wordbuffer.append(self.charlist[wp])
                if suffix and self.charlist[wp].endswith("▁"):
                    if tuple(wordbuffer) in self.encodedset:
                        bwords.append(tuple(wordbuffer))
                    wordbuffer = []
        bwords = [word for word in bwords if random.random() > self.bdrop]
        lextree, worddict = self.construct_blist(bwords)
        if cb:
            cb_tokens, cb_tokens_len = self.contextual_biasing_list(worddict)
            if ret_worddict:
                return bwords, lextree, cb_tokens, cb_tokens_len, worddict
            return bwords, lextree, cb_tokens, cb_tokens_len
        return bwords, lextree

    def contextual_biasing_list(self, bwords):
        btoken_list = []
        btoken_len  = []
        
        for bword in bwords:
            btokens = torch.tensor([self.chardict[bsub] for bsub in bword], dtype=torch.long)
            btoken_list.append(btokens)
            btoken_len.append(len(btokens))
        # with ooKB
        btoken_list.append(torch.tensor([len(self.chardict)], dtype=torch.long))        
        btoken_len.append(1)

        cb_tokens     = pad_sequence(btoken_list, batch_first=True)
        cb_tokens_len = torch.tensor(btoken_len).to(torch.int64)

        assert max(cb_tokens_len) == cb_tokens.shape[1]
        return cb_tokens, cb_tokens_len