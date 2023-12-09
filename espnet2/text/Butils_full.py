import os
import math
import json
import random
import logging
import argparse
import time as time_

import six
import torch
import faiss
import numpy as np

from time import time
from copy import deepcopy

from rapidfuzz import fuzz
from rapidfuzz import process, fuzz

from torch.nn.utils.rnn import pad_sequence

from espnet.lm.lm_utils           import make_lexical_tree
from espnet.lm.lm_utils           import make_lexical_tree_idx
from espnet2.text.build_tokenizer import build_tokenizer

import matplotlib.pyplot as plt

random.seed(0)

FAISS_GPU_ID = 0
FAISS_RES    = faiss.StandardGpuResources()

def read_file(path, sp='\t'):
    datas = []
    with open(path, 'r', encoding='utf-8') as frs:
        for fr in frs:
            data = fr.replace('\n', '')
            data = data.split(sp)
            datas.append(data)
    return datas

class BiasProc(object):
    def __init__(self, blist, maxlen, bdrop, bpemodel, charlist, sdrop=0.0):
        with open(blist) as fin:
            self.wordblist = [line.replace('\n', '') for line in fin]
        self.bword2idx = {bword: i for i, bword in enumerate(self.wordblist)}
        self.bindexis  = set(range(len(self.wordblist)))
        self.tokenizer = build_tokenizer("bpe", bpemodel)
        self.encodedset = self.encode_blist()
        self.maxlen = maxlen
        self.bdrop = bdrop
        self.sdrop = sdrop
        self.chardict = {}
        for i, char in enumerate(charlist):
            self.chardict[char] = i
        self.charlist    = charlist
        self.fuzzy_cache = {}
        self.btokens, self.btokens_len = self.pad_blist(self.encodedlist)
        self.encodedlist_str = ["".join(tokens) for tokens in self.encodedlist]

        # dynamic
        self.index = None

    def encode_blist(self):
        encodedset = set()
        self.encodedlist = []
        for word in self.wordblist:
            bpeword = self.tokenizer.text2tokens(word)
            encodedset.add(tuple(bpeword))
            self.encodedlist.append(tuple(bpeword))
        return encodedset

    def encode_spec_blist(self, blist):
        encoded = []
        for word in blist:
            bpeword = self.tokenizer.text2tokens(word)
            encoded.append(tuple(bpeword))
        return encoded

    @torch.no_grad()
    def build_index(self, embed_matrix, bencoder, Kproj):
        build_start_time = time_.time()
        cb_tokens_embed  = embed_matrix[self.btokens[:-1, :]]
        cb_seq_embed, _  = bencoder(cb_tokens_embed)
        self.bembeds     = torch.mean(cb_seq_embed, dim=1)
        self.bembed_keys = Kproj(self.bembeds).cpu()
        self.index       = faiss.IndexFlatIP(self.bembed_keys.shape[-1])
        self.index       = faiss.index_cpu_to_gpu(FAISS_RES, FAISS_GPU_ID, self.index)
        self.index.add(self.bembed_keys)
        build_time_elapsed  = time_.time() - build_start_time
        print(f'Build index bembeds: {self.bembeds.shape}')
        print(f'Build biasing index done: {self.index}')
        print(f'Build index elapsed time: {build_time_elapsed:.4f}(s)')

    def random_sampling_methods(self, bwords, k):
        distractors = random.sample(self.encodedset, k=k)
        return distractors

    def fuzzy_sampling_methods(self, bwords, k):
        topk = 5
        batch_distractors = []
        bwords = ["".join(bword).replace("▁", "") for bword in bwords]
        if random.random() > self.sdrop:
            for bword in bwords:
                if bword not in self.fuzzy_cache:
                    scores      = torch.tensor([fuzz.ratio(bword, word[0]) for word in self.wordblist])
                    # indexis     = torch.argsort(scores, descending=True)[1:101]
                    indexis     = torch.argsort(scores, descending=True)[1:topk+1]
                    # r_index     = torch.randint(indexis.shape[0], (topk,))
                    # distractors = [self.encodedlist[i] for i in indexis[r_index]]
                    distractors = [self.encodedlist[i] for i in indexis]
                    self.fuzzy_cache[bword] = distractors
                else:
                    distractors = self.fuzzy_cache[bword]
                batch_distractors.extend(distractors)
            logging.info(f'special sampling working ~')
        else:
            logging.info(f'special sampling droped ~')
        # add random samples
        if len(batch_distractors) < self.maxlen:
            k = self.maxlen - len(batch_distractors)
            distractors = self.random_sampling_methods(bwords, k)
            batch_distractors = [word for word in distractors if word not in batch_distractors] + batch_distractors
        return batch_distractors
    
    # TODO: create mask for attention
    @torch.no_grad()
    def fl_sampling_methods(self, bwords, queries, k):
        topk = 5
        fl_distractors    = []
        batch_distractors = []
        fl_indexis        = []
        bwords2id         = [self.bword2idx["".join(bword).replace("▁", "")] for bword in bwords]
        bwords2id         = list(set(bwords2id))
        batch, qlen, _    = queries.shape
        mask              = None
        if random.random() > self.sdrop:
            flatten_queries    = queries.reshape(batch * qlen, -1)
            D, I               = self.index.search(flatten_queries, topk)
            I                  = [i if i not in bwords2id else -1 for i in I.reshape(-1)]
            fl_indexis, inverse_indices = np.unique(I, return_inverse=True)
            # fl_indexis, inverse_indices = torch.unique(torch.tensor(I), sorted=False, return_inverse=True)
            # inverse_indices             = inverse_indices.numpy()
            fl_indexis = fl_indexis.tolist()
            mask    = np.ones((batch, qlen, len(fl_indexis)), dtype=int)
            b_idx   = np.arange(batch)
            b_idx   = np.repeat(b_idx, qlen * topk)
            q_idx   = np.arange(qlen)
            q_idx   = np.repeat(np.repeat(q_idx, topk).reshape(1, -1), batch, axis=0).reshape(-1)
            mask[b_idx, q_idx, inverse_indices] = 0
            if fl_indexis[0] == -1:
                mask       = mask[:, :, 1:]
                fl_indexis = fl_indexis[1:]            
            fl_distractors = [self.encodedlist[i] for i in fl_indexis]
        # add positive and random samples
        candiates = self.bindexis - set(bwords2id) - set(fl_indexis)
        rand_k    = self.maxlen - len(bwords2id) - topk if isinstance(mask, np.ndarray) else self.maxlen - len(bwords2id)
        rand_indexis      = bwords2id + random.sample(candiates, rand_k)
        rand_distractors  = [self.encodedlist[i] for i in rand_indexis]
        mask_rand = np.zeros((batch, qlen, len(rand_distractors)), dtype=int)
        mask      = np.concatenate([mask, mask_rand], axis=-1) if isinstance(mask, np.ndarray) else mask_rand
        batch_distractors = fl_distractors + rand_distractors 
        return batch_distractors, mask
        # return batch_distractors, None

    # TODO: remove this function
    @torch.no_grad()
    def fl_sampling_methods_debug(self, bwords, queries, k):
        logging.info(f'Debuging FL sampling methods...')
        topk = 5
        fl_distractors    = []
        batch_distractors = []
        fl_idx            = []
        bwords2id         = [self.bword2idx["".join(bword).replace("▁", "")] for bword in bwords]
        bwords2id         = list(set(bwords2id))
        batch, qlen, _    = queries.shape
        mask              = None
        if random.random() > self.sdrop:
            flatten_queries     = queries.reshape(batch * qlen, -1)
            D, I                = self.index.search(flatten_queries, topk)
            fl_idx, inverse_idx = np.unique(I, return_inverse=True)
            mask = np.ones((batch, qlen, fl_idx.shape[0]), dtype=int)
            # b_idx   = np.arange(batch)
            # b_idx   = np.repeat(b_idx, qlen * topk)
            # q_idx   = np.arange(qlen)
            # q_idx   = np.repeat(np.repeat(q_idx, topk).reshape(1, -1), batch, axis=0).reshape(-1)
            # mask[b_idx, q_idx, inverse_idx] = 0        
            fl_distractors = [self.encodedlist[i] for i in fl_idx]
        else:
            logging.info('skip fl sampling...')
        
        # add positive and random samples
        candiates = self.bindexis - set(bwords2id)
        rand_k    = self.maxlen - len(bwords2id) - topk if isinstance(mask, np.ndarray) else self.maxlen - len(bwords2id)
        rand_indexis      = bwords2id + random.sample(candiates, rand_k)
        rand_distractors  = [self.encodedlist[i] for i in rand_indexis]
        mask_rand = np.zeros((batch, qlen, len(rand_distractors)), dtype=int)
        mask      = np.concatenate([mask, mask_rand], axis=-1) if isinstance(mask, np.ndarray) else mask_rand
        batch_distractors = fl_distractors + rand_distractors 
        logging.info(f'batch_distractors shape: {len(batch_distractors)}')
        logging.info(f'mask shape       : {mask.shape}')
        # return rand_distractors, mask_rand
        return batch_distractors, mask

    def get_bword_idx(self, bwords):
        idx = set([self.bword2idx["".join(bword).replace("▁", "")] for bword in bwords])
        return torch.tensor(list(idx))

    # TODO: change main fn by this function
    @torch.no_grad()
    def query_sampling_methods(self, bwords, queries, K):
        logging.info(f'Query sampling methods...')
        B, S, D  = queries.shape
        Gold_idx = self.get_bword_idx(bwords)
        skip_sampling = random.random() <= self.sdrop
        if not skip_sampling:
            Q_flatten = queries.reshape(B * S, -1)
            _, I      = self.index.search(Q_flatten, K)
            I_hat     = torch.from_numpy(I).reshape(-1)
            Q_idx, INV_idx = torch.unique(I_hat, return_inverse=True)
            Q_mask         = torch.ones((B, S, Q_idx.shape[0]), dtype=int)
            B_idx = torch.arange(B).repeat(S * K, 1).T.reshape(-1)
            S_idx = torch.arange(S).repeat(K, B).T.reshape(-1)
            # masking
            Q_mask[B_idx, S_idx, INV_idx] = 0
            # masking out duplicate index
            DUP_idx = torch.isin(Q_idx, Gold_idx).repeat(B, S, 1)
            Q_mask  = Q_mask.masked_fill(DUP_idx, 1)
            Q_list  = Q_idx.tolist()
        # Gold biasing word
        Gold_list = Gold_idx.tolist()
        Gold_mask = torch.zeros(B, S, len(Gold_list))
        # Random biasing word
        candiates = self.bindexis - set(Gold_list)
        K_rand    = self.maxlen - len(Gold_list)
        if not skip_sampling:
            candiates = candiates - set(Q_list)
            K_rand    = K_rand - K
        Rand_list = random.sample(candiates, K_rand)
        Rand_mask = torch.zeros(B, S, K_rand)
        # combine all together
        B_list = Gold_list + Rand_list
        mask   = torch.cat([Gold_mask, Rand_mask], dim=-1)
        if not skip_sampling:
            B_list = B_list + Q_list
            mask   = torch.cat([mask, Q_mask], dim=-1)
        distractors = [self.encodedlist[i] for i in B_list]
        logging.info(f'skip sampling: {skip_sampling}')
        logging.info(f'distractors  : {len(distractors)}')
        logging.info(f'mask         : {mask.shape}')
        return distractors, mask

    def pad_blist(self, worddict):
        btoken_list = []
        btoken_len  = []
        
        for bword in worddict:
            try:
                btokens = torch.tensor([self.chardict[bsub] for bsub in bword], dtype=torch.long)
            except:
                # logging.info(f'error bword: {bword}')
                btokens = torch.tensor([self.chardict[bsub] for bsub in bword if bsub in self.chardict], dtype=torch.long)
            btoken_list.append(btokens)
            btoken_len.append(len(btokens))
        # with ooKB
        btoken_list.append(torch.tensor([len(self.chardict)], dtype=torch.long))        
        btoken_len.append(1)

        cb_tokens     = pad_sequence(btoken_list, batch_first=True)
        cb_tokens_len = torch.tensor(btoken_len).to(torch.int64)

        assert max(cb_tokens_len) == cb_tokens.shape[1]
        return cb_tokens, cb_tokens_len

    def construct_blist_fl(self, bwords, queries, sample_fn, K=None):
        uttKB, mask = sample_fn(bwords, queries, K)
        btokens, btokens_len = self.pad_blist(uttKB)
        worddict = {word: i + 1 for i, word in enumerate(uttKB)}
        # worddict    = [word for i, word in enumerate(uttKB)]

        # mask with oov
        B, S, D = mask.shape
        mask    = torch.cat([mask, torch.zeros(B, S, 1)], dim=-1)
        return worddict, btokens, btokens_len, mask
        # return worddict, btokens, btokens_len, None

    def construct_blist(self, bwords, sample_fn):
        if len(bwords) < self.maxlen:
            k = self.maxlen - len(bwords)
            distractors   = sample_fn(bwords, k)
            sampled_words = [word for word in distractors if word not in bwords] + bwords
        else:
            sampled_words = bwords
        uttKB    = sorted(list(set(sampled_words)))
        worddict = {word: i + 1 for i, word in enumerate(uttKB)}
        btokens, btokens_len = self.pad_blist(worddict)
        return worddict, btokens, btokens_len

    def select_biasing_words(
        self, 
        yseqs,
        queries=None,
        suffix=True,
        topk=5,
        sampling_type="random",
        return_trie=False
    ):
        # bwords = []
        # wordbuffer = []
        # yseqs = [[idx for idx in yseq if idx != -1] for yseq in yseqs]
        # for i, yseq in enumerate(yseqs):
        #     for j, wp in enumerate(yseq):
        #         wordbuffer.append(self.charlist[wp])
        #         if suffix and self.charlist[wp].endswith("▁"):
        #             if tuple(wordbuffer) in self.encodedset:
        #                 bwords.append(tuple(wordbuffer))
        #             wordbuffer = []
        bwords = []
        yseqs  = [[idx for idx in yseq if idx != -1] for yseq in yseqs]
        for i, yseq in enumerate(yseqs):
            wordbuffer = []
            for j, wp in enumerate(yseq):
                wordbuffer.append(self.charlist[wp])
            wordbuffer_str = "".join(wordbuffer)
            for i in range(len(self.encodedlist_str)):
                bword_str = self.encodedlist_str[i]
                # start & middle
                bword_length = len(bword_str)
                start = wordbuffer_str.find(f'▁{bword_str}')
                if (wordbuffer_str[:bword_length] == bword_str):
                    wordbuffer_str = wordbuffer_str[bword_length:]
                    bwords.append(self.encodedlist[i])
                elif start > -1:
                    wordbuffer_str = wordbuffer_str[:start] + wordbuffer_str[start+len(bword_str):]
                    bwords.append(self.encodedlist[i])
        bwords = [word for word in bwords if random.random() > self.bdrop]

        if sampling_type == "random":
            sample_fn = self.random_sampling_methods
            worddict, btokens, btokens_len = self.construct_blist(bwords, sample_fn)
        elif sampling_type == "fuzzy":
            sample_fn = self.fuzzy_sampling_methods
            worddict, btokens, btokens_len = self.construct_blist(bwords, sample_fn)
        elif sampling_type == "framelevel":
            sample_fn = self.fl_sampling_methods
            worddict, btokens, btokens_len, mask = self.construct_blist_fl(
                bwords,
                queries,
                sample_fn
            )
            return bwords, worddict, btokens, btokens_len, mask
        elif sampling_type == "framelevel_debug":
            sample_fn = self.fl_sampling_methods_debug
            worddict, btokens, btokens_len, mask = self.construct_blist_fl(
                bwords,
                queries,
                sample_fn
            )
            return bwords, worddict, btokens, btokens_len, mask
        elif sampling_type == "framelevel_qsampling":
            sample_fn = self.query_sampling_methods
            worddict, btokens, btokens_len, mask = self.construct_blist_fl(
                bwords,
                queries,
                sample_fn,
                K=topk
            )
            return bwords, worddict, btokens, btokens_len, mask

        if return_trie:
            lextree = make_lexical_tree_idx(worddict, self.chardict, -1)
            return bwords, worddict, btokens, btokens_len, lextree
        else:
            return bwords, worddict, btokens, btokens_len

if __name__ == "__main__":
    blist    = "/share/nas165/amian/experiments/speech/tcpgen/espnet/egs2/librispeech_100/asr1_biasing/local/rareword_f15.txt"
    maxlen   = 20
    bdrop    = 0.0
    bpemodel = "/share/nas165/amian/experiments/speech/tcpgen/espnet/egs2/librispeech_100/asr1_biasing/data/en_token_list/bpe_unigram600suffix/bpe.model"
    charlist = "/share/nas165/amian/experiments/speech/tcpgen/espnet/egs2/librispeech_100/asr1_biasing/data/en_token_list/bpe_unigram600suffix/tokens.txt"

    charlist = [d[0] for d in read_file(charlist)]
    bproc = BiasProc(blist, maxlen, bdrop, bpemodel, charlist)

    tokens = bproc.tokenizer.text2tokens('<blank>')
    print(tokens)
    idxs   = [bproc.chardict[token] for token in tokens]
    print(idxs)

    # texts = [
    #     "CHAPTER ONE MISSUS RACHEL LYNDE IS SURPRISED MISSUS RACHEL LYNDE LIVED JUST WHERE THE AVONLEA MAIN ROAD DIPPED DOWN INTO A LITTLE HOLLOW FRINGED WITH ALDERS AND LADIES EARDROPS AND TRAVERSED BY A BROOK",
    #     "THAT HAD ITS SOURCE AWAY BACK IN THE WOODS OF THE OLD CUTHBERT PLACE IT WAS REPUTED TO BE AN INTRICATE HEADLONG BROOK IN ITS EARLIER COURSE THROUGH THOSE WOODS WITH DARK SECRETS OF POOL AND CASCADE BUT BY THE TIME IT REACHED LYNDE'S HOLLOW IT WAS A QUIET WELL CONDUCTED LITTLE STREAM",
    #     "FOR NOT EVEN A BROOK COULD RUN PAST MISSUS RACHEL LYNDE'S DOOR WITHOUT DUE REGARD FOR DECENCY AND DECORUM IT PROBABLY WAS CONSCIOUS THAT MISSUS RACHEL WAS SITTING AT HER WINDOW KEEPING A SHARP EYE ON EVERYTHING THAT PASSED FROM BROOKS AND CHILDREN UP",
    #     "AND THAT IF SHE NOTICED ANYTHING ODD OR OUT OF PLACE SHE WOULD NEVER REST UNTIL SHE HAD FERRETED OUT THE WHYS AND WHEREFORES THEREOF THERE ARE PLENTY OF PEOPLE IN AVONLEA AND OUT OF IT WHO CAN ATTEND CLOSELY TO THEIR NEIGHBOR'S BUSINESS BY DINT OF NEGLECTING THEIR OWN",
    # ]

    # yseqs = []
    # for text in texts:
    #     bpewords = bproc.tokenizer.text2tokens(text)
    #     bpeidx   = [bproc.chardict[word] for word in bpewords]
    #     yseqs.append(bpeidx)

    # bwords, worddict, btokens, btokens_len = bproc.select_biasing_words(
    #     yseqs, 
    #     sampling_type="random"
    # )
    # print(btokens.shape)
    # # print('_' * 30)
    # print(f'worddict: {worddict}')

    # worddict = {
    #     ('AL', 'D', 'ERS▁'): 1, 
    #     ('B', 'RO', 'O', 'K', 'S▁'): 2, 
    #     ('C', 'AS', 'CA', 'DE▁'): 3, 
    #     ('D', 'IN', 'T▁'): 4, 
    #     ('DE', 'C', 'EN', 'C', 'Y▁'): 5, 
    #     ('DE', 'COR', 'U', 'M▁'): 6, 
    #     ('DI', 'P', 'P', 'ED▁'): 7, 
    #     ('E', 'AR', 'D', 'RO', 'P', 'S▁'): 8, 
    #     ('FER', 'RE', 'TED▁'): 9, 
    #     ('FR', 'IN', 'G', 'ED▁'): 10, 
    #     ('HE', 'AD', 'LONG▁'): 11, 
    #     ('IN', 'TRI', 'C', 'ATE▁'): 12, 
    #     ('L', 'Y', 'N', 'DE', "'", 'S▁'): 13, 
    #     ('NE', 'G', 'LE', 'C', 'TING▁'): 14, 
    #     ('NE', 'I', 'G', 'H', 'B', 'OR', "'", 'S▁'): 15, 
    #     ('RE', 'PU', 'TED▁'): 16, 
    #     ('SE', 'C', 'RE', 'TS▁'): 17, 
    #     ('TRA', 'VER', 'S', 'ED▁'): 18, 
    #     ('W', 'H', 'Y', 'S▁'): 19, 
    #     ('W', 'HE', 'RE', 'FOR', 'ES▁'): 20,
    #     ('DE', 'C', 'EN', 'S', 'Y▁'): 21, 
    # }

    # biasingwords = ["".join(word).replace("▁", "") for word in worddict]
    # for i, bword in enumerate(biasingwords):
    #     print(f'{i}, {bword}')
    # print()

    # lextree = make_lexical_tree_idx(worddict, bproc.chardict, -1)

    # now = lextree
    # print(f'<ROOT>')
    # print(now[-1])
    # print('-' * 30)
    # for c in ('DE', 'C', 'EN', 'C', 'Y▁'):
    #     id  = bproc.chardict[c]
    #     now = now[0][id]
    #     print(f'<{c}, {id}>')
    #     print(now[-1])
    #     # print(f'{now[id]}')
    #     print('-' * 30)