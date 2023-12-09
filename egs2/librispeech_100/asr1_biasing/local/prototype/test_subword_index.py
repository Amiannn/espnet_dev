import os
import jieba
import torch
import random
import numpy as np
import sentencepiece as spm

from jiwer import wer
from tqdm  import tqdm

from local.utils import read_file
from local.utils import read_json
from local.utils import read_pickle
from local.utils import write_json
from local.utils import write_file

BLIST_PATH    = "./local/rareword_f15.txt"
TOKEN_PATH    = "./data/en_token_list/bpe_unigram600suffix/tokens.txt"
SP_MODEL_PATH = "./data/en_token_list/bpe_unigram600suffix/bpe.model"

CTC_SAMPLED_PATH      = "./local/test/debug/1089-134686-0000_frameout.tsv"
CTC_SAMPLED_MEAN_PATH = "./local/test/debug/1089-134686-0000_frameout_mean.pickle"

sp = spm.SentencePieceProcessor(model_file=SP_MODEL_PATH)

if __name__ == '__main__':
    blist  = [d[0] for d in read_file(BLIST_PATH)]
    tokens = [d[0] for d in read_file(TOKEN_PATH)]

    token2bword = {token: [] for token in tokens}
    for bword in blist:
        subwords = sp.encode(bword, out_type=str)
        for subword in subwords:
            if subword not in token2bword[subword]:
                token2bword[subword].append(bword)

    # for token in token2bword:
    #     token2bword[token] = len(token2bword[token])

    # datas = sorted([[token2bword[token], token] for token in token2bword], reverse=True)
    # for length, token in datas:
    #     if length < 1:
    #         continue
    #     print(f'token: {token}, length: {length}')

    ctc_samples = read_file(CTC_SAMPLED_PATH, sp='\t')
    ctc_samples_prob = torch.exp(read_pickle(CTC_SAMPLED_MEAN_PATH))
    # print(ctc_samples_prob)
    # print(ctc_samples_prob.shape)

    idx = torch.multinomial(ctc_samples_prob[1:], 100)
    # print(idx)
    ctc_sample  = [tokens[i] for i in idx]
    ctc_samples = [ctc_sample]
    # print(ctc_samples)


    # dict_token = {}
    for t, frame_ctc_samples in enumerate(ctc_samples):
        non_hit = 0
        idxs    = []
        print(f'frame {t}:')
        for sample_subword in frame_ctc_samples:
            # dict_token[sample_subword] = dict_token[sample_subword] + 1 if sample_subword in dict_token else 1
            length = len(token2bword[sample_subword])
            if length == 0:
                non_hit +=1
                idxs.append(-1)
            else:
                idx = random.randint(0, length - 1)
                idxs.append(token2bword[sample_subword][idx])
            print(f'CTC: {sample_subword}, Sampled: {idxs[-1]}')
        print(f'_' * 30)
        print()
    
    # print(dict_token)
    # print(len(dict_token))