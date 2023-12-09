import os
import jieba
import torch
import numpy as np

from local.utils import read_file
from local.utils import read_json
from local.utils import write_json
from local.utils import write_file

from espnet2.text.Butils_full import BiasProc

def get_gold_biasingword(yseqs):
    bwords = []
    yseqs  = [[idx for idx in yseq if idx != -1] for yseq in yseqs]
    for i, yseq in enumerate(yseqs):
        wordbuffer = []
        for j, wp in enumerate(yseq):
            wordbuffer.append(bproc.charlist[wp])
        wordbuffer_str = "".join(wordbuffer)
        print(wordbuffer_str)
        for i in range(len(bproc.encodedlist_str)):
            bword_str = bproc.encodedlist_str[i]
            # start & middle
            bword_length = len(bword_str)
            start = wordbuffer_str.find(f'▁{bword_str}')
            if (wordbuffer_str[:bword_length] == bword_str):
                wordbuffer_str = wordbuffer_str[bword_length:]
                bwords.append(bproc.encodedlist[i])
            elif start > -1:
                wordbuffer_str = wordbuffer_str[:start] + wordbuffer_str[start+len(bword_str):]
                bwords.append(bproc.encodedlist[i])
    return bwords

blist    = "./local/all_rare_words.txt"
bpemodel = "./data/en_token_list/bpe_unigram600suffix/bpe.model"
charlist = "./data/en_token_list/bpe_unigram600suffix/tokens.txt"

ref_path   = "./data/test_clean/text"
ref_datas  = [[d[0], ' '.join(d[1:])] for d in read_file(ref_path, sp=' ')]

maxlen   = 20
bdrop    = 0.0

charlist = [d[0] for d in read_file(charlist)]
bproc    = BiasProc(blist, maxlen, bdrop, bpemodel, charlist)

yseqs = []
count = 0
for idx, text in ref_datas[:5]:
    print(text)
    bpewords = bproc.tokenizer.text2tokens(text)
    yseq     = [bproc.chardict[word] for word in bpewords if word in bproc.chardict]
    # bwords = get_gold_biasingword([yseq])
    # pred   = ["".join(word).replace('▁', '') for word in bwords]
    # if len(utt_ent) > 0:
    #     print(f'{", ".join(sorted(utt_ent))} | {", ".join(sorted(pred))}')
    count += 1
    # assert sorted(utt_ent) == sorted(pred)
    yseqs.append(yseq)

bwords, worddict, btokens, btokens_len = bproc.select_biasing_words(yseqs)
print(worddict)

# bwords = []
# index  = []
# for i, bword in enumerate(bproc.wordblist):
#     words = list(jieba.cut(bword, cut_all=False))
#     print(f'{bword} -> {", ".join(words)}')
#     bwords.append(words)
#     index.append([len("".join(words)), i])
# index = sorted(index, reverse=True)
# bwords = [bwords[i[-1]] for i in index]

# print(bwords[:10])
# output_path = os.path.join('./local', 'rareword.all.sep.txt')
# write_file(output_path, bwords, sp=' ')
