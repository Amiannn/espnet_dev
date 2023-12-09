import os
import jieba
import json

import sentencepiece as spm

from jiwer import cer
from jiwer import mer
from tqdm  import tqdm

from local.aligner import CheatDetector
from local.aligner import align_to_index

from local.utils import read_file
from local.utils import read_json
from local.utils import write_file
from local.utils import write_json

rareword_path  = './local/rareword_f10'


def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

if __name__ == '__main__':
    blist = [d[0] for d in read_file(rareword_path)]
    print(blist[:10])

    sp_model_path = "./data/token_list/bpe_unigram3949suffix/bpe.model"
    sp = spm.SentencePieceProcessor(model_file=sp_model_path)

    rarewords = []
    for bword in blist:
        bword = bword.lower()
        bword_cut = jieba.lcut(bword, cut_all=True)
        if len(bword_cut) > 1:
            bword_eng = [word for word in bword_cut if isEnglish(word)]
            if len(bword_eng) > 0:
                bword = bword_eng[0]
        if len(bword) < 2:
            continue
        # tokens = sp.encode(bword, out_type=str)
        # print(f'bword: {bword},  token: {",".join(tokens)}')
        # print('_' * 30)
        rarewords.append(bword)
    rarewords = [[d] for d in sorted(list(set(rarewords)))]
    output_path = './local/rareword_f10_new'
    write_file(output_path, rarewords)