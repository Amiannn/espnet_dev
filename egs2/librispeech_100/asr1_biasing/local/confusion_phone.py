import os
import string
import numpy as np
import eng_to_ipa as ipa

from tqdm import tqdm

from local.utils import read_file
from local.utils import read_json
from local.utils import write_json
from local.utils import write_numpy

from fuzzywuzzy import fuzz
from fuzzywuzzy import process


def distance(a, b):
    return (fuzz.ratio(a, b) / 100)

def isEnglish(s):
    if len(s) < 1:
        return False
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

tokens_path = "./data/en_token_list/bpe_unigram600suffix/tokens.txt"
sp = "▁"

if __name__ == "__main__":
    tokens = [d[0] for d in read_file(tokens_path)]
    eng_tokens   = []
    other_tokens = []
    for index, token in enumerate(tokens):
        if token == "'":
            print('_' * 30)
        if "<" in token:
            print(f'special token: {token}')
            other_tokens.append([index, token])
        elif token in string.punctuation:
            print(f'punctuation token: {token}')
            other_tokens.append([index, token])
        elif isEnglish(token.replace(sp, '')):
            print(f'english: {token}')
            eng_tokens.append([index, token])
        else:
            print(f'other token: {token}')
            other_tokens.append([index, token])

    # pho_tokens = [ipa.convert(t.replace('▁', '').lower()) for t in tqdm(eng_tokens)]
    # output_path = "./local/eng_tokens_pho.json"
    # write_json(output_path, pho_tokens)
    pho_tokens = read_json('./local/eng_tokens_pho.json')

    confusion_table = np.identity(len(tokens), dtype=np.float32)
    print(confusion_table)
    print(confusion_table.shape)

    for eng_data, pho_token in zip(eng_tokens, pho_tokens):
        index, eng_token = eng_data
        for eng_data_hat, pho_token_hat in zip(eng_tokens, pho_tokens):
            index_hat, eng_token_hat = eng_data_hat
            score = distance(pho_token, pho_token_hat)
            confusion_table[index][index_hat] = score
            # print(f'id: {index}, token: {eng_token}, id_hat: {index_hat}, token_hat: {eng_token_hat}, score: {score:.2f}')

    output_path = "/".join(tokens_path.split('/')[:-1] + ['confusion_table.npy'])
    write_numpy(output_path, confusion_table)