import os
import jieba
import numpy as np
import sentencepiece as spm

from jiwer import wer
from tqdm  import tqdm

from local.utils import read_file
from local.utils import read_json
from local.utils import write_json
from local.utils import write_file

TEXT_PATH = "./data/train/text"
words     = []

if __name__ == '__main__':
    data = read_file(TEXT_PATH, sp=" ")

    words = {}
    for index, text in data:
        seg_list = list(jieba.cut(text))
        for seg in seg_list:
            words[seg] = words[seg] + 1 if seg in words else 1
    words = [[words[k], k] for k in words]
    words = sorted(words, reverse=True)
    print(words)
    print(len(words))