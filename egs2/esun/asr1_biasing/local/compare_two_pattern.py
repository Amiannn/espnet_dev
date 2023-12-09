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

asr_error_path  = './error_pattren_asr.tsv'
bias_error_path = './error_pattren_tcpgen.tsv'

if __name__ == '__main__':
    asr_error  = {d[0]: d[1] for d in read_file(asr_error_path, sp='\t')}
    bias_error = {d[0]: d[1] for d in read_file(bias_error_path, sp='\t')}
    
    keys  = list(set(list(asr_error.keys()) + list(bias_error.keys())))
    error = []
    for key in keys:
        asr  = asr_error[key] if key in asr_error else ""
        bias = bias_error[key] if key in bias_error else ""
        error.append([key, asr, bias])

    output_path = './local/error_pattern.tsv'
    write_file(output_path, sorted(error), sp='\t')