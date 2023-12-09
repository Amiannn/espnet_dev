import os
import numpy as np

from numpy.linalg import norm

from local.utils import read_file
from local.utils import read_json
from local.utils import write_json

if __name__ == '__main__':
    PATH  = "/share/nas165/amian/experiments/speech/tcpgen/espnet/egs2/librispeech_100/asr1_biasing/exp/asr_train_conformer_transducer_tcpgen500_deep_sche30_GCN6L_rep_suffix/decode_asr_test_asr_model_valid.loss.ave_10best/test_clean/logdir/asr_inference.json"
    datas = read_json(PATH)
    avg_cosine        = 0
    avg_cosine_tcpgen = 0
    avg_cosine_dummy  = 0
    for data in datas:
        pgens        = data['pgens']
        model_probs  = data['model_probs']
        tcpgen_probs = data['tcpgen_probs']

        pgens        = 1 - np.array(pgens)
        model_probs  = np.array(model_probs)
        tcpgen_probs = np.array(tcpgen_probs)
        dummy        = np.ones(len(pgens))

        cosine        = np.dot(pgens, model_probs) / (norm(pgens) * norm(model_probs))
        cosine_tcpgen = np.dot(pgens, tcpgen_probs) / (norm(pgens) * norm(tcpgen_probs))
        cosine_dummy  = np.dot(pgens, dummy) / (norm(pgens) * norm(dummy))

        avg_cosine        += cosine
        avg_cosine_tcpgen += cosine_tcpgen
        avg_cosine_dummy  += cosine_dummy

    avg_cosine        /= len(datas)
    avg_cosine_tcpgen /= len(datas)
    avg_cosine_dummy  /= len(datas)
    print(f'cosine       : {avg_cosine}')
    print(f'cosine_tcpgen: {avg_cosine_tcpgen}')
    print(f'cosine dummy : {avg_cosine_dummy}')