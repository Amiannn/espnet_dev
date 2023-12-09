import os

from local.utils import read_file
from local.utils import read_pickle
from local.utils import write_json

PATH = "/share/nas165/amian/experiments/speech/tcpgen/espnet/egs2/librispeech_100/asr1/exp/pyf98/librispeech_100_transducer_conformer/decode_transducer_asr_model_valid.acc.ave/test_clean/logdir/output.1/embed.pickle"
TOKEN_PATH = "/share/nas165/amian/experiments/speech/tcpgen/espnet/egs2/librispeech_100/asr1/data/en_token_list/bpe_unigram600suffix/tokens.txt"

datas  = read_pickle(PATH)[1]
tokens = [d[0] for d in read_file(TOKEN_PATH, sp='\t')]

utterance = datas[0]
print(utterance['token_int'])

top_ids = utterance['topk_ids'][:, 0]
print(top_ids)