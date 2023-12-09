import os
import jieba
import pickle

from jiwer import cer
from tqdm  import tqdm

def read_file(path, sp='\t'):
    datas = []
    with open(path, 'r', encoding='utf-8') as frs:
        for fr in frs:
            data = fr.replace('\n', '')
            data = data.split(sp)
            datas.append(data)
    return datas

REF_PATH = "./dump/raw/test/text__"
# HYP_PATH = "./exp/asr_train_asr_transducer_conformer_e15_linear1024_raw_zh_bpe5000_use_wandbtrue_sp_suffix/decode_asr_asr_model_valid.loss.best/test/text_converted"
# HYP_PATH = "/share/nas165/amian/experiments/speech/tcpgen/espnet/egs2/aishell/asr1_biasing/exp/asr_finetune_freeze_conformer_transducer_tcpgen500_deep_sche30_rep_suffix/decode_asr_pinyin_asr_model_valid.loss.best/test/text_converted"
# HYP_PATH = "/share/nas165/amian/experiments/speech/tcpgen/espnet/egs2/aishell/asr1/exp/asr_train_asr_transducer_conformer_e15_linear1024_raw_zh_bpe5000_use_wandbtrue_sp_suffix/decode_asr_asr_model_valid.loss.ave_10best/test/text_converted"
HYP_PATH = "/share/nas165/amian/experiments/speech/tcpgen/espnet/egs2/aishell/asr1_biasing/exp/asr_train_asr_transducer_conformer_e15_linear1024_raw_zh_bpe4500_use_wandbtrue_sp_suffix/decode_asr_asr_model_valid.loss.ave_10best/zh_test/text"
# HYP_PATH = "/share/nas165/amian/experiments/speech/tcpgen/espnet/egs2/aishell/asr1_biasing/exp/asr_finetune_freeze_conformer_transducer_tcpgen500_deep_sche30_rep_zh_suffix/decode_asr_asr_model_valid.loss.best/zh_test/text"

if __name__ == '__main__':
    ref_datas = [d[1] for d in read_file(REF_PATH, sp=' ')]
    hyp_datas = [''.join(d[1:]) for d in read_file(HYP_PATH, sp=' ')]
    print(hyp_datas[:10])
    error = cer(ref_datas, hyp_datas)
    print(f'error: {error}')