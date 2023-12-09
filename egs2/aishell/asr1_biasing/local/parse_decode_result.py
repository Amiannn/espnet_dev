import os
import json

from local.utils import read_file
from local.utils import write_json

REF_PATH    = "/share/nas165/amian/experiments/speech/tcpgen/espnet/egs2/librispeech_100/asr1_biasing/data/test_clean/text"
DECODE_PATH = "/share/nas165/amian/experiments/speech/tcpgen/espnet/egs2/librispeech_100/asr1_biasing/exp/asr_train_conformer_transducer_tcpgen500_deep_sche30_GCN6L_rep_suffix/decode_asr_test_asr_model_valid.loss.ave_10best/test_clean/logdir/asr_inference.1.log"

def process_raw_data(raw_data):
    chunk_data = []
    _temp = []
    for data in raw_data:
        data = data[0]
        if 'speech length:' in data and len(_temp) > 0:
            chunk_data.append(_temp)
            _temp = []
        else:
            _temp.append(data)

    extract_keys = {
        'idx': ['idx', None, None],
        'blist': ['blist', ', ', str],
        'best hypo': ['hyp', '▁', str],
        'best tokens': ['hyp_tokens', ' ', str],
        'best pgens': ['pgens', ' ', float],
        'model probs': ['model_probs', ' ', float],
        'model tokens': ['model_tokens', ' ', float],
        'tcpgen probs': ['tcpgen_probs', ' ', float],
        'tcpgen tokens': ['tcpgen_tokens', ' ', float],
        'model topk probs': ['topk_model_probs', '_', json.loads],
        'model topk tokens': ['topk_model_tokens', '_', json.loads],
        'tcpgen topk probs': ['topk_tcpgen_probs', '_', json.loads],
        'tcpgen topk tokens': ['topk_tcpgen_tokens', '_', json.loads]
    }
    datas = []
    for chunk in chunk_data:
        data = {}
        for line in chunk:
            if ': ' not in line:
                continue
            key, raw = line.split(': ', 1)
            if key not in extract_keys:
                continue
            _key, sp, fn = extract_keys[key]
            data[_key] = [fn(d) for d in raw.split(sp)] if sp != None else raw
        if len(data) > 0:
            data['model_tokens'] = [int(d) for d in data['model_tokens']]
            data['tcpgen_tokens'] = [int(d) for d in data['tcpgen_tokens']]
            datas.append(data)
    return datas

raw_data  = read_file(DECODE_PATH, sp='\t')
ref_datas = read_file(REF_PATH, sp=' ')
refs = {d[0]: " ".join(d[1:]) for d in ref_datas}

chunk_data = process_raw_data(raw_data)
for data in chunk_data:
    idx = data['idx']
    ref = refs[idx]
    data['hyp'] = " ".join(data['hyp'])
    data['ref'] = ref
    # print(data)
    # print('_' * 30)


output_dir  = "/".join(DECODE_PATH.split('/')[:-1])
output_path = os.path.join(output_dir, 'asr_inference.json')

write_json(output_path, chunk_data)