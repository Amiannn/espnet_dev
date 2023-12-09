import os
import jieba
import pickle

from tqdm import tqdm

def read_file(path, sp='\t'):
    datas = []
    with open(path, 'r', encoding='utf-8') as frs:
        for fr in frs:
            data = fr.replace('\n', '')
            data = data.split(sp)
            datas.append(data)
    return datas

def write_file(path, datas, sp=" "):
    with open(path, 'w', encoding='utf-8') as fr:
        for data in datas:
            fr.write(sp.join(data) + '\n')

def read_pickle(dict_path):
    return pickle.load(open(dict_path, "rb"))

pin2zh_path = "/share/nas165/amian/experiments/nlp/SubCharTokenization/data/pinyin_to_chinese.pkl"

PATH        = "/share/nas165/amian/experiments/speech/tcpgen/espnet/egs2/aishell/asr1_biasing/exp/asr_finetune_freeze_conformer_transducer_tcpgen500_deep_sche30_rep_suffix/decode_asr_pinyin_asr_model_valid.loss.best/test/score_wer"

if __name__ == '__main__':
    files  = ['hyp.trn', 'ref.trn']
    pin2zh = read_pickle(pin2zh_path)

    for file in files:
        data_path = os.path.join(PATH, file)
        datas  = [[d[-1], d[0]] for d in read_file(data_path, sp='\t')]
        converted_datas = []
        for uid, words in datas:
            converted_word = []
            for word in words.split(' '):
                converted_char = []
                for char in word.split('쎯'):
                    if char == '':
                        continue
                    try:
                        converted_char.append(pin2zh[char])
                    except:
                        print(char)
                        converted_char.append(char)
                converted_word.append(''.join(converted_char))
            converted_datas.append([' '.join(converted_word), uid])
        
        output_path = os.path.join(PATH, f'converted.{file}')
        print(output_path)
        write_file(output_path, converted_datas, sp='\t')
        