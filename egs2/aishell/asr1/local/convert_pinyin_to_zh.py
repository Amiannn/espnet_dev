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

# PATH        = "/share/nas165/amian/experiments/speech/tcpgen/espnet/egs2/aishell/asr1/exp/asr_train_asr_transducer_conformer_e15_linear1024_raw_zh_bpe5000_use_wandbtrue_sp_suffix/decode_asr_asr_model_valid.loss.ave_10best/test/text"
PATH        = "/share/nas165/amian/experiments/speech/tcpgen/espnet/egs2/aishell/asr1/exp/asr_train_asr_transducer_conformer_e15_linear1024_raw_zh_bpe5000_use_wandbtrue_sp_suffix/decode_asr_asr_model_valid.loss.ave_10best/test/text"
pin2zh_path = "/share/nas165/amian/experiments/nlp/SubCharTokenization/data/pinyin_to_chinese.pkl"

if __name__ == '__main__':
    datas  = [[d[0], [k for k in d[1:] if k != '']] for d in read_file(PATH, sp=' ')]
    pin2zh = read_pickle(pin2zh_path)

    converted_datas = []
    for uid, words in datas:
        converted_word = []
        for word in words:
            converted_char = []
            for char in word.split('쎯'):
                if char == '':
                    continue
                try:
                    # print(f'token: {char}, zh: {pin2zh[char]}')
                    converted_char.append(pin2zh[char])
                except:
                    converted_char.append(char)
            converted_word.append(''.join(converted_char))
            # print('_' * 30)
        converted_datas.append([uid, ''.join(converted_word)])
        # print('-' * 30)
    
    # print(converted_datas)
    
    output_path = f'{PATH}_converted'
    write_file(output_path, converted_datas)
    