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


ch2pinyin = "/share/nas165/amian/experiments/nlp/SubCharTokenization/data/chinese_to_pinyin.pkl"

control_char = u'0123456789abcdefghijklmnopqrstuvwxyz' 
control_uni = [chr(ord(c)+50000) for c in control_char]

CH2EN_PUNC = {
    f: t for f, t in zip(
        u'，。！？【】（）％＃＠＆１２３４５６７８９０；：',
        u',.!?[]()%#@&1234567890;:'
    )
}

## choose accordingly
map_dict = read_pickle(ch2pinyin)
puncs    = '，。！？【】（）％＃＠＆１２３４５６７８９０；：,.!?[]()%#@&1234567890;:'

def convert(line):
    line  = line.strip().lower()
    lines = list(jieba.cut(line))
    out_lines = []
    for line in lines:
        out_line = ""
        for char in line:
            if char in CH2EN_PUNC:
                char = CH2EN_PUNC[char]
            if char in map_dict:
                ## append transliterated char and separation symbol
                out_line += map_dict[char] + chr(ord('_')+50000)
            else:
                if char.isalpha():
                    char = chr(ord(char)+50000)
                out_line += char
        out_lines.append(out_line)
    return ' '.join(out_lines)

if __name__ == '__main__':
    root_path = "./data"
    # folders   = ['dev', 'test', 'train_sp']
    folders   = ['zh_test']

    for folder in folders:
        path  = os.path.join(root_path, folder, 'text__')
        datas = read_file(path, sp=' ')
        print(datas[:10])
        converted_datas = []
        for uid, text in tqdm(datas):
            # text = convert(text)
            text = " ".join(list(jieba.cut(text)))
            converted_datas.append([uid, text])
        output_path = os.path.join(root_path, folder, 'text')
        write_file(output_path, converted_datas)