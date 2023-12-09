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

if __name__ == '__main__':
    # folders = ['test', 'dev', 'train_sp']
    # folders = ['zh_test']
    folders = ['train_sp']
    
    word_count = {}
    for folder in folders:
        data_path = os.path.join('./dump/raw', folder, 'text')
        datasets  = [d[1:] for d in read_file(data_path, sp=' ')]

        for datas in datasets:
            for word in datas:
                word_count[word] = word_count[word] + 1 if word in word_count else 1
        
    rareword_f10 = []
    for word in word_count:
        count = word_count[word]
        if count <= 10:
            rareword_f10.append([word])
    
    rareword_f10 = sorted(rareword_f10)

    output_path = './local/rareword_f10'
    write_file(output_path, rareword_f10)
    
    # rareword_f10_pinyin = [[convert(d[0])] for d in rareword_f10]
    # output_path = './local/rare_words_test.txt'
    # write_file(output_path, rareword_f10_pinyin)
