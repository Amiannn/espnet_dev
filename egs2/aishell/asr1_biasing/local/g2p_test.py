import os
import numpy as np
import matplotlib.pyplot as plt

from local.utils import read_file
from local.utils import read_json
from local.utils import write_file
from local.utils import write_json

# from g2p  import make_g2p
from g2p_en import G2p
from tqdm import tqdm

from matplotlib        import colors
from matplotlib.ticker import PercentFormatter

from scipy.special import rel_entr, kl_div

# transducer = make_g2p('eng', 'eng-arpabet')
g2p = G2p()

def s2p(sentence):
    words = sentence.split(' ')
    phos  = []
    for word in words:
        # phos.extend(transducer(word).output_string)
        phos.extend(g2p(word))
    phos = " ".join(phos)
    phos = phos.replace('0', '') 
    phos = phos.replace('1', '') 
    phos = phos.replace('2', '')
    phos = phos.split(' ') 
    _phos = []
    for pho in phos:
        if pho == '': continue
        _phos.append(pho)
    return _phos

def accumulate(dataset):
    table = {}
    for datas in dataset:
        for data in datas:
            table[data] = table[data] + 1 if data in table else 1
    table_keys   = list(table.keys())
    table_values = list(table.values())
    return table_keys, table_values

def draw(x1, y1, x2, y2, fname):
    legend = ['train dist', 'test dist']
    
    # Creating histogram
    fig, axs = plt.subplots(
        1, 1,
        figsize =(20, 5),
        tight_layout = True
    )
    
    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        axs.spines[s].set_visible(False)
    
    # Remove x, y ticks
    axs.xaxis.set_ticks_position('none')
    axs.yaxis.set_ticks_position('none')
    
    # Add padding between axes and labels
    axs.xaxis.set_tick_params(pad = 5)
    axs.yaxis.set_tick_params(pad = 10)

    axs.set_xticklabels(
        axs.get_xticks(), 
        rotation=90, 
        fontsize=10
    )
    
    # Add x, y gridlines
    # axs.grid(
    #     b = True, color ='grey',
    #     linestyle ='-.', linewidth = 0.5,
    #     alpha = 0.6
    # )

    x_axis = list(range(len(x1)))
    axs.bar(
        x_axis, x1, width=1, 
        edgecolor="white", linewidth=0.05, 
        tick_label=y1,
        alpha=0.5
    )

    axs.bar(
        x_axis,  x2, width=1,
        edgecolor="white", linewidth=0.05, 
        tick_label=y2,
        alpha=0.5
    )
    
    # Adding extra features   
    plt.xlabel("X-axis")
    plt.ylabel("y-axis")
    plt.legend(legend)
    plt.title('Domain Distributions')
    
    # Show plot
    plt.savefig(fname, format="pdf", bbox_inches="tight")
    plt.clf()

def sort(values, keys):
    datas = [[value, key] for value, key in zip(values, keys)]
    datas = sorted(datas, reverse=True)
    values = [d[0] for d in datas]
    keys   = [d[1] for d in datas]
    return values, keys

def normalize(values):
    val = np.array(values) + 0.00000000000001
    val = val / np.sum(val)
    return list(val)

def align_keys(keys, subset_keys, subset_values):
    datas = {k: 0 for k in keys}
    for skey, svalue in zip(subset_keys, subset_values):
        datas[skey] = svalue
    skeys   = list(datas.keys())
    svalues = list(datas.values())
    return skeys, svalues

def mask_key(keys, interval=20):
    _keys = []
    for i in range(len(keys)):
        if i % interval == 0:
            _keys.append(keys[i])
        else:
            _keys.append('')
    return _keys

if __name__ == '__main__':
    sentence = "Watch and learn about democracy in its many forms"
    phos     = s2p(sentence)

    train_set_path = "/share/nas165/amian/experiments/speech/tcpgen/espnet/egs2/librispeech_100/asr1_biasing/data/train_clean_100/text"
    test_set_path  = "/share/nas165/amian/experiments/speech/tcpgen/espnet/egs2/librispeech_100/asr1_biasing/data/test_clean/text"

    train_set = read_file(train_set_path, sp=' ')
    test_set  = read_file(test_set_path, sp=' ')

    train_set = [" ".join(d[1:]).replace("'S", "") for d in train_set]
    test_set  = [" ".join(d[1:]).replace("'S", "") for d in test_set]

    train_set_words = [s.split(' ') for s in train_set]
    test_set_words  = [s.split(' ') for s in test_set]
    
    # train_set_phos = [s2p(s) for s in tqdm(train_set)]
    # test_set_phos  = [s2p(s) for s in tqdm(test_set)]
    # write_json(
    #     './pho.json',
    #     {
    #     'train_set_phos': train_set_phos,
    #     'test_set_phos' : test_set_phos
    #     }
    # )
    # datas = read_json('./pho.json')
    # train_set_phos = datas['train_set_phos']
    # test_set_phos  = datas['test_set_phos']
    # train_table_keys, train_table_values = accumulate(train_set_phos)
    # test_table_keys, test_table_values   = accumulate(test_set_phos)

    train_table_keys, train_table_values = accumulate(train_set_words)
    test_table_keys, test_table_values   = accumulate(test_set_words)

    keys = set(train_table_keys)
    keys.update(test_table_keys)
    keys = list(keys)

    train_table_keys, train_table_values = align_keys(keys, train_table_keys, train_table_values)
    train_table_values, train_table_keys = sort(train_table_values, train_table_keys)
    keys = train_table_keys
    test_table_keys, test_table_values   = align_keys(keys, test_table_keys, test_table_values)

    # train_table_values = normalize(train_table_values)
    # test_table_values = normalize(test_table_values)

    # kl = kl_div(train_table_values, test_table_values)
    step = 5000
    kls  = []
    for i in range(0, len(test_table_keys) - step, step):
        kl = sum(
            kl_div(
                normalize(test_table_values[i:i+step]), 
                normalize(train_table_values[i:i+step])
            )
        )
        kls.append([f'{i}-{i+step}', str(kl)])

    write_file('word_kl.txt', kls, sp='\t')
    # kl = kl_div(test_table_values[:100], train_table_values[:100])
    # print(f'KL: {sum(kl)}')
    
    # keys   = mask_key(keys, interval=1)
    # length = len(keys)
    # step   = 100
    # for i in range(0, length - step, step):
    #     draw(
    #         train_table_values[i: i + step], 
    #         keys[i: i + step],
    #         test_table_values[i: i + step], 
    #         keys[i: i + step], 
    #         f'./dist/pho_dist_{i}_{i+step}.pdf'
    #     )
    # draw(
    #     train_table_values, 
    #     keys,
    #     test_table_values, 
    #     keys, 
    #     f'./pho_dist.pdf'
    # )