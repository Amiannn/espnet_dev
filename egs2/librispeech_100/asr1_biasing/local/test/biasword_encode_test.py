import os
import math  
import torch
import numpy as np
import sentencepiece as spm
import plotly.express as px
import matplotlib.pyplot as plt

from tqdm             import tqdm
from sklearn.manifold import TSNE

from local.utils import read_file
from local.utils import read_pickle
from local.utils import write_json
from local.utils import write_file
from local.utils import write_pickle

MODEL_PATH   = "./exp/asr_finetune_freeze_conformer_transducer_contextual_biasing_sche30_rep_suffix/valid.loss.ave_10best.pth"
TOKENS_PATH  = "./data/en_token_list/bpe_unigram600suffix/tokens.txt"
BIASING_PATH = "./local/rareword_f15.txt"
SPM_PATH     = "./data/en_token_list/bpe_unigram600suffix/bpe.model"

debug_path   = "./local/test/debug"

def extract_model(key_prefix, ckpt):
    _ckpt = {}
    for key in ckpt:
        if key_prefix in key:
            _key = key.replace(f'{key_prefix}.', '')
            _ckpt[_key] = ckpt[key]
    return _ckpt

if __name__ == '__main__':
    vocab_size = 600
    dunits     = 256
    attndim    = 256

    embed   = torch.nn.Embedding(vocab_size, dunits)
    ooKBemb = torch.nn.Embedding(1, dunits)
    CbRNN   = torch.nn.LSTM(
        dunits, 
        attndim // 2, 
        1, 
        batch_first=True, 
        bidirectional=True
    )
    Kproj   = torch.nn.Linear(dunits, attndim)

    key_prefixs = {
        "decoder.embed": embed,
        "CbRNN"        : CbRNN,
        "ooKBemb"      : ooKBemb,
        "Kproj"        : Kproj,
    }
    model_ckpt = torch.load(MODEL_PATH)
    
    sub_module_ckpt = {prefix: extract_model(prefix, model_ckpt) for prefix in key_prefixs}
    for prefix in sub_module_ckpt:
        key_prefixs[prefix].load_state_dict(sub_module_ckpt[prefix])
    
    print(key_prefixs)
    
    # max_blist_length = 30291
    max_blist_length = 10000

    tokens    = [d[0] for d in read_file(TOKENS_PATH, sp=' ')]
    blist     = [d[0] for d in read_file(BIASING_PATH, sp=' ')][:max_blist_length-1]
    spm_model = spm.SentencePieceProcessor(model_file=SPM_PATH)

    cb_words = [torch.tensor([vocab_size], dtype=torch.long)]
    for bword in blist:
        cb_word = torch.tensor(spm_model.encode(bword), dtype=torch.long)
        cb_words.append(cb_word)

    embed_matrix  = torch.cat(
        [embed.weight.data, ooKBemb.weight], dim=0
    )

    output_embed_path = os.path.join(debug_path, f'{max_blist_length}_embeds.pickle')
    if os.path.isfile(output_embed_path):
        cb_embeds = read_pickle(output_embed_path)
    else:
        cb_embeds = []
        for cb_word in tqdm(cb_words):
            cb_embed = embed_matrix[cb_word.unsqueeze(0)]
            cb_seq_embed, _ = CbRNN(cb_embed)
            cb_embed = torch.mean(cb_seq_embed, dim=1)
            cb_embed = Kproj(cb_embed)
            cb_embeds.append(cb_embed)
        cb_embeds = torch.stack(cb_embeds, dim=0).squeeze(1)
        write_pickle(output_embed_path, cb_embeds)
    print(f'cb_embeds: {cb_embeds.shape}')

    tsne = TSNE(n_components=2, learning_rate='auto', init='random')
    z = tsne.fit_transform(cb_embeds.detach().numpy())
    print(f'z shape: {z.shape}')

    index = [0] + [1 for i in range(z.shape[0] - 1)]
    plt.scatter(x=z[:, 0], y=z[:, 1], c=index, s=0.5)
    # plt.scatter(x=z[:, 0], y=z[:, 1])
    output_path = os.path.join(debug_path, f'blist_embeds.pdf')
    plt.savefig(output_path, format="pdf", bbox_inches="tight")