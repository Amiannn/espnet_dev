import os
import torch
import faiss
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

from local.utils import read_file
from local.utils import read_pickle
from local.utils import write_json
from local.utils import write_file
from local.utils import write_pickle

from tqdm import tqdm

from local.prototype.utils.model     import load_espnet_model
from local.prototype.utils.decode    import infernece
from local.prototype.utils.alignment import forward_backward as alignment

from espnet2.asr_transducer.utils import get_transducer_task_io

from sklearn.cluster  import KMeans
from sklearn.manifold import TSNE
from adjustText       import adjust_text

import random

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

debug_path = './local/prototype/debug/test'

@torch.no_grad()
def encode_text(bpemodel, text):
    return torch.tensor(bpemodel.encode(text)) - 1

@torch.no_grad()
def encode_biasword(model, bpemodel, bwords):
    btokens       = [encode_text(bpemodel, bword) for bword in bwords]
    blength       = [btoken.shape[0] for btoken in btokens]
    # with ooKB
    btokens.append(torch.tensor([len(model.token_list)], dtype=torch.long))        
    blength.append(1)

    cb_tokens     = torch.nn.utils.rnn.pad_sequence(btokens, batch_first=True)
    cb_tokens_len = torch.tensor(blength).to(torch.int64)
    embed_matrix  = torch.cat(
        [model.decoder.embed.weight.data, model.ooKBemb.weight], dim=0
    )
    cb_token_embed   = embed_matrix[cb_tokens]
    cb_tokens_packed = torch.nn.utils.rnn.pack_padded_sequence(
        cb_token_embed, 
        cb_tokens_len,
        batch_first=True,
        enforce_sorted=False
    )
    cb_seq_embed_packed, _ = model.CbRNN(cb_tokens_packed)
    cb_seq_embed_packed, input_sizes = torch.nn.utils.rnn.pad_packed_sequence(
        cb_seq_embed_packed, 
        batch_first=True
    )
    cb_embeds = torch.sum(cb_seq_embed_packed, dim=1) / input_sizes.unsqueeze(-1)
    return cb_embeds

@torch.no_grad()
def get_negative_samples(bword_embeds, bwords, idxs, types, use_oov=True):
    _bwords = bwords
    if use_oov:
        idxs.append(torch.tensor([len(bwords)]))
        types.append('OOV')
        _bwords = bwords + ['OOV']
    embeds       = [bword_embeds[idx] for idx in idxs]
    embeds_class = torch.cat([torch.zeros(embeds[i].shape[0]) + i for i in range(len(embeds))])
    embeds       = torch.cat(embeds, dim=0)
    labels       = [_bwords[idx] for idx in torch.cat(idxs)]
    return embeds, labels, embeds_class, types

@torch.no_grad()
def get_biasingword(tokens):
    biasingwords, _, _, _, _  = model.bprocessor.select_biasing_words(
        tokens.tolist(), 
        cb=True,
        ret_worddict=True
    )
    return [''.join(word).replace('▁', '') for word in biasingwords]

def plot_tsne(model, enc_embeds, cb_embeds, cb_class, cb_types, cb_labels, uttid='test'):
    Color  = ["#4C72B0", "#FFA500", "#008000", "#E377C2", "#FF0000", "#808080"]
    Shape  = ['o', '^', '.', '*', 'x']
    plt.rcParams.update({'font.size': 6})

    # unit vector
    enc_embeds = model.Qproj_acoustic(enc_embeds)
    cb_embeds  = model.Kproj(cb_embeds)
    enc_embeds = enc_embeds / torch.norm(enc_embeds, dim=0)
    cb_embeds  = cb_embeds / torch.norm(cb_embeds, dim=0)

    # plot scatter
    X = torch.cat([enc_embeds, cb_embeds], dim=0)
    C = torch.cat([
        torch.zeros(enc_embeds.shape[0]), 
        cb_class + 1
    ], dim=0).to(torch.int)

    colors = [Color[c.item()] for c in C]
    shapes = [Shape[c.item()] for c in C]

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=400)
    X = tsne.fit_transform(X.detach())

    for i in range(X.shape[0]):
        plt.scatter(
            x=X[i, 0], 
            y=X[i, 1], 
            c=colors[i], 
            # s=2, 
            s=5, 
            marker=shapes[i]
        )
    texts = []
    for i in range(enc_embeds.shape[0]):
        x, y = X[i, 0], X[i, 1]
        texts.append(plt.text(x, y, i))
    start = enc_embeds.shape[0]
    for i in range(cb_embeds.shape[0]):
        x, y = X[start + i, 0], X[start + i, 1]
        texts.append(plt.text(x, y, cb_labels[i]))
    adjust_text(texts)
    output_path = os.path.join(debug_path, f'{uttid}_tsne.pdf')
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.clf()
    plt.rcParams.update({'font.size': 6})

def plot_attention_map(
    model,
    speech,
    text,
    biasingwords,
    target, 
    logp, 
    attention,
    enc_embeds,
    cb_embeds,
    labels,
    cb_class,
    cb_types,
    uttid='test',
):
    alignments = alignment(
        logp, 
        target, 
        model.blank_id, 
        model.token_list, 
        speech[0]
    )
    attention   = attention.squeeze(0).T.detach().cpu().resolve_conj().resolve_neg().numpy()
    frame2align = {start: token for token, start, end in alignments}
    xlabels = [
        f'{frame2align[i]} {i}' if i in frame2align else f'{i}' for i in range(attention.shape[1])
    ]
    print(f'xlabels: {len(xlabels)}')

    plot_tsne(model, enc_embeds, cb_embeds, cb_class, cb_types, labels)

    # draw attention map
    fig, axes = plt.subplots(1, 1, figsize=(40, 10))
    axes.xaxis.set_ticks(np.arange(0, attention.shape[1], 1))
    axes.yaxis.set_ticks(np.arange(0, attention.shape[0], 1))
    axes.set_xticks(np.arange(-.5, attention.shape[1], 10), minor=True)
    axes.set_yticks(np.arange(-.5, attention.shape[0], 1), minor=True)
    axes.set_xticklabels(xlabels, rotation=90)
    axes.set_yticklabels(labels)

    axes.imshow(attention, aspect='auto')
    axes.grid(which='minor', color='w', linewidth=0.5, alpha=0.3)
    plt.title(text)
    output_path = os.path.join(debug_path, f'{uttid}_attention_map.pdf')
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.clf()

@torch.no_grad()
def forward(model, bpemodel, speech, text, bwords, bword_embeds):
    bword2idx = {bwords[i]: i for i in range(len(bwords))}
    speech    = speech.unsqueeze(0)
    lengths   = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))

    tokens = encode_text(bpemodel, text).unsqueeze(0)
    encoder_out, enc_olens = model.encode(speech, lengths)
    print(f'encoder_out: {encoder_out.shape}')

    # real baising words
    utt_bwords  = get_biasingword(tokens)
    utt_idx     = torch.tensor([bword2idx[bword] for bword in utt_bwords])
    print(f'utt_idx: {utt_idx}')
    
    # exhausted biasing words
    exhaust_num = 10
    enc_proj    = model.Qproj_acoustic(encoder_out).squeeze(0)
    enc_proj    = enc_proj / torch.norm(enc_proj, dim=0)
    cb_proj     = model.Kproj(bword_embeds[:-1, :])
    cb_proj     = cb_proj / torch.norm(cb_proj, dim=0)
    indexis     = faiss.IndexFlatIP(cb_proj.shape[-1])
    indexis.add(cb_proj)
    D, I        = indexis.search(enc_proj, 1)
    exhaust_idx = torch.from_numpy(I.reshape(-1))
    exhaust_idx = torch.unique(exhaust_idx, sorted=False)
    print(f'exhaust_idx: {exhaust_idx}')

    # random baising words
    rand_num = exhaust_idx.shape[0]
    rand_idx = torch.randint(len(bwords) - 1, (rand_num, ))
    print(f'rand_idx: {rand_idx}')

    # contextual biasing embeddings
    idxs  = [utt_idx, rand_idx, exhaust_idx]
    types = ['real', 'random', 'exhaust']
    cb_embed, labels, cb_class, cb_types = get_negative_samples(bword_embeds, bwords, idxs, types)

    decoder_in, target, t_len, u_len = get_transducer_task_io(
        tokens,
        enc_olens,
        ignore_id=-1,
        blank_id=model.blank_id,
    )
    decoder_out = model.decoder(decoder_in)
    print(f'decoder_out: {decoder_out.shape}')

    aco_bias, aco_atten = model.get_acoustic_biasing_vector(
        encoder_out, cb_embed, return_atten=True
    )

    lin_encoder_out = model.joint_network.lin_enc(encoder_out)
    lin_decoder_out = model.joint_network.lin_dec(decoder_out)
    lin_encoder_out = lin_encoder_out + aco_bias

    joint_out = model.joint_network.joint_activation(
        lin_encoder_out.unsqueeze(2) + lin_decoder_out.unsqueeze(1)
    )
    join_out = model.joint_network.lin_out(joint_out)
    logp     = torch.log_softmax(join_out, dim=-1)[0].transpose(1, 0)
    print(f'logp: {logp.shape}')
    plot_attention_map(
        model,
        speech,
        text,
        utt_bwords,
        target[0], 
        logp, 
        aco_atten, 
        encoder_out.squeeze(0),
        cb_embed,
        labels,
        cb_class,
        cb_types,
        uttid='test',
    )

if __name__ == "__main__":
    spm_path   = "./data/en_token_list/bpe_unigram600suffix/bpe.model"
    token_path = "./data/en_token_list/bpe_unigram600suffix/tokens.txt"
    model_conf = "./conf/tuning/train_rnnt_freeze_contextual_biasing.yaml"
    model_path = "./exp/asr_finetune_freeze_conformer_transducer_contextual_biasing_proj_encode_oov_suffix/valid.loss.ave_10best.pth"
    stats_path = "./exp/asr_stats_raw_en_bpe600_spsuffix/train/feats_stats.npz"
    rare_path  = "./local/rareword_f15.txt"
    scp_path   = "./data/train_clean_100/wav.scp"
    ref_path   = "./data/train_clean_100/text"

    model, bpemodel, tokenizer, converter = load_espnet_model(
        model_conf, 
        token_path, 
        stats_path, 
        spm_path, 
        model_path
    )

    texts  = {d[0]: " ".join(d[1:]) for d in read_file(ref_path, sp=' ')}
    wavscp = [[d[0], d[1], texts[d[0]]] for d in read_file(scp_path, sp=' ')]
    bwords = [d[0] for d in read_file(rare_path, sp=' ')]

    bword_embeds = encode_biasword(model, bpemodel, bwords)

    encs = []
    enc_lengths = []
    for idx, audio_path, text in wavscp:
        if idx != "1447-130551-0003":
            continue
        print(text)
        speech, sample_rate = torchaudio.load(audio_path)
        speech = speech.reshape(-1)
        forward(model, bpemodel, speech, text, bwords, bword_embeds)