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

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

import random

seed = 67
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

debug_path = './local/prototype/debug/single'

def distance(a, b):
    return fuzz.ratio(a, b) / 100

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
    cb_tokens_embed = embed_matrix[cb_tokens]
    cb_seq_embed, _ = model.CbRNN(cb_tokens_embed)
    cb_embed        = torch.mean(cb_seq_embed, dim=1)
    return cb_embed

@torch.no_grad()
def encode_biasword_quick(model, bpemodel, cb_tokens):
    embed_matrix  = torch.cat(
        [model.decoder.embed.weight.data, model.ooKBemb.weight], dim=0
    )
    cb_tokens_embed = embed_matrix[cb_tokens]
    cb_seq_embed, _ = model.CbRNN(cb_tokens_embed)
    cb_embed        = torch.mean(cb_seq_embed, dim=1)
    return cb_embed

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
    biasingwords, _, _, _  = model.bprocessor.select_biasing_words(
        tokens.tolist(),
        sampling_type="random",
    )
    return [''.join(word).replace('▁', '') for word in biasingwords]

def plot_tsne(model, enc_embeds, cb_embeds, cb_class, cb_types, cb_labels, uttid='test'):
    Color  = [
        "#4C72B0", 
        "#FFA500", 
        "#808080", 
        # "#008000", 
        "#E377C2", 
        "#17BECF", 
        "#BCBD22", 
        "#FF0000", 
    ]
    Shape  = [
        'o', 
        '^', 
        's', 
        # 'D', 
        '*', 
        'p', 
        'h', 
        'x'
    ]
    plt.rcParams.update({'font.size': 8})
    # plt.rcParams.update({'font.size': 2})

    # unit vector
    # enc_embeds = torch.mean(enc_embeds[16:21, :], dim=0).unsqueeze(0)
    print(f'enc_embeds shape: {enc_embeds.shape}')
    enc_embeds = model.Qproj_acoustic(enc_embeds)
    cb_embeds  = model.Kproj(cb_embeds)
    enc_embeds = enc_embeds / torch.norm(enc_embeds, dim=-1).unsqueeze(-1)
    cb_embeds  = cb_embeds / torch.norm(cb_embeds, dim=-1).unsqueeze(-1)

    # TODO: plot 11 ~ 22 FL features
    print(f'extract enc_embeds shape: {enc_embeds.shape}')

    # plot scatter
    X = torch.cat([enc_embeds, cb_embeds,], dim=0)
    C = torch.cat([
        torch.zeros(enc_embeds.shape[0]),
        cb_class + 1
    ], dim=0).to(torch.int)

    colors = [Color[c.item()] for c in C]
    shapes = [Shape[c.item()] for c in C]
    label  = [(['Query'] + cb_types)[c.item()] for c in C]
    tsne = TSNE(n_components=2, verbose=1, perplexity=10)
    X = tsne.fit_transform(X.detach())
    X = torch.from_numpy(X[:enc_embeds.shape[0] + cb_embeds.shape[0]])

    label_hited = []
    plt.figure(figsize=(5, 5))
    plt.tick_params(
        left=False, 
        right=False, 
        labelleft=False, 
        labelbottom=False, 
        bottom=False
    ) 
    for i in range(X.shape[0]):
        if label[i] not in label_hited:
            plt.scatter(
                x=X[i, 0], 
                y=X[i, 1], 
                c=colors[i], 
                # s=2, 
                s=15, 
                marker=shapes[i],
                label=label[i]
            )
            label_hited.append(label[i])
        else:
            plt.scatter(
                x=X[i, 0], 
                y=X[i, 1], 
                c=colors[i], 
                # s=2, 
                s=15, 
                marker=shapes[i],
            )
    # texts = []
    # for i in range(enc_embeds.shape[0]):
    #     x, y = X[i, 0], X[i, 1]
    #     texts.append(plt.text(x, y, i))
    # start = enc_embeds.shape[0]
    # for i in range(cb_embeds.shape[0]):
    #     x, y = X[start + i, 0], X[start + i, 1]
    #     texts.append(plt.text(x, y, cb_labels[i]))
    # adjust_text(texts)
    plt.legend()
    output_path = os.path.join(debug_path, f'{uttid}_tsne.pdf')
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.clf()
    plt.rcParams.update({'font.size': 8})

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
    labels = [f'{labels[i]} {int(cb_class[i])}' for i in range(len(labels))]

    # draw attention map
    fig, axes = plt.subplots(1, 1, figsize=(10, 27))
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
def forward(model, bpemodel, speech, text, bwords):
    bword2idx = {bwords[i]: i for i in range(len(bwords))}
    speech    = speech.unsqueeze(0)
    lengths   = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))

    start_frame = 11
    end_frame   = 22

    tokens = encode_text(bpemodel, text).unsqueeze(0)

    encoder_out, enc_olens = model.encode(speech, lengths)
    print(f'encoder_out: {encoder_out.shape}')

    queries = model._encode_query_nongrad(encoder_out)
    # queries = queries[:, start_frame:end_frame, :]
    print(f'queries: {queries.shape}')

    # Qsampling
    Q_length    = 2
    Rand_length = model.bprocessor.maxlen - Q_length
    retval      = model.bprocessor.select_biasing_words(
        tokens.tolist(),
        queries,
        sampling_type="framelevel_qsampling",
        topk=Q_length,
        unique_sorted=False
    )
    (biasingwords, worddict, cb_tokens, cb_tokens_len, mask) = retval

    bword_embeds = encode_biasword_quick(model, bpemodel, cb_tokens)

    bword2idx = model.bprocessor.bword2idx
    # idx       = [bword2idx["".join(word).replace('▁', '')] for word in worddict]
    idx       = torch.arange(cb_tokens.shape[0])
    bwords    = ["".join(key).replace('▁', '') for key in worddict]

    print(f'mask shape: {mask.shape}')

    # real baising words
    utt_bwords  = get_biasingword(tokens)
    Gold_length = len(utt_bwords)
    Gold_idx    = torch.tensor(idx[:Gold_length])
    Gold_mask   = torch.zeros(mask.shape[0], mask.shape[1], 1)
    print(f'Gold_idx: {Gold_idx}')

    OOV_mask   = torch.zeros(mask.shape[0], mask.shape[1], 1)

    Q_idx     = torch.tensor(idx[Rand_length:-1])
    Rand_idx  = torch.tensor(idx[Gold_length:Rand_length])
    Q_mask    = mask[:, :, Rand_length:-1]
    Rand_mask = mask[:, :, Gold_length:Rand_length]
    print(f'Q_idx: {Q_idx}')
    print(f'Rand_idx: {Rand_idx}')

    # contextual biasing embeddings
    idxs  = [
        Gold_idx.to(torch.long), 
        Rand_idx.to(torch.long), 
        # Q_idx.to(torch.long),
    ]
    types = [
        'Gold BW', 
        'Random BW', 
        # 'Q-Sampling BW',
    ]
    print(idxs)
    cb_embed, labels, cb_class, cb_types = get_negative_samples(
        bword_embeds, 
        bwords, 
        idxs, 
        types,
        # use_oov=False
        use_oov=True
    )

    decoder_in, target, t_len, u_len = get_transducer_task_io(
        tokens,
        enc_olens,
        ignore_id=-1,
        blank_id=model.blank_id,
    )
    decoder_out = model.decoder(decoder_in)
    print(f'decoder_out: {decoder_out.shape}')

    mask = torch.cat([Gold_mask, Rand_mask, Q_mask, OOV_mask], axis=-1)
    print(f'mask shape: {mask.shape}')
    aco_bias, aco_atten = model.get_acoustic_biasing_vector(
        encoder_out, 
        cb_embed, 
        return_atten=True, 
        # mask=mask
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
        encoder_out.squeeze(0)[start_frame:end_frame, :],
        cb_embed,
        labels,
        cb_class,
        cb_types,
        uttid='test',
    )

if __name__ == "__main__":
    spm_path   = "./data/en_token_list/bpe_unigram600suffix/bpe.model"
    token_path = "./data/en_token_list/bpe_unigram600suffix/tokens.txt"
    model_conf = "./conf/tuning/train_rnnt_freeze_contextual_biasing_sampling_FL_Qsampling.yaml"
    # model_path = "./exp/asr_finetune_freeze_conformer_transducer_contextual_biasing_proj_suffix/valid.loss.ave_10best.pth"
    model_path = "./exp/asr_finetune_freeze_conformer_transducer_contextual_biasing_proj_QSamlping_warmup_suffix/latest.pth"
    stats_path = "./exp/asr_stats_raw_en_bpe600_spsuffix/train/feats_stats.npz"
    rare_path  = "./local/rareword_f15.txt"
    scp_path   = "./data/train_clean_100/wav.scp"
    ref_path   = "./data/train_clean_100/text"
    # scp_path   = "./data/test_clean/wav.scp"
    # ref_path   = "./data/test_clean/text"

    model, bpemodel, tokenizer, converter = load_espnet_model(
        model_conf, 
        token_path, 
        stats_path, 
        spm_path, 
        model_path
    )
    model.eval()

    texts  = {d[0]: " ".join(d[1:]) for d in read_file(ref_path, sp=' ')}
    wavscp = [[d[0], d[1], texts[d[0]]] for d in read_file(scp_path, sp=' ')]
    bwords = [d[0] for d in read_file(rare_path, sp=' ')]

    embed_matrix = model.get_bias_embeds()
    model.bprocessor.build_index(embed_matrix, model.CbRNN, model.Kproj, use_gpu=False)

    encs = []
    enc_lengths = []
    
    model.bprocessor.maxlen = 20
    for idx, audio_path, text in wavscp:
        if idx != "1963-142776-0027":
            continue
        print(text)
        speech, sample_rate = torchaudio.load(audio_path)
        speech = speech.reshape(-1)
        forward(model, bpemodel, speech, text, bwords)
        # break