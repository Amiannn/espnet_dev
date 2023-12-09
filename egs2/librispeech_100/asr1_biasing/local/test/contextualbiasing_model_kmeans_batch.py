import os
import json
import faiss
import torch
import logging
import numpy as np
import sentencepiece as spm

from local.utils import read_file
from local.utils import read_pickle
from local.utils import write_json
from local.utils import write_file

from torch.nn.utils.rnn import pad_sequence

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from espnet2.tasks.asr import ASRTask
from espnet2.bin.asr_inference import Speech2Text
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.asr.transducer.beam_search_transducer import BeamSearchTransducer
from espnet2.asr.transducer.beam_search_transducer import Hypothesis
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError

from espnet2.asr_transducer.utils import get_transducer_task_io

import math  
import textgrid
import matplotlib.pyplot as plt

debug_path = './local/test/debug'

config = {
    "log_level": "INFO",
    "output_dir": "exp/asr_finetune_freeze_conformer_transducer_contextual_biasing_proj_suffix/decode_asr_tiny_asr_model_valid.loss.ave_10best/test_clean/logdir/output.1",
    "ngpu": 0,
    "seed": 0,
    "dtype": "float32",
    "num_workers": 1,
    "data_path_and_name_and_type": [
        [
            "dump/raw/test_clean/wav.scp",
            "speech",
            "kaldi_ark"
        ]
    ],
    "key_file": "exp/asr_finetune_freeze_conformer_transducer_contextual_biasing_proj_suffix/decode_asr_tiny_asr_model_valid.loss.ave_10best/test_clean/logdir/keys.1.scp",
    "allow_variable_data_keys": False,
    "asr_train_config": "exp/asr_finetune_freeze_conformer_transducer_contextual_biasing_proj_suffix/config.yaml",
    "asr_model_file": "exp/asr_finetune_freeze_conformer_transducer_contextual_biasing_proj_suffix/valid.loss.ave_10best.pth",
    "lm_train_config": None,
    "lm_file": None,
    "word_lm_train_config": None,
    "word_lm_file": None,
    "ngram_file": None,
    "model_tag": None,
    "enh_s2t_task": False,
    "multi_asr": False,
    "quantize_asr_model": False,
    "quantize_lm": False,
    "quantize_modules": [
        "Linear"
    ],
    "quantize_dtype": "qint8",
    "batch_size": 1,
    "nbest": 1,
    "beam_size": 20,
    "penalty": 0.0,
    "maxlenratio": 0.0,
    "minlenratio": 0.0,
    "ctc_weight": 0.0,
    "lm_weight": 0.0,
    "ngram_weight": 0.9,
    "streaming": False,
    "hugging_face_decoder": False,
    "hugging_face_decoder_conf": {},
    "transducer_conf": None,
    "token_type": None,
    "bpemodel": None,
    "time_sync": False,
    "perutt_blist": "data/test_clean/perutt_blist.json",
    "biasinglist": "local/all_rare_words.txt",
    "bmaxlen": 10,
    "bdrop": 0.0
}

def model_forward(
    model,
    speech: torch.Tensor,
    speech_lengths: torch.Tensor,
    text: torch.Tensor,
    text_lengths: torch.Tensor,
    uttid: str,
    ref_text: str,
    enc_cluster: torch.Tensor,
    enc_centers: torch.Tensor,
    bpemodel: str,
    **kwargs,
):
    """Frontend + Encoder + Decoder + Calc loss

    Args:
        speech: (Batch, Length, ...)
        speech_lengths: (Batch, )
        text: (Batch, Length)
        text_lengths: (Batch,)
        kwargs: "utt_id" is among the input.
    """
    assert text_lengths.dim() == 1, text_lengths.shape
    batch_size = speech.shape[0]

    # for data-parallel
    text = text[:, : text_lengths.max()]

    # 1. Encoder
    encoder_out, encoder_out_lens = model.encode(speech, speech_lengths)
    lin_encoder_out = model.joint_network.lin_enc(encoder_out)
    # lin_encoder_out = lin_encoder_out + aco_bias

    # 1.2 Acoustic biasing
    (
        biasingwords, 
        lextree, 
        cb_tokens, 
        cb_tokens_len,
        worddict
    ) = model.bprocessor.select_biasing_words(
        text.tolist(), 
        cb=True,
        ret_worddict=True
    )
    print(f'cb_tokens: {cb_tokens}')
    print(f'worddict: {worddict}')
    # labels       = [i for i in range(len(worddict) + len(biasingwords))] + ['<OOL>']
    # labels       = [i for i in range(len(biasingwords))] + ['<OOL>']
    labels       = [i for i in range(len(biasingwords))]
    # labels       = [i for i in range(len(worddict))]
    biasingwords = [''.join(word).replace('▁', '') for word in biasingwords]
    print(f'labels: {labels}')
    model.decoder.set_device(encoder_out.device)
    print(f'ref_text: {ref_text}')
    print(f'biasingwords: {biasingwords}')
    ref_text = [text if text not in biasingwords else f'<{text}>' for text in ref_text.split(' ')]
    ref_text = " ".join(ref_text)

    enc = speech2text.asr_model.Qproj_acoustic(encoder_out)
    enc = enc.squeeze(0).detach()
    # _centers = torch.from_numpy(centers)
    # _b_embeds = torch.from_numpy(b_embeds)
    # topk = torch.argsort(
    #     torch.einsum("cd,bd->cb", enc, b_embeds), 
    #     descending=True, 
    #     dim=-1
    # )[:, :10].numpy()
    # print(f'topk: {topk.shape}')

    D, topk = indexis.search(enc, 10)
    print(f'D: {D}')
    print(f'topk: {topk}')
    print(f'topk: {topk.shape}')

    frame2blist = []
    frame2bsubs = []
    bembeds = []
    embed_matrix  = torch.cat(
        [model.decoder.embed.weight.data, model.ooKBemb.weight], dim=0
    )
    realbwords = []
    for bword in biasingwords:
        tokens = torch.tensor(bpemodel.encode(bword)).unsqueeze(0)
        cb_tokens_embed = embed_matrix[tokens]
        cb_seq_embed, _ = model.CbRNN(cb_tokens_embed)
        cb_embed = torch.mean(cb_seq_embed, dim=1)
        realbwords.append(cb_embed)

    oov = torch.tensor([[600]])
    oov_embed, _ = model.CbRNN(embed_matrix[oov])
    oov_embed = torch.mean(oov_embed, dim=1)
    for i in range(enc.shape[0]):
        # blist_str = ", ".join(label2bwords[str(topk[i])])
        frame = []
        frame2bsub = []
        bembed = []
        for j in range(topk.shape[-1]):
            bembed.append(b_embeds[topk[i][j]])
            frame.append(f'{j}. {bwords[topk[i][j]]}')
            tokens = torch.tensor(bpemodel.encode(bwords[topk[i][j]])).unsqueeze(0)
            cb_tokens_embed = embed_matrix[tokens]
            print(f'cb_tokens_embed.shape: {cb_tokens_embed.shape}')
            cb_seq_embed, _ = model.CbRNN(cb_tokens_embed)
            cb_embed = torch.mean(cb_seq_embed, dim=1)
            frame2bsub.append(cb_embed)
        bembeds.append(torch.stack(bembed))
        # frame2bsub = torch.stack(frame2bsub + realbwords + [oov_embed]).squeeze(1)
        # frame2bsub = torch.stack(realbwords + [oov_embed]).squeeze(1)
        frame2bsub = torch.stack(realbwords).squeeze(1)
        print(f'frame2bsub.shape: {frame2bsub.shape}')
        frame2bsubs.append(frame2bsub)
        # frame2blist.append(", ".join(frame + [f'{k}. {biasingwords[k]}' for k in range(len(biasingwords))] + ['<oov>']))
        frame2blist.append(", ".join([f'{k}. {biasingwords[k]}' for k in range(len(biasingwords))] + ['<oov>']))
    bembeds = torch.stack(bembeds)
    print(f'bembeds shape: {bembeds.shape}')
    # print(f'enc_cluster: {enc_centers.shape}')
    # _, topk = kmeans.index.search(enc_centers, 1)
    # topk = topk.squeeze(-1)
    # print(f'topk: {topk}')
    # print(f'topk: {topk.shape}')

    # frame2blist = []
    # for i in range(enc_cluster.shape[0]):
    #     blist_str = ", ".join(label2bwords[str(topk[enc_cluster[i]])])
    #     frame2blist.append(blist_str)

    aco_attens = []
    
    for t in range(encoder_out.shape[1]):
        cb_embed = frame2bsubs[t]
        # cb_embed = bembeds[t]
        query = encoder_out[0, t, :].reshape(1, 1, -1)
        # print(f'query {t}: {query.shape}')
        # print(f'cb_embed {t}: {cb_embed.shape}')
        aco_bias, aco_atten = model.get_acoustic_biasing_vector(
            query, 
            cb_embed, 
            return_atten=True
        )
        aco_attens.append(aco_atten.squeeze(1))

    aco_atten = torch.stack(aco_attens).squeeze(1)
    print(f'aco_atten: {aco_atten}')
    print(f'aco_atten shape: {aco_atten.shape}')
    # 2a. Transducer decoder branch
    decoder_in, target, t_len, u_len = get_transducer_task_io(
        text,
        encoder_out_lens,
        ignore_id=-1,
        blank_id=model.blank_id,
    )
    model.decoder.set_device(encoder_out.device)
    decoder_out = model.decoder(decoder_in)

    
    lin_decoder_out = model.joint_network.lin_dec(decoder_out)

    joint_out = model.joint_network.joint_activation(
        lin_encoder_out.unsqueeze(2) + lin_decoder_out.unsqueeze(1)
    )
    join_out = model.joint_network.lin_out(joint_out)
    
    logp = torch.log_softmax(
        join_out,
        dim=-1,
    )[0]

    logp   = logp.to('cpu').transpose(1, 0)
    target = target.to('cpu')[0]
    print(f'joint out dim: {logp.shape}')

    alignments = forward_backward(
        logp, 
        target, 
        model.blank_id, 
        model.token_list, 
        speech.to('cpu')[0]
    )
    aco_atten   = aco_atten.squeeze(0).T.detach().cpu().resolve_conj().resolve_neg().numpy()
    frame2align = {start: token for token, start, end in alignments}
    xlabels = [
        f'({frame2blist[i]}) {frame2align[i]} {enc_cluster[i]}' if i in frame2align else f'{enc_cluster[i]}' for i in range(aco_atten.shape[1])
    ]
    print(f'xlabels: {len(xlabels)}')

    # draw attention map
    fig, axes = plt.subplots(1, 1, figsize=(40, 10))
    axes.xaxis.set_ticks(np.arange(0, aco_atten.shape[1], 1))
    axes.yaxis.set_ticks(np.arange(0, aco_atten.shape[0], 1))
    axes.set_xticks(np.arange(-.5, aco_atten.shape[1], 10), minor=True)
    axes.set_yticks(np.arange(-.5, aco_atten.shape[0], 1), minor=True)
    axes.set_xticklabels(xlabels, rotation=90)
    axes.set_yticklabels(labels)

    axes.imshow(aco_atten, aspect='auto')
    axes.grid(which='minor', color='w', linewidth=0.5, alpha=0.3)
    plt.title(ref_text)
    output_path = os.path.join(debug_path, f'{uttid}_BA_kmeans.pdf')
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.clf()
    return logp

def get_vertical_transition(logp, target):
    u_len, t_len = logp.shape[:2]

    u_idx = torch.arange(0, u_len - 1, dtype=torch.long).repeat(t_len, 1).T.reshape(-1)
    t_idx = torch.arange(0, t_len, dtype=torch.long).repeat(u_len - 1)
    y_idx = target.repeat(t_len, 1).T.reshape(-1).long()

    y_logp = torch.zeros(u_len, t_len)
    y_logp[:-1, :] = logp[u_idx, t_idx, y_idx].reshape(u_len - 1, t_len)
    y_logp[-1:, :] = -float("inf")
    return y_logp

def get_horizontal_transition(logp, target, blank_id):
    u_len, t_len = logp.shape[:2]

    u_idx   = torch.arange(0, u_len, dtype=torch.long).repeat(t_len - 1, 1).T.reshape(-1)
    t_idx   = torch.arange(0, t_len - 1, dtype=torch.long).repeat(u_len)
    phi_idx = torch.zeros(u_len * (t_len - 1), dtype=torch.long) + blank_id

    phi_logp = torch.zeros(u_len, t_len)
    phi_logp[:, :-1] = logp[u_idx, t_idx, phi_idx].reshape(u_len, t_len - 1)
    # phi_logp[:, -1:] = -float("inf")
    phi_logp[:, -1:] = -float("inf")
    # phi_logp[-1, -1] = 0
    return phi_logp

@torch.no_grad()
def forward_backward(logp, target, blank_id, token_list, waveform):
    u_len, t_len = logp.shape[:2]

    y_logp   = get_vertical_transition(logp, target)
    phi_logp = get_horizontal_transition(logp, target, blank_id)

    alpha = torch.zeros(u_len, t_len)
    zero_tensor = torch.zeros(1)
    inf_tensor  = torch.zeros(1) + -float("inf")
    for u in range(u_len):
        for t in range(t_len):
            if u == 0 and t == 0: continue
            alpha_y_partial   = alpha[u - 1, t] + y_logp[u - 1, t] if (u - 1) >= 0 else inf_tensor
            alpha_phi_partial = alpha[u, t - 1] + phi_logp[u, t - 1] if (t - 1) >= 0 else inf_tensor
            alpha[u, t] = torch.logaddexp(alpha_y_partial, alpha_phi_partial)
    plt.imshow(alpha, origin="lower")
    output_path = os.path.join(debug_path, 'alpha.pdf')
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.clf()

    beta = torch.zeros(u_len, t_len)
    for u in range(u_len - 1, -1, -1):
        for t in range(t_len - 1, -1, -1):
            if u == (u_len - 1) and t == (t_len - 1): continue
            beta_y_partial   = beta[u + 1, t] + y_logp[u, t] if (u + 1) < u_len else inf_tensor
            beta_phi_partial = beta[u, t + 1] + phi_logp[u, t] if (t + 1) < t_len else inf_tensor
            beta[u, t] = torch.logaddexp(beta_y_partial, beta_phi_partial)

    plt.imshow(beta, origin="lower")
    output_path = os.path.join(debug_path, 'beta.pdf')
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.clf()

    ab_log_prob = (alpha + beta)
    ab_prob     = torch.exp(ab_log_prob)

    plt.imshow(ab_log_prob, origin="lower")
    output_path = os.path.join(debug_path, 'ab_log_prob.pdf')
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.clf()

    plt.imshow(ab_prob, origin="lower")
    output_path = os.path.join(debug_path, 'ab.pdf')
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.clf()

    last_token  = 0
    align_paths = []
    align_path  = []
    target      = [0] + target.tolist()
    tokens      = [token_list[t] for t in target]
    
    now_u, now_t = u_len - 1, t_len - 1
    while (now_u >= 0 and now_t >= 0):
        if ab_prob[now_u, now_t - 1] > ab_prob[now_u - 1, now_t]:
            # stay
            now_t -= 1
            align_path = [now_t] + align_path
        else:
            # leave
            now_u -= 1
            if len(align_path) == 0:
                align_path = [now_t]
            align_paths = [align_path] + align_paths
            align_path = []

    fig, [ax1, ax2] = plt.subplots(
        2, 1,
        # figsize=(50, 10)
    )
    ax1.imshow(ab_prob, origin="lower", aspect='auto', interpolation='none')
    ax1.set_facecolor("lightgray")
    ax1.set_xticks([])
    ax1.set_yticks([])

    sample_rate = 16000
    # The original waveform
    ratio = waveform.size(0) / sample_rate / t_len
    maxTime = math.ceil(waveform.size(0) / sample_rate)
    alignments = []
    tg = textgrid.TextGrid(minTime=0, maxTime=maxTime)
    tier_word = textgrid.IntervalTier(name="subword", minTime=0., maxTime=maxTime)
    for i, align_path in enumerate(align_paths):
        token = tokens[i]
        if len(align_path) == 0:
            continue
        start, end = min(align_path), max(align_path)
        if start == end:
            continue
        # print(f'{token}\t{start}\t{end}')
        alignments.append([token, start, end])
        start    = ratio * start
        end      = ratio * end
        interval = textgrid.Interval(minTime=start, maxTime=end, mark=token)
        tier_word.addInterval(interval)
        start = f'{start:.2f}'
        end   = f'{end:.2f}'
    tg.tiers.append(tier_word)
    tg.write("1.TextGrid")
    # output_path = './alignment.tsv'
    # write_file(output_path, alignments, sp='\t')
    ax2.specgram(waveform, Fs=sample_rate)
    ax2.set_yticks([])
    ax2.set_xlabel("time [second]")
    fig.tight_layout()
    output_path = os.path.join(debug_path, 'ab_prob.pdf')
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.clf()
    return alignments

def inference(
    output_dir: str,
    maxlenratio: float,
    minlenratio: float,
    batch_size: int,
    dtype: str,
    beam_size: int,
    ngpu: int,
    seed: int,
    ctc_weight: float,
    lm_weight: float,
    ngram_weight: float,
    penalty: float,
    nbest: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    asr_train_config: Optional[str],
    asr_model_file: Optional[str],
    lm_train_config: Optional[str],
    lm_file: Optional[str],
    word_lm_train_config: Optional[str],
    word_lm_file: Optional[str],
    ngram_file: Optional[str],
    model_tag: Optional[str],
    token_type: Optional[str],
    bpemodel: Optional[str],
    allow_variable_data_keys: bool,
    transducer_conf: Optional[dict],
    streaming: bool,
    enh_s2t_task: bool,
    quantize_asr_model: bool,
    quantize_lm: bool,
    quantize_modules: List[str],
    quantize_dtype: str,
    hugging_face_decoder: bool,
    hugging_face_decoder_conf: Dict[str, Any],
    time_sync: bool,
    multi_asr: bool,
    perutt_blist: str = "",
    biasinglist: str = "",
    bmaxlen: int = 0,
    bdrop: float = 0.0,
):
    # device = "cpu" if ngpu < 1 else "cuda"
    device = "cpu"
    
    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build speech2text
    speech2text_kwargs = dict(
        asr_train_config=asr_train_config,
        asr_model_file=asr_model_file,
        transducer_conf=transducer_conf,
        lm_train_config=lm_train_config,
        lm_file=lm_file,
        ngram_file=ngram_file,
        token_type=token_type,
        bpemodel=bpemodel,
        device=device,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        dtype=dtype,
        beam_size=beam_size,
        ctc_weight=ctc_weight,
        lm_weight=lm_weight,
        ngram_weight=ngram_weight,
        penalty=penalty,
        nbest=nbest,
        streaming=streaming,
        enh_s2t_task=enh_s2t_task,
        multi_asr=multi_asr,
        quantize_asr_model=quantize_asr_model,
        quantize_lm=quantize_lm,
        quantize_modules=quantize_modules,
        quantize_dtype=quantize_dtype,
        hugging_face_decoder=hugging_face_decoder,
        hugging_face_decoder_conf=hugging_face_decoder_conf,
        time_sync=time_sync,
        biasinglist=biasinglist,
        bmaxlen=bmaxlen,
        bdrop=bdrop,
    )
    speech2text = Speech2Text.from_pretrained(
        model_tag=model_tag,
        **speech2text_kwargs,
    )

    multi_blank_durations = getattr(
        speech2text.asr_model, "transducer_multi_blank_durations", []
    )[::-1] + [1]
    multi_blank_indices = [
        speech2text.asr_model.blank_id - i + 1
        for i in range(len(multi_blank_durations), 0, -1)
    ]
    if transducer_conf is None:
        transducer_conf = {}
    speech2text.beam_search_transducer = BeamSearchTransducer(
        decoder=speech2text.asr_model.decoder,
        joint_network=speech2text.asr_model.joint_network,
        beam_size=beam_size,
        lm=None,
        lm_weight=lm_weight,
        multi_blank_durations=multi_blank_durations,
        multi_blank_indices=multi_blank_indices,
        token_list=speech2text.asr_model.token_list,
        biasing=getattr(speech2text.asr_model, "biasing", False),
        deepbiasing=getattr(speech2text.asr_model, "deepbiasing", False),
        BiasingBundle=None,
        **transducer_conf,
    )

    # 3. Build data-iterator
    preprocess = ASRTask.build_preprocess_fn(speech2text.asr_train_args, False)
    loader = ASRTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=preprocess,
        collate_fn=ASRTask.build_collate_fn(speech2text.asr_train_args, False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )
    # load biasing list
    blist_perutt = None
    if getattr(speech2text.asr_model, "biasing", False) and perutt_blist != "":
    # if speech2text.asr_model.biasing and perutt_blist != "":
        with open(perutt_blist) as fin:
            blist_perutt = json.load(fin)
    elif perutt_blist == "":
        speech2text.asr_model.biasing = False
        if speech2text.asr_model.use_transducer_decoder:
            speech2text.beam_search_transducer.biasing = False

    return speech2text, loader, preprocess

embed_path = './local/bword_embeds.pickle'
rare_path  = "./local/rareword_f15.txt"
bwords     = [d[0] for d in read_file(rare_path, sp=' ')]
b_embeds   = read_pickle(embed_path)

k = 1000
kmeans = faiss.Kmeans(
    b_embeds.shape[-1], 
    k, 
    niter=20, 
    verbose=True,
    gpu=True
)
kmeans.train(b_embeds)
centers = kmeans.centroids
print(f'centers: {centers.shape}')
labels = kmeans.index.search(x=b_embeds, k=1)[1].reshape(-1)
print(f'labels: {labels}')
print(f'labels: {labels.shape}')
indexis = faiss.IndexFlatIP(b_embeds.shape[-1])
indexis.add(b_embeds)

label2bwords = {}
for label, bword in zip(labels, bwords):
    label = str(label)
    label2bwords[label] = label2bwords[label] + [bword] if label in label2bwords else [bword]

for label in label2bwords:
    label2bwords[label] = sorted(label2bwords[label])
    # print(f'{label}: {len(label2bwords[label])}')
output_path = f'./local/cluster_embeds_{k}.json'
write_json(output_path, label2bwords)

if __name__ == "__main__":
    ref_path = "./dump/raw/test_clean/text"
    refs     = {d[0]: " ".join(d[1:]) for d in read_file(ref_path, sp=' ')}

    speech2text, loader, preprocess = inference(**config)
    
    debug_embed_results = {n: [] for n in range(1, 11)}
    bpemodel = spm.SentencePieceProcessor(speech2text.asr_model.bpemodel)
    count = 0
    encs = []
    enc_lengths = []
    _keys = []
    for keys, batch in loader:
        batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}

        ref_texts      = [refs[key] for key in keys]
        speech         = batch['speech'].reshape(1, -1)
        batch_size     = 1
        speech_lengths = torch.zeros(batch_size, dtype=torch.int) + speech.shape[-1]
        # b. Forward Encoder
        enc, enc_olens = speech2text.asr_model.encode(speech, speech_lengths)
        enc = speech2text.asr_model.Qproj_acoustic(enc)
        enc = enc.squeeze(0)
        encs.append(enc)
        enc_lengths.append(enc.shape[0])
        _keys.append(keys[0])
        # if count > 10:
        #     break
        # count += 1
        break
    encs = torch.cat(encs, dim=0).detach()
    enc_lengths = torch.tensor(enc_lengths)
    
    print(f'encs: {encs.shape}')
    print(f'enc_lengths: {enc_lengths}')
    print(f'enc_lengths: {enc_lengths.shape}')

    enc_k = 10
    enc_kmeans = faiss.Kmeans(
        encs.shape[-1], 
        enc_k, 
        niter=20, 
        verbose=True,
        gpu=True
    )
    enc_kmeans.train(encs)
    enc_centers = enc_kmeans.centroids
    print(f'centers: {enc_centers.shape}')
    labels = enc_kmeans.index.search(x=encs, k=1)[1].reshape(-1)
    last = 0
    key2cluster = {}
    for i in range(enc_lengths.shape[0]):
        now = last + enc_lengths[i]
        key2cluster[_keys[i]] = labels[last:now]
        last = now
    print(key2cluster)

    count = 0
    for keys, batch in loader:
        batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}

        ref_texts      = [refs[key] for key in keys]
        speech         = batch['speech'].reshape(1, -1)
        batch_size     = 1
        speech_lengths = torch.zeros(batch_size, dtype=torch.int) + speech.shape[-1]
        tokens         = [preprocess._text_process({'text': t})['text'] for t in ref_texts[0].split(' ')]

        _tokens = []
        for t in tokens:
            # _tokens = _tokens + t.tolist() + [28]
            _tokens = _tokens + t.tolist()
        tokens = np.array([_tokens])
        print(tokens)

        text           = torch.tensor(tokens, dtype=torch.int)
        text_lengths   = torch.zeros(batch_size, dtype=torch.int) + text.shape[-1] 

        # speech         = speech.to('cuda')
        # speech_lengths = speech_lengths.to('cuda')
        # text           = text.to('cuda')
        # text_lengths   = text_lengths.to('cuda')
        speech2text.asr_model.bprocessor.maxlen = 10
        model_forward(
            speech2text.asr_model,
            speech,
            speech_lengths, 
            text, 
            text_lengths,
            keys[0],
            ref_texts[0],
            key2cluster[keys[0]],
            enc_centers,
            bpemodel
        )

        # if count > 10:
        #     break
        # count += 1
        break