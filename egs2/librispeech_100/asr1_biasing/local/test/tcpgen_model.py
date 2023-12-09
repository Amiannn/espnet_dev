import os
import json
import torch
import logging
import numpy as np

from local.utils import read_file
from local.utils import read_pickle
from local.utils import write_json
from local.utils import write_file

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

from espnet.nets.pytorch_backend.nets_utils import th_accuracy, to_device

debug_path = './local/test/debug'

config = {
    "log_level": "INFO",
    "output_dir": "exp/asr_finetune_freeze_conformer_transducer_tcpgen500_nondeep_sche30_rep_suffix/decode_asr_asr_model_valid.loss.best/test_clean/logdir/output.1",
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
    "key_file": "exp/asr_finetune_freeze_conformer_transducer_tcpgen500_nondeep_sche30_rep_suffix/decode_asr_asr_model_valid.loss.best/test_clean/logdir/keys.1.scp",
    "allow_variable_data_keys": False,
    "asr_train_config": "exp/asr_finetune_freeze_conformer_transducer_tcpgen500_nondeep_sche30_rep_suffix/config.yaml",
    "asr_model_file": "exp/asr_finetune_freeze_conformer_transducer_tcpgen500_nondeep_sche30_rep_suffix/valid.loss.best.pth",
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
    "bmaxlen": 1000,
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

    lextrees = []
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
    lextrees = [lextree] * text.size(0)

    labels       = [key for key in model.bprocessor.chardict] + ['<OOL>']
    biasingwords = [''.join(word).replace('▁', '') for word in biasingwords]
    # print(f'worddict: {worddict}')
    # print(f'labels: {labels}')
    # print(f'biasingwords: {biasingwords}')

    ref_text = [text if text not in biasingwords else f'<{text}>' for text in ref_text.split(' ')]
    ref_text = " ".join(ref_text)

    # 2a. Transducer decoder branch
    decoder_in, target, t_len, u_len = get_transducer_task_io(
        text,
        encoder_out_lens,
        ignore_id=-1,
        blank_id=model.blank_id,
    )

    model.decoder.set_device(encoder_out.device)
    decoder_out = model.decoder(decoder_in)
    # biasing
    trees = lextrees
    p_gen_mask_all = []
    KBembedding = []
    ptr_dist = []
    node_encs = None
    # Encode prefix tree using GNN
    if model.GNN != "":
        node_encs = model.gnn(lextrees[0], model.decoder.embed)

    # Forward TCPGen
    query_acoustic = model.Qproj_acoustic(encoder_out)
    for i in range(decoder_in.size(1)):
        retval = model.get_step_biasing_embs(
            decoder_in[:, i], trees, lextrees, node_encs=node_encs
        )
        step_mask = retval[0]
        step_embs = retval[1]
        trees = retval[2]
        p_gen_mask = retval[3]
        back_transform = retval[4]
        index_list = retval[5]

        p_gen_mask_all.append(p_gen_mask)
        query_char = model.decoder.dropout_embed(
            model.decoder.embed(decoder_in[:, i])
        )
        query_char = model.Qproj_char(query_char).unsqueeze(1)
        query = query_char + query_acoustic  # nutts * T * attn_dim
        hptr_i, tcpgen_dist_i = model.get_meetingKB_emb_map(
            query, step_mask, back_transform, index_list, meeting_KB=step_embs
        )
        ptr_dist.append(tcpgen_dist_i.unsqueeze(2))
        KBembedding.append(hptr_i.unsqueeze(2))

    KBembedding = torch.cat(KBembedding, dim=2)
    ptr_dist = torch.cat(ptr_dist, dim=2)
    
    joint_out, joint_acts = model.joint_network(
        encoder_out.unsqueeze(2),
        decoder_out.unsqueeze(1),
        KBembedding,
    )
    # biasing
    p_gen = torch.sigmoid(
        model.pointer_gate(torch.cat((joint_acts, KBembedding), dim=-1))
    )
    ptr_mask = to_device(model, torch.tensor(p_gen_mask_all)).t()
    p_gen = p_gen.masked_fill(ptr_mask.unsqueeze(1).unsqueeze(-1).bool(), 0)
    # Get factorised loss
    model_dist = torch.softmax(joint_out, dim=-1)
    p_not_null = 1.0 - model_dist[:, :, :, 0:1]
    ptr_dist_fact = ptr_dist[:, :, :, 1:] * p_not_null
    ptr_gen_complement = (ptr_dist[:, :, :, -1:]) * p_gen
    p_partial = ptr_dist_fact[:, :, :, :-1] * p_gen + model_dist[
        :, :, :, 1:
    ] * (1 - p_gen + ptr_gen_complement)
    p_final = torch.cat([model_dist[:, :, :, 0:1], p_partial], dim=-1)
    joint_out = torch.log(p_final + 1e-12)

    joint_out = torch.log_softmax(joint_out, dim=-1).squeeze(0)
    print(f'joint_out.shape: {joint_out.shape}')
    logp   = joint_out.to('cpu').transpose(1, 0)
    target = target.to('cpu')[0]

    alignments = forward_backward(
        logp, 
        target, 
        model.blank_id, 
        model.token_list, 
        speech.to('cpu')[0]
    )
    # draw attention map
    # dist = torch.zeros(ptr_dist.shape[3], ptr_dist.shape[1])
    # print(f'dist: {dist.shape}')
    # index = 0
    # for token, start, end in alignments:
    #     print(f'token: {token}, index: {index}')
    #     for i in range(start, end):
    #         print(i)
    #         dist[:, (i - start)] = ptr_dist[0, i, index, :]
    #     index += 1
    dist = torch.mean(ptr_dist.squeeze(0), dim=1)
    dist = torch.softmax(dist, dim=-1)
    dist = dist.T.detach().cpu().resolve_conj().resolve_neg().numpy()
    print(f'dist: {dist.shape}')

    frame2align = {start: token for token, start, end in alignments}
    xlabels = [frame2align[i] if i in frame2align else '' for i in range(dist.shape[1])]
    print(f'xlabels: {len(xlabels)}')

    fig, axes = plt.subplots(1, 1, figsize=(60, 150))
    axes.xaxis.set_ticks(np.arange(0, dist.shape[1], 1))
    axes.yaxis.set_ticks(np.arange(0, dist.shape[0], 1))
    axes.set_xticks(np.arange(-.5, dist.shape[1], 10), minor=True)
    axes.set_yticks(np.arange(-.5, dist.shape[0], 1), minor=True)
    axes.set_xticklabels(xlabels)
    axes.set_yticklabels(labels)

    axes.imshow(dist, aspect='auto')
    # axes.grid(which='minor', color='w', linewidth=0.5)
    axes.grid(which='minor', linewidth=0.1)
    plt.title(ref_text)
    output_path = os.path.join(debug_path, f'{uttid}_dist.pdf')
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
        print(f'{token}\t{start}\t{end}')
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

if __name__ == "__main__":
    ref_path = "./dump/raw/test_clean/text"
    refs     = {d[0]: " ".join(d[1:]) for d in read_file(ref_path, sp=' ')}

    speech2text, loader, preprocess = inference(**config)
    
    debug_embed_results = {n: [] for n in range(1, 11)}

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
        speech2text.asr_model.bprocessor.maxlen = 100
        model_forward(
            speech2text.asr_model,
            speech,
            speech_lengths, 
            text, 
            text_lengths,
            keys[0],
            ref_texts[0]
        )

        if count > 10:
            break
        count += 1
    