import os
import faiss
import torch
import argparse
import numpy as np
import torchaudio
import sentencepiece as spm

from local.utils import read_file
from local.utils import read_pickle
from local.utils import write_json
from local.utils import write_file
from local.utils import read_yml
from local.utils import write_pickle

from tqdm import tqdm
from sklearn.cluster import KMeans

from typing import (
    Any, 
    Dict, 
    List, 
    Optional, 
    Sequence, 
    Tuple, 
    Union
)

from espnet2.asr.ctc import CTC
from espnet2.asr.transducer.beam_search_transducer import Hypothesis
from espnet2.asr.transducer.beam_search_transducer import Hypothesis as TransHypothesis

from espnet2.tasks.asr          import ASRTask
from espnet2.train.preprocessor import CommonPreprocessor

from espnet2.text.build_tokenizer     import build_tokenizer
from espnet2.text.token_id_converter  import TokenIDConverter
from espnet2.utils.get_default_kwargs import get_default_kwargs

def greedy_search(asr_model, enc_out):
    dec_state = asr_model.decoder.init_state(1)

    hyp = Hypothesis(score=0.0, yseq=[asr_model.blank_id], dec_state=dec_state)
    cache = {}

    dec_out, state, _ = asr_model.decoder.score(hyp, cache)
    for enc_out_t in enc_out:
        logp = torch.log_softmax(
            asr_model.joint_network(enc_out_t, dec_out),
            dim=-1,
        )
        top_logp, pred = torch.max(logp, dim=-1)

        if pred != asr_model.blank_id:
            hyp.yseq.append(int(pred))
            hyp.score += float(top_logp)
            hyp.dec_state = state
            dec_out, state, _ = asr_model.decoder.score(hyp, cache)
    return [hyp]

def decode_single_sample(asr_model, tokenizer, converter, enc, nbest):
    nbest_hyps = greedy_search(asr_model, enc)
    nbest_hyps = nbest_hyps[: nbest]

    results = []
    for hyp in nbest_hyps:
        assert isinstance(hyp, (Hypothesis, TransHypothesis)), type(hyp)

        # remove sos/eos and get results
        last_pos = None if asr_model.use_transducer_decoder else -1
        if isinstance(hyp.yseq, list):
            token_int = hyp.yseq[1:last_pos]
        else:
            token_int = hyp.yseq[1:last_pos].tolist()

        # remove blank symbol id, which is assumed to be 0
        token_int = list(filter(lambda x: x != 0, token_int))

        # Change integer-ids to tokens
        token = converter.ids2tokens(token_int)

        if tokenizer is not None:
            text = tokenizer.tokens2text(token)
        else:
            text = None
        results.append((text, token, token_int, hyp))
    return results

@torch.no_grad()
def forward(
    asr_model,
    tokenizer,
    converter,
    speech, 
    device
):
    """Inference

    Args:
        data: Input speech data
    Returns:
        text, token, token_int, hyp

    """
    # data: (Nsamples,) -> (1, Nsamples)
    speech = speech.unsqueeze(0)
    # lengths: (1,)
    lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
    batch = {"speech": speech, "speech_lengths": lengths}
    print("speech length: " + str(speech.size(1)))

    # b. Forward Encoder
    enc, enc_olens = asr_model.encode(**batch)
    
    # Normal ASR
    intermediate_outs = None
    if isinstance(enc, tuple):
        intermediate_outs = enc[1]
        enc = enc[0]
    assert len(enc) == 1, len(enc)

    # c. Passed the encoder result and the beam search
    results = decode_single_sample(
        asr_model, tokenizer, converter, enc[0], nbest=1,
    )
    return results

def single(candiates):
    result = []
    for can in candiates:
        if can not in result:
            result.append(can)
    return result

if __name__ == "__main__":
    spm_path   = "./data/en_token_list/bpe_unigram600suffix/bpe.model"
    token_path = "./data/en_token_list/bpe_unigram600suffix/tokens.txt"
    model_conf = "./conf/tuning/train_rnnt_freeze_contextual_biasing.yaml"
    model_path = "./exp/asr_finetune_freeze_conformer_transducer_contextual_biasing_proj_suffix/valid.loss.ave_10best.pth"
    stats_path = "./exp/asr_stats_raw_en_bpe600_spsuffix/train/feats_stats.npz"
    rare_path  = "./local/rareword_f15.txt"
    audio_root = "/share/corpus/LibriSpeech/LibriSpeech/test-clean/1089/134686"

    audio_filenames = [
        '1089-134686-0000.flac',
        '1089-134686-0001.flac',
        '1089-134686-0002.flac',
        '1089-134686-0003.flac',
    ]

    conf = read_yml(model_conf)
    conf['token_list']     = token_path
    conf['input_size']     = None
    conf['specaug']        = None
    conf['normalize']      = 'global_mvn'
    conf['frontend']       = 'default'
    conf['ctc_conf']       = get_default_kwargs(CTC)
    conf['init']           = None
    conf['normalize_conf'] = {
        'stats_file': stats_path
    }

    args = argparse.Namespace(**conf)
    args.model_conf['bpemodel'] = spm_path
    print(args.model_conf)
    bpemodel  = spm.SentencePieceProcessor(model_file=spm_path)
    tokenizer = build_tokenizer(token_type="bpe", bpemodel=spm_path)
    converter = TokenIDConverter(token_list=args.token_list)
    
    model = ASRTask.build_model(args)
    model.load_state_dict(torch.load(model_path))  
    
    encs = []
    enc_lengths = []
    for filename in audio_filenames:
        audio_path = os.path.join(audio_root, filename)
        speech, sample_rate = torchaudio.load(audio_path)
        speech  = speech.reshape(1, -1)
        
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        batch   = {"speech": speech, "speech_lengths": lengths}
        print("speech length: " + str(speech.size(1)))

        # b. Forward Encoder
        enc, enc_olens = model.encode(**batch)
        enc = model.Qproj_acoustic(enc)
        enc = enc.squeeze(0)
        encs.append(enc)
        enc_lengths.append(enc.shape[0])
    encs = torch.cat(encs, dim=0).detach()
    enc_lengths = torch.tensor(enc_lengths)

    embed_matrix = torch.cat(
        [model.decoder.embed.weight.data, model.ooKBemb.weight], dim=0
    )

    enc_k = 50
    kmeans = faiss.Kmeans(
        encs.shape[-1], 
        enc_k, 
        niter=20, 
        verbose=True,
        gpu=True
    )
    kmeans.train(encs)
    enc_centers = kmeans.centroids
    print(f'centers: {enc_centers.shape}')


    enc_labels = kmeans.index.search(x=encs, k=1)[1].reshape(-1)
    last = 0
    for i in range(enc_lengths.shape[0]):
        now = last + enc_lengths[i]
        print(f'{i}: {enc_labels[last:now]}')
        print(f'_' * 30)
        last = now

    bwords    = [d[0] for d in read_file(rare_path, sp=' ')]
    bsubwords = [torch.tensor(bpemodel.encode(bword)).unsqueeze(0) for bword in bwords]
    b_embeds  = []

    embed_path = './local/bword_embeds.pickle'
    if os.path.isfile(embed_path):
        b_embeds = read_pickle(embed_path)
    else:
        model.bprocessor.maxlen = len(bwords)
        (
            biasingwords, 
            lextree, 
            cb_tokens, 
            cb_tokens_len,
            worddict
        ) = model.bprocessor.select_biasing_words(
            [], 
            cb=True,
            ret_worddict=True
        )
        # for bsubword in tqdm(bsubwords):
        #     b_seq_embed, _ = model.CbRNN(embed_matrix[bsubword])
        #     b_embed = torch.mean(b_seq_embed, dim=1)
        #     b_embed = model.Kproj(b_embed)
        #     b_embeds.append(b_embed)
        # b_embeds = torch.stack(b_embeds, dim=0).squeeze(1).detach()
        # print(b_embeds.shape)
        cb_tokens = cb_tokens[:-1, :]
        cb_tokens_embed = embed_matrix[cb_tokens]
        cb_seq_embed, _ = model.CbRNN(cb_tokens_embed)
        cb_embed = torch.mean(cb_seq_embed, dim=1)
        b_embeds = model.Kproj(cb_embed)
        b_embeds = b_embeds.detach()
        write_pickle(embed_path, b_embeds)

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
    
    label2bwords = {}
    for label, bword in zip(labels, bwords):
        label = str(label)
        label2bwords[label] = label2bwords[label] + [bword] if label in label2bwords else [bword]

    for label in label2bwords:
        label2bwords[label] = sorted(label2bwords[label])
        # print(f'{label}: {len(label2bwords[label])}')
    output_path = f'./local/cluster_embeds_{k}.json'
    write_json(output_path, label2bwords)

    _, topk = kmeans.index.search(enc_centers, 1)
    topk = topk.squeeze(-1)
    print(f'topk: {topk}')
    print(f'topk: {topk.shape}')

    last = 0
    results = {}
    for i in range(enc_lengths.shape[0]):
        now = last + enc_lengths[i]
        print(f'({i})')
        label = enc_labels[last:now]
        result = []
        for j in range(label.shape[0]):
            out = f'{label[j]} -> {topk[label[j]]}: {", ".join(label2bwords[str(topk[label[j]])])}'
            result.append({j: out})
            print(out)
        results[i] = result
        print(f'_' * 30)
        last = now

    output_path = f'local/sampling_{len(audio_filenames)}_{enc_k}_{k}.json'
    write_json(output_path, results)

    # print(f'cb_tokens_len: {cb_tokens_len}')
    # print(f'cb_tokens: {cb_tokens.shape}')
    # print(f'cb_tokens_len: {cb_tokens_len.shape}')
    # topk = single(topk.tolist())
    # print(topk)
    # print(len(topk))

    # indexis = torch.randint(len(topk), (10,))
    # sampled = [topk[i] for i in indexis]
    # print(f'sampled: {sampled}')


    # aco_kmeans = KMeans(n_clusters=10, random_state=0, n_init="auto").fit(
    #     enc.detach().numpy()
    # )
    # print(aco_kmeans.labels_)