import os
import numpy as np
import sentencepiece as spm

from jiwer import wer
from tqdm  import tqdm

from local.utils import read_file
from local.utils import read_json
from local.utils import write_json
from local.utils import write_file

from local.aligner import align_to_index
from local.aligner import CheatDetector

PATH       = "/share/nas165/amian/experiments/speech/tcpgen/espnet/egs2/librispeech_100/asr1_biasing/exp/asr_train_conformer_transducer_tcpgen500_deep_sche30_GCN6L_rep_suffix/decode_asr_test_asr_model_valid.loss.ave_10best/test_clean/logdir/asr_inference.json"
TOKEN_PATH = "./data/en_token_list/bpe_unigram600suffix/tokens.txt"
SPM_PATH   = "./data/en_token_list/bpe_unigram600suffix/bpe.model"

if __name__ == '__main__':
    datas = read_json(PATH)
    token_list     = [d[0] for d in read_file(TOKEN_PATH)]
    ref_token_list = [''] + [d[0] for d in read_file(TOKEN_PATH)]
    tcp_token_list = [d[0] for d in read_file(TOKEN_PATH)] + ['null']

    tokenizer  = spm.SentencePieceProcessor(model_file=SPM_PATH)

    results = []
    title   = ['index', 'ref', 'hyp', 'asr', 'tcpgen', 'gate', 'conf', 'pgen', 'conf', 'debug']
    title   = title + [f'top{k + 1}' for k in range(10)]
    # print("\t".join(title))
    results.append(title)

    topks      = {k: 0 for k in [1, 3, 5, 10, 50, 100]}
    topk_hyps  = {k: [] for k in topks}
    data_count = 0
    refs       = []
    uttids     = []
    for data in tqdm(datas):
        model_token_ids  = data['model_tokens']
        tcpgen_token_ids = data['tcpgen_tokens']
        ref              = data['ref']
        hyp_tokens       = data['hyp_tokens']
        confidence       = data['model_probs']
        p_confidence     = data['tcpgen_probs']
        pgens            = data['pgens']
        uttid            = data['idx']
        blist            = data['blist']
        # print(ref)
        # print(blist)
        topk_tcpgen_token_ids = data['topk_tcpgen_tokens'][0]
        topk_tcpgen_probs     = data['topk_tcpgen_probs'][0]

        refs.append(ref)
        uttids.append(uttid)

        model_tokens  = [token_list[d] for d in model_token_ids]
        tcpgen_tokens = [tcp_token_list[d] for d in tcpgen_token_ids]
        ref_tokens    = [ref_token_list[t] for t in tokenizer.encode(ref)]
        spm_blist     = ["_".join([ref_token_list[t] for t in tokenizer.encode(b)]) for b in blist]
        # print("_".join(ref_tokens))
        detector   = CheatDetector(spm_blist)
        prediction = detector.predict_one_step("_".join(ref_tokens), "_".join(hyp_tokens))

        gt_pgen = [0 for _ in range(len(hyp_tokens))]
        for _, pos, _ in prediction:
            start, end = pos
            gt_pgen[start:end] = [1 for _ in range(end - start)]

        chunks = align_to_index(ref_tokens, hyp_tokens)
        # print(f'_' * 50)
        # print(f'idx: {uttid}')
        results.append([uttid])
        # print(f'_' * 50)
        
        topk     = {k: 0 for k in [1, 3, 5, 10, 50, 100]}
        topk_hyp = {k: [hyp_tokens[t] for t in range(len(hyp_tokens))] for k in topk}
        count = 0
        for i in range(len(chunks)):
            ref_idx = chunks[i][-2][0] if len(chunks[i][-2]) > 0 else -1
            hyp_idx = chunks[i][-1][0] if len(chunks[i][-1]) > 0 else -1

            result = [
                i,
                ref_tokens[ref_idx]    if ref_idx != -1 else '-',
                hyp_tokens[hyp_idx]    if hyp_idx != -1 else '-',
                model_tokens[hyp_idx]  if hyp_idx != -1 else '-',
                tcpgen_tokens[hyp_idx] if hyp_idx != -1 else '-',
                gt_pgen[hyp_idx]       if hyp_idx != -1 else '-',
                confidence[hyp_idx]    if hyp_idx != -1 else '-',
                pgens[hyp_idx]         if hyp_idx != -1 else '-',
                p_confidence[hyp_idx]  if hyp_idx != -1 else '-'
            ]
            tcpgen_topk_tokens = []
            if hyp_idx != -1:
                for j in range(len(topk_tcpgen_token_ids[hyp_idx])):
                    token = tcp_token_list[topk_tcpgen_token_ids[hyp_idx][j]]
                    tcpgen_topk_tokens.append(token)
                    result.append(token)
            else:
                result.extend(['-' for _ in range(10)])
            result = [str(d) for d in result]
            # st = "\t".join(result)
            # print(f'{st}')
            results.append(result)

            if hyp_idx != -1:
                if pgens[hyp_idx] >= 0.5:
                # if gt_pgen[hyp_idx] >= 0.5:
                # if confidence[hyp_idx] < 0.5:
                    r_token = ref_tokens[ref_idx]
                    h_token = hyp_tokens[hyp_idx]
                    if r_token != h_token:
                        hit = False
                        for k in topk:
                            if r_token in tcpgen_topk_tokens[:k]:
                                topk[k] += 1
                                hit = True
                        count += 1
            
            for k in topk_hyp:
                r_token = ref_tokens[ref_idx]
                if hyp_idx != -1:
                    # if pgens[hyp_idx] >= 0.5:
                    #     topk_hyp[k][hyp_idx] = tcpgen_tokens[hyp_idx]
                    # if confidence[hyp_idx] < 0.5:
                    #     topk_hyp[k][hyp_idx] = tcpgen_tokens[hyp_idx]
                    # if gt_pgen[hyp_idx] >= 0.5 and tcpgen_tokens[hyp_idx] != 'null':
                    if gt_pgen[hyp_idx] >= 0.5:
                        if r_token in tcpgen_topk_tokens[:k]:
                            topk_hyp[k][hyp_idx] = r_token
                        # topk_hyp[k][hyp_idx] = tcpgen_tokens[hyp_idx]

        for k in topk_hyps:
            # print(topk_hyp[k])
            text = ''.join(topk_hyp[k]).replace('▁', ' ')
            topk_hyps[k].append(text)

        if count > 0:
            data_count += 1
            for k in topks:
                topks[k] += topk[k] / count
        # print('=' * 50)
        # print()
    output_path = 'atten_debug_v2.tsv'
    write_file(output_path, results[:5000], sp='\t')

    for k in topks:
        topks[k] /= data_count
    print(topks)

    for k in topk_hyps:
        reference  = refs
        hypothesis = topk_hyps[k]
        error = wer(reference, hypothesis)
        print(f'top {k} wer: {error:.4f}')

        hypothesis = [[hyp, f'({uttid})'] for hyp, uttid in zip(hypothesis, uttids)]
        output_path = f'./atten/top{k}.hyp.trn'
        write_file(output_path, hypothesis, sp='\t')

    reference = [[ref, f'({uttid})'] for ref, uttid in zip(refs, uttids)]
    output_path = f'./atten/top{k}.ref.trn'
    write_file(output_path, reference, sp='\t')