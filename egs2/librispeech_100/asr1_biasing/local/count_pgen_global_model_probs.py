import os
import sentencepiece as spm

from local.utils import read_file
from local.utils import read_json
from local.utils import write_json

from local.aligner import CheatDetector

def check_right(ref, hyp, gt_pgens, pgens):
    ref = ref.split(' ')
    hyp = hyp.split(' ')

    TP, FP, TN, FN = 0, 0, 0, 0
    length = min([len(ref), len(hyp)])
    for i in range(length):
        print(f'ref    : {ref[i]}')
        print(f'hyp    : {hyp[i]}')
        print(f'gt_pgen: {gt_pgens[i]}')
        print(f'pgen   : {pgens[i]}')
        print()
            
        if ref[i] == hyp[i]:
            if pgens[i] == 0 or (pgens[i] == 1 and gt_pgens[i] == 1):
                # true positive
                TP += 1
            else:
                # false negative
                FN += 1
        else:
            if pgens[i] == 0 and gt_pgens[i] == 0:
                # true negative
                TN += 1
            elif pgens[i] == 1 and gt_pgens[i] == 0:
                # false negative
                FN += 1
            elif pgens[i] == 0 and gt_pgens[i] == 1:
                # false positive
                FP += 1
            elif pgens[i] == 1 and gt_pgens[i] == 1:
                # true negative
                TN += 1
    # TP /= length
    # FN /= length
    # FP /= length
    # TN /= length
    return TP, FN, FP, TN

if __name__ == '__main__':

    PATH       = "/share/nas165/amian/experiments/speech/tcpgen/espnet/egs2/librispeech_100/asr1_biasing/exp/asr_train_conformer_transducer_tcpgen500_deep_sche30_GCN6L_rep_suffix/decode_asr_test_asr_model_valid.loss.ave_10best/test_clean/logdir/asr_inference.json"
    # PATH       = "./exp/espnet/guangzhisun_librispeech100_asr_train_conformer_transducer_tcpgen500_deep_sche30_GCN6L_rep_suffix/decode_asr_test_asr_model_valid.loss.ave/test_clean/logdir/asr_inference.json"
    SPM_PATH   = "./data/en_token_list/bpe_unigram600suffix/bpe.model"
    TOKEN_PATH = "./data/en_token_list/bpe_unigram600suffix/tokens.txt"

    tokenizer  = spm.SentencePieceProcessor(model_file=SPM_PATH)
    token_list = [''] + [d[0] for d in read_file(TOKEN_PATH)]

    datas = read_json(PATH)

    avg_tp, avg_fn, avg_fp, avg_tn = 0, 0, 0, 0
    count = 0
    for data in datas:
        blist      = data['blist']
        hyp_tokens = data['hyp_tokens']
        ref        = data['ref']
        pgens      = data['pgens']

        ref_tokens = "_".join([token_list[t] for t in tokenizer.encode(ref)])
        hyp_tokens = "_".join(hyp_tokens)
        spm_blist  = ["_".join([token_list[t] for t in tokenizer.encode(b)]) for b in blist]
        
        # pgens = [1 if (1 - pgen) >= 0.5 else 0 for pgen in pgens]
        pgens = [1 if (pgen) >= 0.5 else 0 for pgen in pgens]

        detector   = CheatDetector(spm_blist)
        prediction = detector.predict_one_step(ref_tokens, hyp_tokens)

        if len(prediction) < 1:
            continue

        utt_tp, utt_fn, utt_fp, utt_tn = 0, 0, 0, 0
        gt_pgen = [0 for _ in range(len(pgens))]
        for pred in prediction:
            hyp_ent, pos, biasword = pred
            start, end = pos
            for i in range(start, end):
                gt_pgen[i] = 1
        
        hyp_tokens  = hyp_tokens.replace("_", " ") 
        ref_tokens  = ref_tokens.replace("_", " ") 

        print(f'ref_tokens: {ref_tokens}')
        print(f'hyp_tokens: {hyp_tokens}')
        print(f'gt_pgen   : {gt_pgen}')
        print(f'pgen      : {pgens}')

        TP, FN, FP, TN = check_right(ref_tokens, hyp_tokens, gt_pgen, pgens)
        print(f'TP: {TP:.2f}, FN: {FN:.2f}, FP: {FP:.2f}, TN: {TN:.2f}')

        print()
        avg_tp += TP
        avg_fn += FN
        avg_fp += FP
        avg_tn += TN
        count += 1
        print('_' * 30)
    # avg_tp = avg_tp / count
    # avg_fn = avg_fn / count
    # avg_fp = avg_fp / count
    # avg_tn = avg_tn / count

    recall    = (avg_tp) / (avg_tp + avg_fn)
    precision = (avg_tp) / (avg_tp + avg_fp)
    
    print(f'AVG      : TP: {avg_tp}, FN: {avg_fn}, FP: {avg_fp}, TN: {avg_tn}')
    print(f'Recall   : {recall:.2f}')
    print(f'Precision: {precision:.2f}')