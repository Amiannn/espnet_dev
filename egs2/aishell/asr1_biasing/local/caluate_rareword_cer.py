import os
import jieba
import json

from jiwer import cer
from tqdm  import tqdm

from local.aligner import CheatDetector
from local.aligner import align_to_index

from local.utils import read_file
from local.utils import read_json
from local.utils import write_file
from local.utils import write_json

rareword_list  = './local/rareword.all.txt'
# rareword_list  = './local/all_rare_words.txt'
utt_blist_path = './data/zh_test/perutt_blist.json'
# utt_blist_path = './data/zh_test/perutt_blist.json'
ref_path       = './data/zh_test/text'
# hyp_path       = './exp/asr_train_asr_transducer_conformer_e15_linear1024_raw_zh_bpe4500_use_wandbtrue_sp_suffix/decode_asr_asr_model_valid.loss.ave_10best/zh_test/text'
# hyp_path       = './exp/asr_finetune_freeze_conformer_transducer_tcpgen500_deep_sche30_rep_zh_suffix/decode_asr_asr_model_valid.loss.best/zh_test/text'
hyp_path       = './exp/asr_finetune_freeze_conformer_transducer_contextual_biasing_proj_zh_suffix/decode_asr_asr_model_valid.loss.ave_10best/zh_test/text'

def check_passed(indexis, memory):
    for index in indexis:
        if index in memory:
            return True
    return False

if __name__ == '__main__':
    uttblist = read_json(utt_blist_path)
    rareword = [d[0] for d in read_file(rareword_list, sp=' ')]

    hyps = [[d[0], [i for i in d[1:] if i != '']] for d in read_file(hyp_path, sp=' ')]
    refs = [[d[0], d[1:]] for d in read_file(ref_path, sp=' ')]
    
    ref_rareword_sents = []
    hyp_rareword_sents = []
    ref_common_sents   = []
    hyp_common_sents   = []
    ref_sents          = []
    hyp_sents          = []
    for ref, hyp in zip(refs, hyps):
        blist = uttblist[ref[0]]
        print(f'blist : {blist}')
        print(f'ref: {ref[1]}')
        print(f'hyp: {hyp[1]}')
        chunks = align_to_index(ref[1], hyp[1])
        print(f'chunks: {chunks}')
        ref_sents.append(''.join(ref[1]))
        hyp_sents.append(''.join(hyp[1]))
        ref_rareword_sent = []
        hyp_rareword_sent = []
        ref_common_sent   = []
        hyp_common_sent   = []
        passed_index      = []
        for chunk in chunks:
            wref, whyps, rindex, hindexis = chunk
            wref = wref.replace('-', '')
            whyps = ''.join(whyps).replace('-', '')
            if wref in blist:
                ref_rareword_sent.append(wref)
                hyp_rareword_sent.append(whyps)
            elif not check_passed(hindexis, passed_index):
                ref_common_sent.append(wref)
                hyp_common_sent.append(whyps)
                passed_index.extend(hindexis)
        if len(ref_rareword_sent) > 0:
            ref_rareword_sents.append(''.join(ref_rareword_sent))
            hyp_rareword_sents.append(''.join(hyp_rareword_sent))
        if len(ref_common_sent) > 0:
            ref_common_sents.append(''.join(ref_common_sent))
            hyp_common_sents.append(''.join(hyp_common_sent))

    all_cer      = cer(ref_sents, hyp_sents)
    rareword_cer = cer(ref_rareword_sents, hyp_rareword_sents)
    common_cer   = cer(ref_common_sents, hyp_common_sents)

    print(f'all_cer     : {all_cer}')
    print(f'rareword_cer: {rareword_cer}')
    print(f'common_cer  : {common_cer}')

    # output_path = './ref_common_sents'
    # write_file(output_path, [[d] for d in ref_common_sents], sp='')
    # output_path = './hyp_common_sents'
    # write_file(output_path, [[d] for d in hyp_common_sents], sp='')

