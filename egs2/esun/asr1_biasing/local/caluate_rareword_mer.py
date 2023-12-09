import os
import jieba
import json

from jiwer import cer
from jiwer import mer
from tqdm  import tqdm

from local.aligner import CheatDetector
from local.aligner import align_to_index

from local.utils import read_file
from local.utils import read_json
from local.utils import write_file
from local.utils import write_json

rareword_list  = './local/all_rare_words_new.txt'
# rareword_list  = '/share/nas165/amian/experiments/speech/preprocess/dataset_pre/others/esun_dumps/crawler/esun.entity.txt'
utt_blist_path = './data/test/perutt_blist.json'
ref_path       = './data/test/text'
hyp_path       = './exp/asr_train_asr_transducer_conformer_raw_bpe3949_use_wandbtrue_sp_suffix/decode_asr_rnnt_transducer_asr_model_valid.loss.ave_10best/test/text'
# hyp_path       = './exp/asr_train_rnnt_freeze_tcpgen_raw_bpe3949_use_wandbtrue_sp_suffix/decode_asr_asr_model_valid.loss.ave_10best/test/text'

def check_passed(indexis, memory):
    for index in indexis:
        if index in memory:
            return True
    return False

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def preprocess_sent(words):
    new_words = []
    for word in words:
        if isEnglish(word):
            new_words.append(word)
        else:
            new_words.extend(list(word))
    return new_words

def preprocess_sent_sen(words):
    new_words = []
    new_word  = ""
    for word in words:
        if isEnglish(word):
            if new_word != "":
                new_words.append(new_word)
                new_word = ""
            new_words.append(word)
        else:
            new_word += word
    if new_word != "":
        new_words.append(new_word)
    return ' '.join(new_words)

def find_rareword(sent, rarewords):
    # print(f'sent: {sent}')
    blist = []
    for word in rarewords:
        if isEnglish(word):
            if word in sent.split(' '):
                blist.append(word)
        else:
            if word in sent.replace(' ', ''):
                blist.append(word)
    return blist

if __name__ == '__main__':
    # uttblist = read_json(utt_blist_path)
    rareword = [d[0] for d in read_file(rareword_list, sp=' ')]
    print(rareword[:10])

    hyps = [[d[0], [i for i in d[1:] if i != '']] for d in read_file(hyp_path, sp=' ')]
    refs = [[d[0], d[1:]] for d in read_file(ref_path, sp=' ')]
    
    ref_rareword_sents = []
    hyp_rareword_sents = []
    ref_common_sents   = []
    hyp_common_sents   = []
    ref_sents          = []
    hyp_sents          = []
    error_pattern      = {}
    for ref, hyp in zip(refs, hyps):
        # blist = uttblist[ref[0]]
        blist     = find_rareword(" ".join(ref[1]), rareword)
        # ref_words = preprocess_sent_sen(ref[1]).split(' ')
        # hyp_words = preprocess_sent_sen(hyp[1]).split(' ')
        ref_words = ref[1]
        hyp_words = hyp[1]
        # print(f'blist : {blist}')
        # print(f'ref: {ref_words}')
        # print(f'hyp: {hyp_words}')
        chunks = align_to_index(ref_words, hyp_words)
        # print(f'chunks: {chunks}')
        ref_words = preprocess_sent(ref[1])
        hyp_words = preprocess_sent(hyp[1])
        ref_sents.append(' '.join(ref_words))
        hyp_sents.append(' '.join(hyp_words))
        ref_rareword_sent = []
        hyp_rareword_sent = []
        ref_common_sent   = []
        hyp_common_sent   = []
        passed_index      = []
        # print(f'blist: {blist}')
        # print(chunks)
        for chunk in chunks:
            wref, whyps, rindex, hindexis = chunk
            wref = wref.replace('-', '')
            whyps = ''.join(whyps).replace('-', '')
            if wref in blist:
                if wref != whyps:
                    # print(f'wrong: {wref}')
                    error_pattern[wref] = error_pattern[wref] + [whyps] if wref in error_pattern else [whyps]
                # else:
                    # print(f'right: {wref}')
                ref_rareword_sent.append(wref)
                hyp_rareword_sent.append(whyps)
            elif not check_passed(hindexis, passed_index):
                ref_common_sent.append(wref)
                hyp_common_sent.append(whyps)
                passed_index.extend(hindexis)
        # print("_" * 30)
        if len(ref_rareword_sent) > 0:
            ref_rareword_sents.append(' '.join(ref_rareword_sent))
            hyp_rareword_sents.append(' '.join(hyp_rareword_sent))
        if len(ref_common_sent) > 0:
            ref_common_sents.append(' '.join(ref_common_sent))
            hyp_common_sents.append(' '.join(hyp_common_sent))

    all_mer      = mer(ref_sents, hyp_sents)
    rareword_mer = mer(ref_rareword_sents, hyp_rareword_sents)
    common_mer   = mer(ref_common_sents, hyp_common_sents)

    print(ref_rareword_sents[:10])
    print(hyp_rareword_sents[:10])

    print(f'all_mer     : {all_mer}')
    print(f'rareword_mer: {rareword_mer}')
    print(f'common_mer  : {common_mer}')

    output_path = './ref_rareword_sents'
    write_file(output_path, [[d] for d in ref_rareword_sents], sp='')
    output_path = './hyp_rareword_sents'
    write_file(output_path, [[d] for d in hyp_rareword_sents], sp='')

    output_path = './error_pattren_asr.tsv'
    error_pattern = [[key, ", ".join([d for d in error_pattern[key] if d != ''])] for key in error_pattern]
    write_file(output_path, sorted(error_pattern), sp='\t')
