import os
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

from espnet2.asr.ctc            import CTC
from espnet2.tasks.asr          import ASRTask

from espnet2.text.build_tokenizer     import build_tokenizer
from espnet2.text.token_id_converter  import TokenIDConverter
from espnet2.utils.get_default_kwargs import get_default_kwargs

def load_espnet_model(model_conf, token_path, stats_path, spm_path, model_path):
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
    bpemodel  = spm.SentencePieceProcessor(model_file=spm_path)
    tokenizer = build_tokenizer(token_type="bpe", bpemodel=spm_path)
    converter = TokenIDConverter(token_list=args.token_list)
    
    model = ASRTask.build_model(args)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model, bpemodel, tokenizer, converter