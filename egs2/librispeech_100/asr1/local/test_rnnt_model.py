import os
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

class CustomBeamSearchTransducer(BeamSearchTransducer):
    def greedy_search(self, enc_out: torch.Tensor):
        print('Custom Greedy decoding!')
        dec_state = self.decoder.init_state(1)

        hyp = Hypothesis(score=0.0, yseq=[self.blank_id], dec_state=dec_state)
        cache = {}

        dec_out, state, _ = self.decoder.score(hyp, cache)
        topk_logps = []
        topk_ids   = []
        for enc_out_t in enc_out:
            logp = torch.log_softmax(
                self.joint_network(enc_out_t, dec_out),
                dim=-1,
            )
            top_logp, pred = torch.max(logp, dim=-1)

            if pred != self.blank_id:
                _topk_logp, _topk_ids = torch.topk(logp, 50)
                hyp.yseq.append(int(pred))
                hyp.score += float(top_logp)

                hyp.dec_state = state
                topk_logps.append(_topk_logp)
                topk_ids.append(_topk_ids)
                dec_out, state, _ = self.decoder.score(hyp, cache)

        hyp.topk_logp = torch.stack(topk_logps)
        hyp.topk_ids  = torch.stack(topk_ids)
        return [hyp]

class CustomSpeech2Text(Speech2Text):
    def _decode_single_sample(self, enc: torch.Tensor, lextree: list = None):
        if self.beam_search_transducer:
            nbest_hyps = self.beam_search_transducer(enc, lextree=lextree)
            best = nbest_hyps[0]
        nbest_hyps = nbest_hyps[: self.nbest]

        results = []
        for hyp in nbest_hyps:
            assert isinstance(hyp, (Hypothesis, TransHypothesis)), type(hyp)

            # remove sos/eos and get results
            last_pos = None if self.asr_model.use_transducer_decoder else -1
            if isinstance(hyp.yseq, list):
                token_int = hyp.yseq[1:last_pos]
            else:
                token_int = hyp.yseq[1:last_pos].tolist()

            # remove blank symbol id, which is assumed to be 0
            token_int = list(filter(lambda x: x != 0, token_int))

            # Change integer-ids to tokens
            token = self.converter.ids2tokens(token_int)

            if self.tokenizer is not None:
                text = self.tokenizer.tokens2text(token)
            else:
                text = None
            results.append((text, token, token_int, hyp))

        return results

config = {
    "log_level": "INFO",
    "output_dir": "exp/asr_train_asr_transducer_conformer_e15_linear1024_raw_en_bpe600_use_wandbtrue_sp_suffix/decode_asr_test_asr_model_valid.loss.ave_10best/test_clean/logdir/output.1",
    "ngpu": 1,
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
    "key_file": "exp/asr_train_asr_transducer_conformer_e15_linear1024_raw_en_bpe600_use_wandbtrue_sp_suffix/decode_asr_test_asr_model_valid.loss.ave_10best/test_clean/logdir/keys.1.scp",
    "allow_variable_data_keys": False,
    "asr_train_config": "exp/asr_train_asr_transducer_conformer_e15_linear1024_raw_en_bpe600_use_wandbtrue_sp_suffix/config.yaml",
    "asr_model_file": "exp/asr_train_asr_transducer_conformer_e15_linear1024_raw_en_bpe600_use_wandbtrue_sp_suffix/valid.loss.ave_10best.pth",
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
    "beam_size": 1,
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
    "perutt_blist": "",
    "biasinglist": "",
    "bmaxlen": 100,
    "bdrop": 0.0
}

def model_forward(
        model,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
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

        # 2a. Transducer decoder branch
        decoder_in, target, t_len, u_len = get_transducer_task_io(
            text,
            encoder_out_lens,
            ignore_id=-1,
            blank_id=model.blank_id,
        )
        print(decoder_in)
        # print(target)
        # print(target.shape)
        model.decoder.set_device(encoder_out.device)
        decoder_out = model.decoder(decoder_in)

        logp = torch.log_softmax(
            model.joint_network(encoder_out.unsqueeze(2), decoder_out.unsqueeze(1)),
            dim=-1,
        )[0]

        logp   = logp.to('cpu').transpose(1, 0)
        target = target.to('cpu')[0]

        print(f'joint out dim: {logp.shape}')

        forward_backward(logp, target, model.blank_id, model.token_list, speech.to('cpu')[0])
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

import matplotlib.pyplot as plt
import textgrid
import math  

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
    plt.savefig('./alpha.pdf', format="pdf", bbox_inches="tight")
    plt.clf()

    beta = torch.zeros(u_len, t_len)
    for u in range(u_len - 1, -1, -1):
        for t in range(t_len - 1, -1, -1):
            if u == (u_len - 1) and t == (t_len - 1): continue
            beta_y_partial   = beta[u + 1, t] + y_logp[u, t] if (u + 1) < u_len else inf_tensor
            beta_phi_partial = beta[u, t + 1] + phi_logp[u, t] if (t + 1) < t_len else inf_tensor
            beta[u, t] = torch.logaddexp(beta_y_partial, beta_phi_partial)

    plt.imshow(beta, origin="lower")
    plt.savefig('./beta.pdf', format="pdf", bbox_inches="tight")
    plt.clf()

    ab_log_prob = (alpha + beta)
    ab_prob     = torch.exp(ab_log_prob)

    plt.imshow(ab_log_prob, origin="lower")
    plt.savefig('./ab_log_prob.pdf', format="pdf", bbox_inches="tight")
    plt.clf()

    plt.imshow(ab_prob, origin="lower")
    plt.savefig('./ab.pdf', format="pdf", bbox_inches="tight")
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
    print(align_paths)
    for i, align_path in enumerate(align_paths):
        token = tokens[i]
        if len(align_path) == 0:
            continue
        start, end = min(align_path), max(align_path)
        if start == end:
            continue
        start    = ratio * start
        end      = ratio * end
        interval = textgrid.Interval(minTime=start, maxTime=end, mark=token)
        tier_word.addInterval(interval)
        start = f'{start:.2f}'
        end   = f'{end:.2f}'
        print(f'{token}\t{start}\t{end}')
        alignments.append([token, start, end])
    tg.tiers.append(tier_word)
    tg.write("1.TextGrid")

    output_path = './alignment.tsv'
    write_file(output_path, alignments, sp='\t')
    ax2.specgram(waveform, Fs=sample_rate)
    ax2.set_yticks([])
    ax2.set_xlabel("time [second]")
    fig.tight_layout()

    plt.savefig('./ab_prob.pdf', format="pdf", bbox_inches="tight")
    plt.clf()

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
    device = "cpu" if ngpu < 1 else "cuda"
    
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
    speech2text = CustomSpeech2Text.from_pretrained(
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
    speech2text.beam_search_transducer = CustomBeamSearchTransducer(
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

        speech         = speech.to('cuda')
        speech_lengths = speech_lengths.to('cuda')
        text           = text.to('cuda')
        text_lengths   = text_lengths.to('cuda')

        # model_forward(
        #     speech2text.asr_model,
        #     speech,
        #     speech_lengths, 
        #     text, 
        #     text_lengths
        # )
        
        # N-best list of (text, token, token_int, hyp_object)
        try:
            results = speech2text(**batch)
        except TooShortUttError as e:
            logging.warning(f"Utterance {keys} {e}")
            hyp = Hypothesis(score=0.0, scores={}, states={}, yseq=[])
            results = [[" ", ["<space>"], [2], hyp]] * nbest
            if enh_s2t_task:
                num_spk = getattr(speech2text.asr_model.enh_model, "num_spk", 1)
                results = [results for _ in range(num_spk)]

        # Only supporting batch_size==1
        key = keys[0]

        # Normal ASR
        encoder_interctc_res = None
        if isinstance(results, tuple):
            results, encoder_interctc_res = results

        for n, (text, token, token_int, hyp) in zip(
            range(1, nbest + 1), results
        ):
            print(f'hyp: {hyp}')
            print(f'text: {text}')
            debug_embed_results[n].append({
                'idx': key,
                'token': token,
                'token_int': token_int,
                'topk_logp': hyp.topk_logp,
                'topk_ids' : hyp.topk_ids
            })
        break