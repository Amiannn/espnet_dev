"""Search algorithms for Transducer models."""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr_transducer.joint_network import JointNetwork
from espnet2.lm.transformer_lm import TransformerLM
from espnet.nets.pytorch_backend.transducer.utils import (
    is_prefix,
    recombine_hyps,
    select_k_expansions,
    subtract,
)


@dataclass
class Hypothesis:
    """Default hypothesis definition for Transducer search algorithms."""

    score    : float
    yseq     : List[int]
    dec_state: Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        List[Optional[torch.Tensor]],
        torch.Tensor,
    ]
    lm_state : Union[Dict[str, Any], List[Any]] = None
    # biasing
    lextree         : list = None
    pgens           : List[int] = None
    topk_logp       : torch.Tensor = None
    topk_ids        : torch.Tensor = None
    top_model_prob  : List[int] = None
    top_model_pred  : List[int] = None
    top_tcpgen_prob : List[int] = None
    top_tcpgen_pred : List[int] = None
    topk_model_prob : List[int] = None
    topk_model_pred : List[int] = None
    topk_tcpgen_prob: List[int] = None
    topk_tcpgen_pred: List[int] = None


@dataclass
class ExtendedHypothesis(Hypothesis):
    """Extended hypothesis definition for NSC beam search and mAES."""

    dec_out: List[torch.Tensor] = None
    lm_scores: torch.Tensor = None


class BeamSearchTransducer:
    """Beam search implementation for Transducer."""

    def __init__(
        self,
        decoder: AbsDecoder,
        joint_network: JointNetwork,
        beam_size: int,
        lm: torch.nn.Module = None,
        lm_weight: float = 0.1,
        search_type: str = "default",
        max_sym_exp: int = 2,
        u_max: int = 50,
        nstep: int = 1,
        prefix_alpha: int = 1,
        expansion_gamma: int = 2.3,
        expansion_beta: int = 2,
        multi_blank_durations: List[int] = [],
        multi_blank_indices: List[int] = [],
        score_norm: bool = True,
        score_norm_during: bool = False,
        nbest: int = 1,
        token_list: Optional[List[str]] = None,
        biasing: bool = False,
        biasing_type: str = 'tcpgen',
        deepbiasing: bool = False,
        BiasingBundle: dict = None,
    ):
        """Initialize Transducer search module.

        Args:
            decoder: Decoder module.
            joint_network: Joint network module.
            beam_size: Beam size.
            lm: LM class.
            lm_weight: LM weight for soft fusion.
            search_type: Search algorithm to use during inference.
            max_sym_exp: Number of maximum symbol expansions at each time step. (TSD)
            u_max: Maximum output sequence length. (ALSD)
            nstep: Number of maximum expansion steps at each time step. (NSC/mAES)
            prefix_alpha: Maximum prefix length in prefix search. (NSC/mAES)
            expansion_beta:
              Number of additional candidates for expanded hypotheses selection. (mAES)
            expansion_gamma: Allowed logp difference for prune-by-value method. (mAES)
            multi_blank_durations: The duration of each blank token. (MBG)
            multi_blank_indices: The index of each blank token in token_list. (MBG)
            score_norm: Normalize final scores by length. ("default")
            score_norm_during:
              Normalize scores by length during search. (default, TSD, ALSD)
            nbest: Number of final hypothesis.

        """
        self.decoder = decoder
        self.joint_network = joint_network

        self.beam_size = beam_size
        self.hidden_size = decoder.dunits
        self.vocab_size = decoder.odim

        self.sos = self.vocab_size - 1
        self.token_list = token_list

        self.blank_id     = decoder.blank_id
        self.biasing_type = biasing_type
        logging.info(f'biasing type: {biasing_type}')
        if search_type == "mbg":
            self.beam_size = 1
            self.multi_blank_durations = multi_blank_durations
            self.multi_blank_indices = multi_blank_indices
            self.search_algorithm = self.multi_blank_greedy_search

        elif self.beam_size <= 1:
            if biasing:
                if self.biasing_type == 'tcpgen':
                    self.search_algorithm = self.TCPGen_biasing_greedy_search
                elif self.biasing_type == 'contextualbiasing':
                    # self.search_algorithm = self.ContextualBiasing_greedy_search
                    self.search_algorithm = self.ContextualBiasing_trie_greedy_search
                elif self.biasing_type == 'contextualbiasing_predictor':
                    self.search_algorithm = self.ContextualBiasingPredictor_greedy_search
            else:
                # TODO: Remember to change it back!
                self.search_algorithm = self.greedy_search
                # self.search_algorithm = self.greedy_ctc_search
        elif search_type == "default":
            if biasing:
                if self.biasing_type == 'tcpgen':
                    self.search_algorithm = self.TCPGen_biasing_beam_search
                elif self.biasing_type == 'contextualbiasing':
                    # self.search_algorithm = self.ContextualBiasing_beam_search
                    self.search_algorithm = self.ContextualBiasing_trie_beam_search
                elif self.biasing_type == 'contextualbiasing_predictor':
                    self.search_algorithm = self.ContextualBiasingPredictor_beam_search
            else:
                self.search_algorithm = self.default_beam_search
        elif search_type == "tsd":
            if isinstance(lm, TransformerLM):
                raise NotImplementedError

            self.max_sym_exp = max_sym_exp

            self.search_algorithm = self.time_sync_decoding
        elif search_type == "alsd":
            if isinstance(lm, TransformerLM):
                raise NotImplementedError

            self.u_max = u_max

            self.search_algorithm = self.align_length_sync_decoding
        elif search_type == "nsc":
            if isinstance(lm, TransformerLM):
                raise NotImplementedError

            self.nstep = nstep
            self.prefix_alpha = prefix_alpha

            self.search_algorithm = self.nsc_beam_search
        elif search_type == "maes":
            if isinstance(lm, TransformerLM):
                raise NotImplementedError

            self.nstep = nstep if nstep > 1 else 2
            self.prefix_alpha = prefix_alpha
            self.expansion_gamma = expansion_gamma

            assert self.vocab_size >= beam_size + expansion_beta, (
                "beam_size (%d) + expansion_beta (%d) "
                "should be smaller or equal to vocabulary size (%d)."
                % (beam_size, expansion_beta, self.vocab_size)
            )
            self.max_candidates = beam_size + expansion_beta

            self.search_algorithm = self.modified_adaptive_expansion_search

        else:
            raise NotImplementedError

        self.use_lm = lm is not None
        self.lm = lm
        self.lm_weight = lm_weight
        
        # biasing
        self.biasing = biasing
        self.deepbiasing = deepbiasing
        self.gnn = None
        if self.biasing and BiasingBundle is not None:
            if self.biasing_type == 'tcpgen':
                self.Qproj_acoustic = BiasingBundle["Qproj_acoustic"]
                self.Qproj_char = BiasingBundle["Qproj_char"]
                self.Kproj = BiasingBundle["Kproj"]
                self.ooKBemb = BiasingBundle["ooKBemb"]
                self.pointer_gate = BiasingBundle["pointer_gate"]
                self.gnn = BiasingBundle["gnn"]
            elif self.biasing_type == 'contextualbiasing':
                self.Qproj_acoustic = BiasingBundle["Qproj_acoustic"]
                self.Kproj = BiasingBundle["Kproj"]
                self.Vproj = BiasingBundle["Vproj"]
                self.proj  = BiasingBundle["proj"]
                self.CbRNN = BiasingBundle["CbRNN"]
                self.ooKBemb = BiasingBundle["ooKBemb"]
            elif self.biasing_type == 'contextualbiasing_predictor':
                self.Qproj_semantic = BiasingBundle["Qproj_semantic"]
                self.Kproj = BiasingBundle["Kproj"]
                self.Vproj = BiasingBundle["Vproj"]
                self.proj  = BiasingBundle["proj"]
                self.CbRNN = BiasingBundle["CbRNN"]
                self.ooKBemb = BiasingBundle["ooKBemb"]
                
        if self.use_lm and self.beam_size == 1:
            logging.warning("LM is provided but not used, since this is greedy search.")

        self.score_norm = score_norm
        self.score_norm_during = score_norm_during
        self.nbest = nbest
        self.search_type = search_type

    def __call__(
        self,
        enc_out: torch.Tensor,
        lextree: list = None,
        cb_tokens: torch.Tensor = None, 
        cb_tokens_len: torch.Tensor = None
    ) -> Union[List[Hypothesis], List[ExtendedHypothesis]]:
        """Perform beam search.

        Args:
            enc_out: Encoder output sequence. (T, D_enc)

        Returns:
            nbest_hyps: N-best decoding results

        """
        self.decoder.set_device(enc_out.device)
        
        if self.use_lm:
            self.lm.to(enc_out.device)

        if self.biasing:
            if self.biasing_type == 'tcpgen':
                nbest_hyps = self.search_algorithm(enc_out, lextree=lextree)
            elif self.biasing_type == 'contextualbiasing':
                nbest_hyps = self.search_algorithm(enc_out, cb_tokens=cb_tokens, cb_tokens_len=cb_tokens_len, lextree=lextree)
            elif self.biasing_type == 'contextualbiasing_predictor':
                nbest_hyps = self.search_algorithm(enc_out, cb_tokens=cb_tokens, cb_tokens_len=cb_tokens_len)
        else:
            nbest_hyps = self.search_algorithm(enc_out)

        return nbest_hyps

    def sort_nbest(
        self, hyps: Union[List[Hypothesis], List[ExtendedHypothesis]]
    ) -> Union[List[Hypothesis], List[ExtendedHypothesis]]:
        """Sort hypotheses by score or score given sequence length.

        Args:
            hyps: Hypothesis.

        Return:
            hyps: Sorted hypothesis.

        """
        if self.score_norm:
            hyps.sort(key=lambda x: x.score / len(x.yseq), reverse=True)
        else:
            hyps.sort(key=lambda x: x.score, reverse=True)

        return hyps[: self.nbest]

    def prefix_search(
        self, hyps: List[ExtendedHypothesis], enc_out_t: torch.Tensor
    ) -> List[ExtendedHypothesis]:
        """Prefix search for NSC and mAES strategies.

        Based on https://arxiv.org/pdf/1211.3711.pdf

        """
        for j, hyp_j in enumerate(hyps[:-1]):
            for hyp_i in hyps[(j + 1) :]:
                curr_id = len(hyp_j.yseq)
                pref_id = len(hyp_i.yseq)

                if (
                    is_prefix(hyp_j.yseq, hyp_i.yseq)
                    and (curr_id - pref_id) <= self.prefix_alpha
                ):
                    logp = torch.log_softmax(
                        self.joint_network(enc_out_t, hyp_i.dec_out[-1]),
                        dim=-1,
                    )

                    curr_score = hyp_i.score + float(logp[hyp_j.yseq[pref_id]])

                    for k in range(pref_id, (curr_id - 1)):
                        logp = torch.log_softmax(
                            self.joint_network(enc_out_t, hyp_j.dec_out[k]),
                            dim=-1,
                        )

                        curr_score += float(logp[hyp_j.yseq[k + 1]])

                    hyp_j.score = np.logaddexp(hyp_j.score, curr_score)

        return hyps

    def greedy_search(self, enc_out: torch.Tensor) -> List[Hypothesis]:
        """Greedy search implementation.

        Args:
            enc_out: Encoder output sequence. (T, D_enc)

        Returns:
            hyp: 1-best hypotheses.

        """
        logging.info('Greedy decoding!')
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

        # hyp.topk_logp = torch.stack(topk_logps)
        # hyp.topk_ids  = torch.stack(topk_ids)

        # logging.info(f'yseq length: {len(hyp.yseq)}')
        # logging.info(f'topk_logp shape: {hyp.topk_logp.shape}')
        # logging.info(f'topk_ids  shape: {hyp.topk_ids.shape}')
        return [hyp]

    def greedy_ctc_search(self, enc_out: torch.Tensor) -> List[Hypothesis]:
        """Greedy search implementation.

        Args:
            enc_out: Encoder output sequence. (T, D_enc)

        Returns:
            hyp: 1-best hypotheses.

        """
        logging.info('Greedy CTC decoding!')
        dec_state = self.decoder.init_state(1)

        hyp = Hypothesis(score=0.0, yseq=[self.blank_id], dec_state=dec_state)
        cache = {}

        dec_out, state, _ = self.decoder.score(hyp, cache)
        dec_out = torch.zeros_like(dec_out)

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
        return [hyp]

    def TCPGen_biasing_greedy_search(
        self, enc_out: torch.Tensor, lextree: list = None
    ) -> List[Hypothesis]:
        """Greedy search implementation.

        Args:
            enc_out: Encoder output sequence. (T, D_enc)

        Returns:
            hyp: 1-best hypotheses.

        """
        logging.info('Using TCPGen greedy search!')
        dec_state = self.decoder.init_state(1)

        hyp = Hypothesis(
            score=0.0, 
            yseq=[self.blank_id], 
            dec_state=dec_state, 
            lextree=lextree, 
            pgens=[0],
            top_model_prob=[0],
            top_model_pred=[0],
            top_tcpgen_prob=[0],
            top_tcpgen_pred=[0],
            topk_tcpgen_prob=[0], 
            topk_tcpgen_pred=[0],
            topk_model_prob=[0],
            topk_model_pred=[0]
        )
        cache = {}

        # Encode prefix tree using GNN
        node_encs = None
        if self.gnn is not None:
            logging.info('using GCN!')
            node_encs = self.gnn(lextree, self.decoder.embed)

        dec_out, state, _ = self.decoder.score(hyp, cache)
        p_gens = []
        for enc_out_t in enc_out:
            # biasing
            trees = [None]
            if self.biasing and lextree is not None:
                vy = hyp.yseq[-1] if len(hyp.yseq) > 1 else self.blank_id
                # print(f'vy: {vy}, token: {self.token_list[vy]}, tree len: {len(hyp.lextree)}, tree: {hyp.lextree}')
                print(f'vy: {vy}, token: {self.token_list[vy]}')
                retval = self.get_step_biasing_embs(
                    [vy], [hyp.lextree], [lextree], node_encs
                )
                step_mask = retval[0]
                step_embs = retval[1]
                trees = retval[2]
                p_gen_mask = retval[3]
                back_transform = retval[4]
                index_list = retval[5]
                print(f'index list: {index_list}')
                ookb_list = self.token_list + ['<ool>']
                print(f'index list: {[ookb_list[k] for k in index_list[0]]}')
                query_char = self.decoder.embed(
                    # torch.LongTensor([vy]).to(node_encs.device)
                    torch.LongTensor([vy]).to(dec_out.device)
                ).squeeze(0)
                query_char = self.Qproj_char(query_char)
                query_acoustic = self.Qproj_acoustic(enc_out_t)
                query = (query_char + query_acoustic).unsqueeze(0).unsqueeze(0)
                hptr, tcpgen_dist = self.get_meetingKB_emb_map(
                    query,
                    step_mask,
                    back_transform,
                    index_list,
                    meeting_KB=step_embs,
                )
                tcpgen_dist = tcpgen_dist[0, 0]
                hptr = hptr[0, 0]
            print('_' * 30)
            if self.biasing and self.deepbiasing:
                joint_out, joint_act = self.joint_network(enc_out_t, dec_out, hptr)
            else:
                joint_out = self.joint_network(enc_out_t, dec_out)
        
            # biasing
            if self.biasing and lextree is not None:
                p_gen = torch.sigmoid(
                    self.pointer_gate(torch.cat([joint_act, hptr], dim=-1))
                )
                model_dist = torch.softmax(joint_out, dim=-1)
                top_model_prob, top_model_pred   = torch.max(model_dist, dim=-1)
                top_tcpgen_prob, top_tcpgen_pred = torch.max(tcpgen_dist, dim=-1)
                topk_model_prob, topk_model_pred = torch.topk(model_dist, 100, dim=-1)
                topk_tcpgen_prob, topk_tcpgen_pred = torch.topk(tcpgen_dist, 100, dim=-1)

                p_gen = p_gen.item() if p_gen_mask[0] == 0 else 0
                # p_gen = (1 - top_model_prob.item()) if p_gen_mask[0] == 0 else 0
                # p_gen = (1 - top_tcpgen_prob.item()) if p_gen_mask[0] == 0 else 0
                # Get factorised loss
                p_not_null = 1.0 - model_dist[0:1]
                ptr_dist_fact = tcpgen_dist[1:] * p_not_null
                ptr_gen_complement = tcpgen_dist[-1:] * p_gen
                p_partial = ptr_dist_fact[:-1] * p_gen + model_dist[1:] * (
                    1 - p_gen + ptr_gen_complement
                )
                p_final = torch.cat([model_dist[0:1], p_partial], dim=-1)
                logp = torch.log(p_final)
            else:
                logp = torch.log_softmax(
                    joint_out,
                    dim=-1,
                )
            top_logp, pred = torch.max(logp, dim=-1)
            if pred != self.blank_id:
                hyp.yseq.append(int(pred))
                hyp.score += float(top_logp)
                hyp.dec_state = state
                hyp.lextree=trees[0] if self.biasing else None

                dec_out, state, _ = self.decoder.score(hyp, cache)
                if self.biasing:
                    hyp.pgens.append(p_gen)
                    hyp.top_model_prob.append(float(top_model_prob))
                    hyp.top_model_pred.append(int(top_model_pred))
                    hyp.top_tcpgen_prob.append(float(top_tcpgen_prob))
                    hyp.top_tcpgen_pred.append(int(top_tcpgen_pred))
                    hyp.topk_tcpgen_prob.append(topk_tcpgen_prob.tolist())
                    hyp.topk_tcpgen_pred.append(topk_tcpgen_pred.tolist())
                    hyp.topk_model_prob.append(topk_model_prob.tolist())
                    hyp.topk_model_pred.append(topk_model_pred.tolist())
        
        return [hyp]

    def ContextualBiasing_greedy_search(
            self, 
            enc_out      : torch.Tensor, 
            cb_tokens    : torch.Tensor,
            cb_tokens_len: torch.Tensor,
        ) -> List[Hypothesis]:
        """Greedy search implementation.

        Args:
            enc_out: Encoder output sequence. (T, D_enc)

        Returns:
            hyp: 1-best hypotheses.

        """
        logging.info('Using Contextual Biasing greedy search!')
        dec_state = self.decoder.init_state(1)

        hyp = Hypothesis(score=0.0, yseq=[self.blank_id], dec_state=dec_state)
        cache = {}

        cb_tokens     = cb_tokens.to(enc_out.device)
        cb_tokens_len = cb_tokens_len.to(enc_out.device)
        embed_matrix  = torch.cat(
            [self.decoder.embed.weight.data, self.ooKBemb.weight], dim=0
        )
        cb_tokens_embed = embed_matrix[cb_tokens]
        cb_seq_embed, _ = self.CbRNN(cb_tokens_embed)
        cb_embeds       = torch.mean(cb_seq_embed, dim=1)

        dec_out, state, _ = self.decoder.score(hyp, cache)
        topk_logps = []
        topk_ids   = []
        for enc_out_t in enc_out:
            lin_encoder_out = self.joint_network.lin_enc(enc_out_t)
            aco_bias = self.get_acoustic_biasing_vector(enc_out_t, cb_embeds)

            lin_decoder_out = self.joint_network.lin_dec(dec_out)
            lin_encoder_out = lin_encoder_out + aco_bias

            joint_out = self.joint_network.joint_activation(
                lin_encoder_out + lin_decoder_out
            )
            joint_out = self.joint_network.lin_out(joint_out)
            logp      = torch.log_softmax(
                joint_out,
                dim=-1,
            )
            # logp = torch.log_softmax(
            #     self.joint_network(enc_out_t, dec_out),
            #     dim=-1,
            # )
            top_logp, pred = torch.max(logp, dim=-1)

            if pred != self.blank_id:
                _topk_logp, _topk_ids = torch.topk(logp, 50)
                hyp.yseq.append(int(pred))
                hyp.score += float(top_logp)

                hyp.dec_state = state
                topk_logps.append(_topk_logp)
                topk_ids.append(_topk_ids)
                dec_out, state, _ = self.decoder.score(hyp, cache)

        return [hyp]

    def ContextualBiasing_trie_greedy_search(
        self, 
        enc_out      : torch.Tensor, 
        cb_tokens    : torch.Tensor,
        cb_tokens_len: torch.Tensor, 
        lextree      : list = None
    ) -> List[Hypothesis]:
        """Greedy search implementation.

        Args:
            enc_out: Encoder output sequence. (T, D_enc)

        Returns:
            hyp: 1-best hypotheses.

        """
        logging.info('Using Contextual Biasing Trie greedy search!')
        dec_state = self.decoder.init_state(1)

        hyp = Hypothesis(score=0.0, yseq=[self.blank_id], dec_state=dec_state, lextree=lextree)
        cache = {}

        cb_tokens     = cb_tokens.to(enc_out.device)
        cb_tokens_len = cb_tokens_len.to(enc_out.device)
        embed_matrix  = torch.cat(
            [self.decoder.embed.weight.data, self.ooKBemb.weight], dim=0
        )
        cb_tokens_embed = embed_matrix[cb_tokens]
        cb_seq_embed, _ = self.CbRNN(cb_tokens_embed)
        cb_embeds       = torch.mean(cb_seq_embed, dim=1)
        # print(f'cb_embeds: {cb_embeds.shape}')
        dec_out, state, _ = self.decoder.score(hyp, cache)

        trees = [None]
        for enc_out_t in enc_out:
            lin_encoder_out = self.joint_network.lin_enc(enc_out_t)
            lin_decoder_out = self.joint_network.lin_dec(dec_out)
            
            if lextree is not None:
                vy = hyp.yseq[-1] if len(hyp.yseq) > 1 else self.blank_id
                trees, p_gen_mask, index_list = self.get_step_biasing_embs_cb(
                    [vy], [hyp.lextree], [lextree]
                )
                gate = True if p_gen_mask[0] == 0 else False
                cb_embeds_sub = cb_embeds[index_list[0] + [cb_embeds.shape[0] - 1]]
                if gate:
                    aco_bias = self.get_acoustic_biasing_vector(enc_out_t, cb_embeds_sub)
                    lin_encoder_out = lin_encoder_out + aco_bias
            else:
                aco_bias = self.get_acoustic_biasing_vector(enc_out_t, cb_embeds)
                lin_encoder_out = lin_encoder_out + aco_bias
            
            joint_out = self.joint_network.joint_activation(
                lin_encoder_out + lin_decoder_out
            )
            joint_out = self.joint_network.lin_out(joint_out)
            logp      = torch.log_softmax(
                joint_out,
                dim=-1,
            )
            top_logp, pred = torch.max(logp, dim=-1)

            if pred != self.blank_id:
                hyp.yseq.append(int(pred))
                hyp.score += float(top_logp)
                hyp.dec_state = state
                if lextree != None:
                    hyp.lextree=trees[0]
                dec_out, state, _ = self.decoder.score(hyp, cache)
        return [hyp]

    def default_beam_search(self, enc_out: torch.Tensor) -> List[Hypothesis]:
        """Beam search implementation.

        Modified from https://arxiv.org/pdf/1211.3711.pdf

        Args:
            enc_out: Encoder output sequence. (T, D)

        Returns:
            nbest_hyps: N-best hypothesis.

        """
        beam = min(self.beam_size, self.vocab_size)
        beam_k = min(beam, (self.vocab_size - 1))

        dec_state = self.decoder.init_state(1)

        kept_hyps = [
            Hypothesis(
                score=0.0, 
                yseq=[self.blank_id], 
                dec_state=dec_state
            )
        ]
        cache = {}
        cache_lm = {}

        for enc_out_t in enc_out:
            hyps = kept_hyps
            kept_hyps = []

            if self.token_list is not None:
                logging.debug(
                    "\n"
                    + "\n".join(
                        [
                            "hypo: "
                            + "".join([self.token_list[x] for x in hyp.yseq[1:]])
                            + f", score: {round(float(hyp.score), 2)}"
                            for hyp in sorted(hyps, key=lambda x: x.score, reverse=True)
                        ]
                    )
                )

            while True:
                if self.score_norm_during:
                    max_hyp = max(hyps, key=lambda x: x.score / len(x.yseq))
                else:
                    max_hyp = max(hyps, key=lambda x: x.score)
                hyps.remove(max_hyp)

                dec_out, state, lm_tokens = self.decoder.score(max_hyp, cache)

                logp = torch.log_softmax(
                    self.joint_network(enc_out_t, dec_out),
                    dim=-1,
                )
                top_k = logp[1:].topk(beam_k, dim=-1)

                kept_hyps.append(
                    Hypothesis(
                        score=(max_hyp.score + float(logp[0:1])),
                        yseq=max_hyp.yseq[:],
                        dec_state=max_hyp.dec_state,
                        lm_state=max_hyp.lm_state,
                    )
                )
                if self.use_lm:
                    if tuple(max_hyp.yseq) not in cache_lm:
                        _tmp = torch.LongTensor(
                            [self.sos] + max_hyp.yseq[1:],
                            # device=self.decoder.device,
                        )
                        _tmp = _tmp.to(self.decoder.device)
                        lm_scores, lm_state = self.lm.score(
                            _tmp,
                            max_hyp.lm_state,
                            None,
                        )
                        cache_lm[tuple(max_hyp.yseq)] = (lm_scores, lm_state)
                    else:
                        lm_scores, lm_state = cache_lm[tuple(max_hyp.yseq)]
                else:
                    lm_state = max_hyp.lm_state

                for logp, k in zip(*top_k):
                    score = max_hyp.score + float(logp)

                    if self.use_lm:
                        score += self.lm_weight * lm_scores[k + 1]

                    hyps.append(
                        Hypothesis(
                            score=score,
                            yseq=max_hyp.yseq[:] + [int(k + 1)],
                            dec_state=state,
                            lm_state=lm_state,
                        )
                    )

                if self.score_norm_during:
                    hyps_max = float(
                        max(hyps, key=lambda x: x.score / len(x.yseq)).score
                    )
                else:
                    hyps_max = float(max(hyps, key=lambda x: x.score).score)
                kept_most_prob = sorted(
                    [hyp for hyp in kept_hyps if hyp.score > hyps_max],
                    key=lambda x: x.score,
                )
                if len(kept_most_prob) >= beam:
                    kept_hyps = kept_most_prob
                    break

        return self.sort_nbest(kept_hyps)

    def TCPGen_biasing_beam_search(
        self, enc_out: torch.Tensor, lextree: list = None
    ) -> List[Hypothesis]:
        """Beam search implementation.

        Modified from https://arxiv.org/pdf/1211.3711.pdf

        Args:
            enc_out: Encoder output sequence. (T, D)
            lextree: prefix tree structure of biasing list

        Returns:
            nbest_hyps: N-best hypothesis.

        """
        beam = min(self.beam_size, self.vocab_size)
        beam_k = min(beam, (self.vocab_size - 1))

        dec_state = self.decoder.init_state(1)

        kept_hyps = [
            Hypothesis(
                score=0.0, 
                yseq=[self.blank_id], 
                dec_state=dec_state, 
                lextree=lextree,
                pgens=[0],
                top_model_prob=[0],
                top_model_pred=[0],
                top_tcpgen_prob=[0],
                top_tcpgen_pred=[0],
                topk_tcpgen_prob=[0],
                topk_tcpgen_pred=[0],
                topk_model_prob=[0],
                topk_model_pred=[0]
            )
        ]
        cache = {}
        cache_lm = {}

        # Encode prefix tree using GNN
        node_encs = None
        if self.gnn is not None:
            node_encs = self.gnn(lextree, self.decoder.embed)

        for enc_out_t in enc_out:
            hyps = kept_hyps
            kept_hyps = []

            if self.token_list is not None:
                logging.debug(
                    "\n"
                    + "\n".join(
                        [
                            "hypo: "
                            + "".join([self.token_list[x] for x in hyp.yseq[1:]])
                            + f", score: {round(float(hyp.score), 2)}"
                            for hyp in sorted(hyps, key=lambda x: x.score, reverse=True)
                        ]
                    )
                )

            while True:
                max_hyp = max(hyps, key=lambda x: x.score)
                hyps.remove(max_hyp)

                dec_out, state, lm_tokens = self.decoder.score(max_hyp, cache)

                # biasing
                trees = [None]
                if self.biasing and lextree is not None:
                    vy = max_hyp.yseq[-1] if len(max_hyp.yseq) > 1 else self.blank_id
                    retval = self.get_step_biasing_embs(
                        [vy], [max_hyp.lextree], [lextree], node_encs
                    )
                    step_mask = retval[0]
                    step_embs = retval[1]
                    trees = retval[2]
                    p_gen_mask = retval[3]
                    back_transform = retval[4]
                    index_list = retval[5]

                    query_char = self.decoder.embed(
                        torch.LongTensor([vy]).to(dec_out.device)
                    ).squeeze(0)
                    query_char = self.Qproj_char(query_char)
                    query_acoustic = self.Qproj_acoustic(enc_out_t)
                    query = (query_char + query_acoustic).unsqueeze(0).unsqueeze(0)
                    hptr, tcpgen_dist = self.get_meetingKB_emb_map(
                        query,
                        step_mask,
                        back_transform,
                        index_list,
                        meeting_KB=step_embs,
                    )
                    tcpgen_dist = tcpgen_dist[0, 0]
                    hptr = hptr[0, 0]

                if self.biasing and self.deepbiasing:
                    joint_out, joint_act = self.joint_network(enc_out_t, dec_out, hptr)
                elif self.biasing:
                    joint_out, joint_act = self.joint_network(enc_out_t, dec_out)
                else:
                    joint_out = self.joint_network(enc_out_t, dec_out)

                # biasing
                if self.biasing and lextree is not None:
                    p_gen = torch.sigmoid(
                        self.pointer_gate(torch.cat([joint_act, hptr], dim=-1))
                    )
                    model_dist = torch.softmax(joint_out, dim=-1)
                    top_model_prob, top_model_pred = torch.max(model_dist, dim=-1)
                    p_gen = p_gen if p_gen_mask[0] == 0 else 0
                    # p_gen = (1 - top_model_prob) if p_gen_mask[0] == 0 else 0
                    # Get factorised loss
                    p_not_null = 1.0 - model_dist[0:1]
                    ptr_dist_fact = tcpgen_dist[1:] * p_not_null
                    ptr_gen_complement = tcpgen_dist[-1:] * p_gen
                    p_partial = ptr_dist_fact[:-1] * p_gen + model_dist[1:] * (
                        1 - p_gen + ptr_gen_complement
                    )
                    p_final = torch.cat([model_dist[0:1], p_partial], dim=-1)
                    logp = torch.log(p_final)
                else:
                    logp = torch.log_softmax(
                        joint_out,
                        dim=-1,
                    )
                top_k = logp[1:].topk(beam_k, dim=-1)

                kept_hyps.append(
                    Hypothesis(
                        score=(max_hyp.score + float(logp[0:1])),
                        yseq=max_hyp.yseq[:],
                        dec_state=max_hyp.dec_state,
                        lm_state=max_hyp.lm_state,
                        lextree=max_hyp.lextree
                    )
                )

                if self.use_lm:
                    if tuple(max_hyp.yseq) not in cache_lm:
                        lm_scores, lm_state = self.lm.score(
                            torch.LongTensor(
                                [self.sos] + max_hyp.yseq[1:],
                                device=self.decoder.device,
                            ),
                            max_hyp.lm_state,
                            None,
                        )
                        cache_lm[tuple(max_hyp.yseq)] = (lm_scores, lm_state)
                    else:
                        lm_scores, lm_state = cache_lm[tuple(max_hyp.yseq)]
                else:
                    lm_state = max_hyp.lm_state

                for logp, k in zip(*top_k):
                    score = max_hyp.score + float(logp)

                    if self.use_lm:
                        score += self.lm_weight * lm_scores[k + 1]

                    hyps.append(
                        Hypothesis(
                            score=score,
                            yseq=max_hyp.yseq[:] + [int(k + 1)],
                            dec_state=state,
                            lm_state=lm_state,
                            lextree=trees[0] if self.biasing else None,
                        )
                    )

                hyps_max = float(max(hyps, key=lambda x: x.score).score)
                kept_most_prob = sorted(
                    [hyp for hyp in kept_hyps if hyp.score > hyps_max],
                    key=lambda x: x.score,
                )
                if len(kept_most_prob) >= beam:
                    kept_hyps = kept_most_prob
                    break

        return self.sort_nbest(kept_hyps)

    def ContextualBiasing_beam_search(
            self, 
            enc_out      : torch.Tensor, 
            cb_tokens    : torch.Tensor,
            cb_tokens_len: torch.Tensor,
        ) -> List[Hypothesis]:
        """Beam search implementation.

        Modified from https://arxiv.org/pdf/1211.3711.pdf

        Args:
            enc_out: Encoder output sequence. (T, D)

        Returns:
            nbest_hyps: N-best hypothesis.

        """
        logging.info('contextual biasing beam search!')
        beam = min(self.beam_size, self.vocab_size)
        beam_k = min(beam, (self.vocab_size - 1))

        dec_state = self.decoder.init_state(1)

        kept_hyps = [
            Hypothesis(
                score=0.0, 
                yseq=[self.blank_id], 
                dec_state=dec_state
            )
        ]
        cache = {}
        cache_lm = {}

        # Acoustic biasing
        # cb_tokens     = cb_tokens.to(enc_out.device)
        # embed_matrix  = torch.cat(
        #     [self.decoder.embed.weight.data, self.ooKBemb.weight], dim=0
        # )
        # cb_token_embed   = embed_matrix[cb_tokens]
        # cb_tokens_packed = torch.nn.utils.rnn.pack_padded_sequence(
        #     cb_token_embed, 
        #     cb_tokens_len,
        #     batch_first=True,
        #     enforce_sorted=False
        # )
        # cb_seq_embed_packed, _ = self.CbRNN(cb_tokens_packed)
        # cb_seq_embed_packed, input_sizes = torch.nn.utils.rnn.pad_packed_sequence(
        #     cb_seq_embed_packed, 
        #     batch_first=True
        # )
        # input_sizes = input_sizes.to(enc_out.device)
        # cb_embeds   = torch.sum(cb_seq_embed_packed, dim=1) / input_sizes.unsqueeze(-1)
        cb_tokens     = cb_tokens.to(enc_out.device)
        cb_tokens_len = cb_tokens_len.to(enc_out.device)
        embed_matrix  = torch.cat(
            [self.decoder.embed.weight.data, self.ooKBemb.weight], dim=0
        )
        cb_tokens_embed = embed_matrix[cb_tokens]
        cb_seq_embed, _ = self.CbRNN(cb_tokens_embed)
        cb_embeds       = torch.mean(cb_seq_embed, dim=1)
        
        for enc_out_t in enc_out:
            hyps = kept_hyps
            kept_hyps = []

            if self.token_list is not None:
                logging.debug(
                    "\n"
                    + "\n".join(
                        [
                            "hypo: "
                            + "".join([self.token_list[x] for x in hyp.yseq[1:]])
                            + f", score: {round(float(hyp.score), 2)}"
                            for hyp in sorted(hyps, key=lambda x: x.score, reverse=True)
                        ]
                    )
                )

            while True:
                if self.score_norm_during:
                    max_hyp = max(hyps, key=lambda x: x.score / len(x.yseq))
                else:
                    max_hyp = max(hyps, key=lambda x: x.score)
                hyps.remove(max_hyp)

                dec_out, state, lm_tokens = self.decoder.score(max_hyp, cache)

                lin_encoder_out = self.joint_network.lin_enc(enc_out_t)
                aco_bias = self.get_acoustic_biasing_vector(enc_out_t, cb_embeds)

                lin_decoder_out = self.joint_network.lin_dec(dec_out)
                lin_encoder_out = lin_encoder_out + aco_bias

                joint_out = self.joint_network.joint_activation(
                    lin_encoder_out + lin_decoder_out
                )
                joint_out = self.joint_network.lin_out(joint_out)

                # joint_out, _ = self.joint_network(enc_out_t, dec_out)
                logp = torch.log_softmax(
                    joint_out,
                    dim=-1,
                )
                top_k = logp[1:].topk(beam_k, dim=-1)

                kept_hyps.append(
                    Hypothesis(
                        score=(max_hyp.score + float(logp[0:1])),
                        yseq=max_hyp.yseq[:],
                        dec_state=max_hyp.dec_state,
                        lm_state=max_hyp.lm_state,
                    )
                )
                if self.use_lm:
                    if tuple(max_hyp.yseq) not in cache_lm:
                        _tmp = torch.LongTensor(
                            [self.sos] + max_hyp.yseq[1:],
                            # device=self.decoder.device,
                        )
                        _tmp = _tmp.to(self.decoder.device)
                        lm_scores, lm_state = self.lm.score(
                            _tmp,
                            max_hyp.lm_state,
                            None,
                        )
                        cache_lm[tuple(max_hyp.yseq)] = (lm_scores, lm_state)
                    else:
                        lm_scores, lm_state = cache_lm[tuple(max_hyp.yseq)]
                else:
                    lm_state = max_hyp.lm_state

                for logp, k in zip(*top_k):
                    score = max_hyp.score + float(logp)

                    if self.use_lm:
                        score += self.lm_weight * lm_scores[k + 1]

                    hyps.append(
                        Hypothesis(
                            score=score,
                            yseq=max_hyp.yseq[:] + [int(k + 1)],
                            dec_state=state,
                            lm_state=lm_state,
                        )
                    )

                if self.score_norm_during:
                    hyps_max = float(
                        max(hyps, key=lambda x: x.score / len(x.yseq)).score
                    )
                else:
                    hyps_max = float(max(hyps, key=lambda x: x.score).score)
                kept_most_prob = sorted(
                    [hyp for hyp in kept_hyps if hyp.score > hyps_max],
                    key=lambda x: x.score,
                )
                if len(kept_most_prob) >= beam:
                    kept_hyps = kept_most_prob
                    break

        return self.sort_nbest(kept_hyps)

    def ContextualBiasing_trie_beam_search(
            self, 
            enc_out      : torch.Tensor, 
            cb_tokens    : torch.Tensor,
            cb_tokens_len: torch.Tensor,
            lextree      : list = None
        ) -> List[Hypothesis]:
        """Beam search implementation.

        Modified from https://arxiv.org/pdf/1211.3711.pdf

        Args:
            enc_out: Encoder output sequence. (T, D)

        Returns:
            nbest_hyps: N-best hypothesis.

        """
        logging.info('contextual biasing trie beam search!')
        beam = min(self.beam_size, self.vocab_size)
        beam_k = min(beam, (self.vocab_size - 1))

        dec_state = self.decoder.init_state(1)

        kept_hyps = [
            Hypothesis(
                score=0.0, 
                yseq=[self.blank_id], 
                dec_state=dec_state,
                lextree=lextree
            )
        ]
        cache = {}
        cache_lm = {}

        # Acoustic biasing
        cb_tokens     = cb_tokens.to(enc_out.device)
        cb_tokens_len = cb_tokens_len.to(enc_out.device)
        embed_matrix  = torch.cat(
            [self.decoder.embed.weight.data, self.ooKBemb.weight], dim=0
        )
        cb_tokens_embed = embed_matrix[cb_tokens]
        cb_seq_embed, _ = self.CbRNN(cb_tokens_embed)
        cb_embeds       = torch.mean(cb_seq_embed, dim=1)
        
        for enc_out_t in enc_out:
            hyps = kept_hyps
            kept_hyps = []

            if self.token_list is not None:
                logging.debug(
                    "\n"
                    + "\n".join(
                        [
                            "hypo: "
                            + "".join([self.token_list[x] for x in hyp.yseq[1:]])
                            + f", score: {round(float(hyp.score), 2)}"
                            for hyp in sorted(hyps, key=lambda x: x.score, reverse=True)
                        ]
                    )
                )

            while True:
                if self.score_norm_during:
                    max_hyp = max(hyps, key=lambda x: x.score / len(x.yseq))
                else:
                    max_hyp = max(hyps, key=lambda x: x.score)
                hyps.remove(max_hyp)

                dec_out, state, lm_tokens = self.decoder.score(max_hyp, cache)

                lin_encoder_out = self.joint_network.lin_enc(enc_out_t)
                lin_decoder_out = self.joint_network.lin_dec(dec_out)

                if lextree is not None:
                    vy = max_hyp.yseq[-1] if len(max_hyp.yseq) > 1 else self.blank_id
                    trees, p_gen_mask, index_list = self.get_step_biasing_embs_cb(
                        [vy], [max_hyp.lextree], [lextree]
                    )
                    gate = True if p_gen_mask[0] == 0 else False
                    cb_embeds_sub = cb_embeds[index_list[0] + [cb_embeds.shape[0] - 1]]
                    if gate:
                        aco_bias = self.get_acoustic_biasing_vector(enc_out_t, cb_embeds_sub)
                        lin_encoder_out = lin_encoder_out + aco_bias
                else:
                    aco_bias = self.get_acoustic_biasing_vector(enc_out_t, cb_embeds)
                    lin_encoder_out = lin_encoder_out + aco_bias

                joint_out = self.joint_network.joint_activation(
                    lin_encoder_out + lin_decoder_out
                )
                joint_out = self.joint_network.lin_out(joint_out)

                # joint_out, _ = self.joint_network(enc_out_t, dec_out)
                logp = torch.log_softmax(
                    joint_out,
                    dim=-1,
                )
                top_k = logp[1:].topk(beam_k, dim=-1)

                kept_hyps.append(
                    Hypothesis(
                        score=(max_hyp.score + float(logp[0:1])),
                        yseq=max_hyp.yseq[:],
                        dec_state=max_hyp.dec_state,
                        lm_state=max_hyp.lm_state,
                        lextree=max_hyp.lextree,
                    )
                )
                if self.use_lm:
                    if tuple(max_hyp.yseq) not in cache_lm:
                        _tmp = torch.LongTensor(
                            [self.sos] + max_hyp.yseq[1:],
                            # device=self.decoder.device,
                        )
                        _tmp = _tmp.to(self.decoder.device)
                        lm_scores, lm_state = self.lm.score(
                            _tmp,
                            max_hyp.lm_state,
                            None,
                        )
                        cache_lm[tuple(max_hyp.yseq)] = (lm_scores, lm_state)
                    else:
                        lm_scores, lm_state = cache_lm[tuple(max_hyp.yseq)]
                else:
                    lm_state = max_hyp.lm_state

                for logp, k in zip(*top_k):
                    score = max_hyp.score + float(logp)

                    if self.use_lm:
                        score += self.lm_weight * lm_scores[k + 1]

                    hyps.append(
                        Hypothesis(
                            score=score,
                            yseq=max_hyp.yseq[:] + [int(k + 1)],
                            dec_state=state,
                            lm_state=lm_state,
                            lextree=trees[0] if lextree else None
                        )
                    )

                if self.score_norm_during:
                    hyps_max = float(
                        max(hyps, key=lambda x: x.score / len(x.yseq)).score
                    )
                else:
                    hyps_max = float(max(hyps, key=lambda x: x.score).score)
                kept_most_prob = sorted(
                    [hyp for hyp in kept_hyps if hyp.score > hyps_max],
                    key=lambda x: x.score,
                )
                if len(kept_most_prob) >= beam:
                    kept_hyps = kept_most_prob
                    break

        return self.sort_nbest(kept_hyps)

    def ContextualBiasingPredictor_beam_search(
            self, 
            enc_out      : torch.Tensor, 
            cb_tokens    : torch.Tensor,
            cb_tokens_len: torch.Tensor,
        ) -> List[Hypothesis]:
        """Beam search implementation.

        Modified from https://arxiv.org/pdf/1211.3711.pdf

        Args:
            enc_out: Encoder output sequence. (T, D)

        Returns:
            nbest_hyps: N-best hypothesis.

        """
        logging.info('contextual biasing beam search!')
        beam = min(self.beam_size, self.vocab_size)
        beam_k = min(beam, (self.vocab_size - 1))

        dec_state = self.decoder.init_state(1)

        kept_hyps = [
            Hypothesis(
                score=0.0, 
                yseq=[self.blank_id], 
                dec_state=dec_state
            )
        ]
        cache = {}
        cache_lm = {}

        # Acoustic biasing
        cb_tokens     = cb_tokens.to(enc_out.device)
        # cb_tokens_len = (cb_tokens_len - 1).to(enc_out.device)
        cb_tokens_len = cb_tokens_len.to(enc_out.device)
        embed_matrix  = torch.cat(
            [self.decoder.embed.weight.data, self.ooKBemb.weight], dim=0
        )
        cb_tokens_embed = embed_matrix[cb_tokens]
        cb_seq_embed, _ = self.CbRNN(cb_tokens_embed)
        cb_embed = torch.mean(cb_seq_embed, dim=1)
        
        for enc_out_t in enc_out:
            hyps = kept_hyps
            kept_hyps = []

            if self.token_list is not None:
                logging.debug(
                    "\n"
                    + "\n".join(
                        [
                            "hypo: "
                            + "".join([self.token_list[x] for x in hyp.yseq[1:]])
                            + f", score: {round(float(hyp.score), 2)}"
                            for hyp in sorted(hyps, key=lambda x: x.score, reverse=True)
                        ]
                    )
                )

            while True:
                if self.score_norm_during:
                    max_hyp = max(hyps, key=lambda x: x.score / len(x.yseq))
                else:
                    max_hyp = max(hyps, key=lambda x: x.score)
                hyps.remove(max_hyp)

                dec_out, state, lm_tokens = self.decoder.score(max_hyp, cache)
                sem_bias, atten = self.get_semantic_biasing_vector(
                    dec_out, cb_embed, return_atten=True
                )
                
                lin_encoder_out = self.joint_network.lin_enc(enc_out_t)
                lin_decoder_out = self.joint_network.lin_dec(dec_out)
                lin_decoder_out = lin_decoder_out + sem_bias

                joint_out = self.joint_network.joint_activation(
                    lin_encoder_out + lin_decoder_out
                )
                joint_out = self.joint_network.lin_out(joint_out)

                # joint_out, _ = self.joint_network(enc_out_t, dec_out)
                logp = torch.log_softmax(
                    joint_out,
                    dim=-1,
                )
                top_k = logp[1:].topk(beam_k, dim=-1)

                kept_hyps.append(
                    Hypothesis(
                        score=(max_hyp.score + float(logp[0:1])),
                        yseq=max_hyp.yseq[:],
                        dec_state=max_hyp.dec_state,
                        lm_state=max_hyp.lm_state,
                    )
                )
                if self.use_lm:
                    if tuple(max_hyp.yseq) not in cache_lm:
                        _tmp = torch.LongTensor(
                            [self.sos] + max_hyp.yseq[1:],
                            # device=self.decoder.device,
                        )
                        _tmp = _tmp.to(self.decoder.device)
                        lm_scores, lm_state = self.lm.score(
                            _tmp,
                            max_hyp.lm_state,
                            None,
                        )
                        cache_lm[tuple(max_hyp.yseq)] = (lm_scores, lm_state)
                    else:
                        lm_scores, lm_state = cache_lm[tuple(max_hyp.yseq)]
                else:
                    lm_state = max_hyp.lm_state

                for logp, k in zip(*top_k):
                    score = max_hyp.score + float(logp)

                    if self.use_lm:
                        score += self.lm_weight * lm_scores[k + 1]

                    hyps.append(
                        Hypothesis(
                            score=score,
                            yseq=max_hyp.yseq[:] + [int(k + 1)],
                            dec_state=state,
                            lm_state=lm_state,
                        )
                    )

                if self.score_norm_during:
                    hyps_max = float(
                        max(hyps, key=lambda x: x.score / len(x.yseq)).score
                    )
                else:
                    hyps_max = float(max(hyps, key=lambda x: x.score).score)
                kept_most_prob = sorted(
                    [hyp for hyp in kept_hyps if hyp.score > hyps_max],
                    key=lambda x: x.score,
                )
                if len(kept_most_prob) >= beam:
                    kept_hyps = kept_most_prob
                    break

        return self.sort_nbest(kept_hyps)

    def get_step_biasing_embs(self, char_ids, trees, origTries, node_encs=None):
        ooKB_id = self.vocab_size
        p_gen_mask = []
        maxlen = 0
        index_list = []
        new_trees = []
        masks_list = []
        nodes_list = []
        step_embs = []
        for i, vy in enumerate(char_ids):
            new_tree = trees[i][0]
            if vy == self.blank_id:
                new_tree = origTries[i]
                p_gen_mask.append(0)
            elif self.token_list[vy].endswith("▁"):
                if vy in new_tree and new_tree[vy][0] != {}:
                    new_tree = new_tree[vy]
                else:
                    new_tree = origTries[i]
                p_gen_mask.append(0)
            elif vy not in new_tree:
                new_tree = [{}]
                p_gen_mask.append(1)
                # new_tree = origTries[i]
                # p_gen_mask.append(0)
            else:
                new_tree = new_tree[vy]
                p_gen_mask.append(0)
            new_trees.append(new_tree)
            if len(new_tree[0].keys()) > maxlen:
                maxlen = len(new_tree[0].keys())
            index_list.append(list(new_tree[0].keys()))
            if node_encs is not None:
                nodes_list.append([value[4] for key, value in new_tree[0].items()])

        maxlen += 1
        step_mask = []
        back_transform = torch.zeros(
            len(new_trees), maxlen, ooKB_id + 1, device=self.decoder.device
        )
        ones_mat = torch.ones(back_transform.size()).to(self.decoder.device)
        for i, indices in enumerate(index_list):
            step_mask.append(
                len(indices) * [0] + (maxlen - len(indices) - 1) * [1] + [0]
            )
            if node_encs is not None:
                nodes_list[i] = nodes_list[i] + [node_encs.size(0)] * (
                    maxlen - len(indices)
                )
            indices += [ooKB_id] * (maxlen - len(indices))
        step_mask = torch.tensor(step_mask).byte().to(self.decoder.device)
        index_list = torch.LongTensor(index_list).to(self.decoder.device)
        back_transform.scatter_(dim=-1, index=index_list.unsqueeze(-1), src=ones_mat)
        if node_encs is not None:
            node_encs = torch.cat([node_encs, self.ooKBemb.weight], dim=0)
            step_embs = node_encs[torch.tensor(nodes_list).to(node_encs.device)]

        return step_mask, step_embs, new_trees, p_gen_mask, back_transform, index_list

    def get_step_biasing_embs_cb(self, char_ids, trees, origTries):
        ooKB_id = self.vocab_size
        p_gen_mask = []
        index_list = []
        new_trees  = []
        for i, vy in enumerate(char_ids):
            new_tree = trees[i][0]
            if vy == self.blank_id:
                new_tree = origTries[i]
                p_gen_mask.append(0)
            elif self.token_list[vy].endswith("▁"):
                if vy in new_tree and new_tree[vy][0] != {}:
                    new_tree = new_tree[vy]
                else:
                    new_tree = origTries[i]
                p_gen_mask.append(0)
            elif vy not in new_tree:
                new_tree = [{}]
                p_gen_mask.append(1)
            else:
                new_tree = new_tree[vy]
                p_gen_mask.append(0)
            new_trees.append(new_tree)
            index_list.append(list(new_tree[-1]))

        return new_trees, p_gen_mask, index_list

    def get_meetingKB_emb_map(
        self,
        query,
        meeting_mask,
        back_transform,
        index_list,
        meeting_KB=[],
    ):
        if meeting_KB == []:
            meeting_KB = torch.cat(
                [self.decoder.embed.weight.data, self.ooKBemb.weight], dim=0
            )
            meeting_KB = meeting_KB[index_list]
        meeting_KB = self.Kproj(meeting_KB)
        KBweight = torch.einsum("ijk,itk->itj", meeting_KB, query)
        KBweight = KBweight / math.sqrt(query.size(-1))
        KBweight.masked_fill_(
            meeting_mask.bool().unsqueeze(1).repeat(1, query.size(1), 1), -1e9
        )
        KBweight = torch.nn.functional.softmax(KBweight, dim=-1)
        if meeting_KB.size(1) > 1:
            KBembedding = torch.einsum(
                "ijk,itj->itk", meeting_KB[:, :-1, :], KBweight[:, :, :-1]
            )
        else:
            KBembedding = KBweight.new_zeros(
                meeting_KB.size(0), query.size(1), meeting_KB.size(-1)
            )
        KBweight = torch.einsum("ijk,itj->itk", back_transform, KBweight)
        return KBembedding, KBweight

    def get_acoustic_biasing_vector(
        self,
        encoder_out,
        biasing_embed,
        return_atten=False
    ):
        query = self.Qproj_acoustic(encoder_out).unsqueeze(0)
        key   = self.Kproj(biasing_embed)
        value = self.Vproj(biasing_embed)
        # attention
        qk    = torch.einsum("tk,jk->jt", key, query)
        qk    = qk / math.sqrt(query.size(-1))
        score = torch.nn.functional.softmax(qk, dim=-1)
        attn  = torch.einsum("tk,jt->jk", value, score)
        if return_atten:
            return self.proj(attn).squeeze(0), score
        else:
            return self.proj(attn).squeeze(0)

    def get_semantic_biasing_vector(
        self,
        decoder_out,
        biasing_embed,
        return_atten=False
    ):
        query = self.Qproj_semantic(decoder_out).unsqueeze(0)
        key   = self.Kproj(biasing_embed)
        value = self.Vproj(biasing_embed)
        # attention
        qk    = torch.einsum("tk,jk->jt", key, query)
        qk    = qk / math.sqrt(query.size(-1))
        score = torch.nn.functional.softmax(qk, dim=-1)
        attn  = torch.einsum("tk,jt->jk", value, score)
        if return_atten:
            return self.proj(attn).squeeze(0), score
        else:
            return self.proj(attn).squeeze(0)

    def time_sync_decoding(self, enc_out: torch.Tensor) -> List[Hypothesis]:
        """Time synchronous beam search implementation.

        Based on https://ieeexplore.ieee.org/document/9053040

        Args:
            enc_out: Encoder output sequence. (T, D)

        Returns:
            nbest_hyps: N-best hypothesis.

        """
        beam = min(self.beam_size, self.vocab_size)

        beam_state = self.decoder.init_state(beam)

        B = [
            Hypothesis(
                yseq=[self.blank_id],
                score=0.0,
                dec_state=self.decoder.select_state(beam_state, 0),
            )
        ]
        cache = {}

        if self.use_lm:
            B[0].lm_state = self.lm.zero_state()

        for enc_out_t in enc_out:
            A = []
            C = B

            enc_out_t = enc_out_t.unsqueeze(0)

            for v in range(self.max_sym_exp):
                D = []

                beam_dec_out, beam_state, beam_lm_tokens = self.decoder.batch_score(
                    C,
                    beam_state,
                    cache,
                    self.use_lm,
                )

                beam_logp = torch.log_softmax(
                    self.joint_network(enc_out_t, beam_dec_out),
                    dim=-1,
                )
                beam_topk = beam_logp[:, 1:].topk(beam, dim=-1)

                seq_A = [h.yseq for h in A]

                for i, hyp in enumerate(C):
                    if hyp.yseq not in seq_A:
                        A.append(
                            Hypothesis(
                                score=(hyp.score + float(beam_logp[i, 0])),
                                yseq=hyp.yseq[:],
                                dec_state=hyp.dec_state,
                                lm_state=hyp.lm_state,
                            )
                        )
                    else:
                        dict_pos = seq_A.index(hyp.yseq)

                        A[dict_pos].score = np.logaddexp(
                            A[dict_pos].score, (hyp.score + float(beam_logp[i, 0]))
                        )

                if v < (self.max_sym_exp - 1):
                    if self.use_lm:
                        beam_lm_scores, beam_lm_states = self.lm.batch_score(
                            beam_lm_tokens, [c.lm_state for c in C], None
                        )

                    for i, hyp in enumerate(C):
                        for logp, k in zip(beam_topk[0][i], beam_topk[1][i] + 1):
                            new_hyp = Hypothesis(
                                score=(hyp.score + float(logp)),
                                yseq=(hyp.yseq + [int(k)]),
                                dec_state=self.decoder.select_state(beam_state, i),
                                lm_state=hyp.lm_state,
                            )

                            if self.use_lm:
                                new_hyp.score += self.lm_weight * beam_lm_scores[i, k]
                                new_hyp.lm_state = beam_lm_states[i]

                            D.append(new_hyp)

                if self.score_norm_during:
                    C = sorted(D, key=lambda x: x.score / len(x.yseq), reverse=True)[
                        :beam
                    ]
                else:
                    C = sorted(D, key=lambda x: x.score, reverse=True)[:beam]

            if self.score_norm_during:
                B = sorted(A, key=lambda x: x.score / len(x.yseq), reverse=True)[:beam]
            else:
                B = sorted(A, key=lambda x: x.score, reverse=True)[:beam]

        return self.sort_nbest(B)

    def align_length_sync_decoding(self, enc_out: torch.Tensor) -> List[Hypothesis]:
        """Alignment-length synchronous beam search implementation.

        Based on https://ieeexplore.ieee.org/document/9053040

        Args:
            h: Encoder output sequences. (T, D)

        Returns:
            nbest_hyps: N-best hypothesis.

        """
        beam = min(self.beam_size, self.vocab_size)

        t_max = int(enc_out.size(0))
        u_max = min(self.u_max, (t_max - 1))

        beam_state = self.decoder.init_state(beam)

        B = [
            Hypothesis(
                yseq=[self.blank_id],
                score=0.0,
                dec_state=self.decoder.select_state(beam_state, 0),
            )
        ]
        final = []
        cache = {}

        if self.use_lm:
            B[0].lm_state = self.lm.zero_state()

        for i in range(t_max + u_max):
            A = []

            B_ = []
            B_enc_out = []
            for hyp in B:
                u = len(hyp.yseq) - 1
                t = i - u

                if t > (t_max - 1):
                    continue

                B_.append(hyp)
                B_enc_out.append((t, enc_out[t]))

            if B_:
                beam_dec_out, beam_state, beam_lm_tokens = self.decoder.batch_score(
                    B_,
                    beam_state,
                    cache,
                    self.use_lm,
                )

                beam_enc_out = torch.stack([x[1] for x in B_enc_out])

                beam_logp = torch.log_softmax(
                    self.joint_network(beam_enc_out, beam_dec_out),
                    dim=-1,
                )
                beam_topk = beam_logp[:, 1:].topk(beam, dim=-1)

                if self.use_lm:
                    beam_lm_scores, beam_lm_states = self.lm.batch_score(
                        beam_lm_tokens,
                        [b.lm_state for b in B_],
                        None,
                    )

                for i, hyp in enumerate(B_):
                    new_hyp = Hypothesis(
                        score=(hyp.score + float(beam_logp[i, 0])),
                        yseq=hyp.yseq[:],
                        dec_state=hyp.dec_state,
                        lm_state=hyp.lm_state,
                    )

                    A.append(new_hyp)

                    if B_enc_out[i][0] == (t_max - 1):
                        final.append(new_hyp)

                    for logp, k in zip(beam_topk[0][i], beam_topk[1][i] + 1):
                        new_hyp = Hypothesis(
                            score=(hyp.score + float(logp)),
                            yseq=(hyp.yseq[:] + [int(k)]),
                            dec_state=self.decoder.select_state(beam_state, i),
                            lm_state=hyp.lm_state,
                        )

                        if self.use_lm:
                            new_hyp.score += self.lm_weight * beam_lm_scores[i, k]
                            new_hyp.lm_state = beam_lm_states[i]

                        A.append(new_hyp)

                if self.score_norm_during:
                    B = sorted(A, key=lambda x: x.score / len(x.yseq), reverse=True)[
                        :beam
                    ]
                else:
                    B = sorted(A, key=lambda x: x.score, reverse=True)[:beam]
                B = recombine_hyps(B)

        if final:
            return self.sort_nbest(final)
        else:
            return B

    def nsc_beam_search(self, enc_out: torch.Tensor) -> List[ExtendedHypothesis]:
        """N-step constrained beam search implementation.

        Based on/Modified from https://arxiv.org/pdf/2002.03577.pdf.
        Please reference ESPnet (b-flo, PR #2444) for any usage outside ESPnet
        until further modifications.

        Args:
            enc_out: Encoder output sequence. (T, D_enc)

        Returns:
            nbest_hyps: N-best hypothesis.

        """
        beam = min(self.beam_size, self.vocab_size)
        beam_k = min(beam, (self.vocab_size - 1))

        beam_state = self.decoder.init_state(beam)

        init_tokens = [
            ExtendedHypothesis(
                yseq=[self.blank_id],
                score=0.0,
                dec_state=self.decoder.select_state(beam_state, 0),
            )
        ]

        cache = {}

        beam_dec_out, beam_state, beam_lm_tokens = self.decoder.batch_score(
            init_tokens,
            beam_state,
            cache,
            self.use_lm,
        )

        state = self.decoder.select_state(beam_state, 0)

        if self.use_lm:
            beam_lm_scores, beam_lm_states = self.lm.batch_score(
                beam_lm_tokens,
                [i.lm_state for i in init_tokens],
                None,
            )
            lm_state = beam_lm_states[0]
            lm_scores = beam_lm_scores[0]
        else:
            lm_state = None
            lm_scores = None

        kept_hyps = [
            ExtendedHypothesis(
                yseq=[self.blank_id],
                score=0.0,
                dec_state=state,
                dec_out=[beam_dec_out[0]],
                lm_state=lm_state,
                lm_scores=lm_scores,
            )
        ]

        for enc_out_t in enc_out:
            hyps = self.prefix_search(
                sorted(kept_hyps, key=lambda x: len(x.yseq), reverse=True),
                enc_out_t,
            )
            kept_hyps = []

            beam_enc_out = enc_out_t.unsqueeze(0)

            S = []
            V = []
            for n in range(self.nstep):
                beam_dec_out = torch.stack([hyp.dec_out[-1] for hyp in hyps])

                beam_logp = torch.log_softmax(
                    self.joint_network(beam_enc_out, beam_dec_out),
                    dim=-1,
                )
                beam_topk = beam_logp[:, 1:].topk(beam_k, dim=-1)

                for i, hyp in enumerate(hyps):
                    S.append(
                        ExtendedHypothesis(
                            yseq=hyp.yseq[:],
                            score=hyp.score + float(beam_logp[i, 0:1]),
                            dec_out=hyp.dec_out[:],
                            dec_state=hyp.dec_state,
                            lm_state=hyp.lm_state,
                            lm_scores=hyp.lm_scores,
                        )
                    )

                    for logp, k in zip(beam_topk[0][i], beam_topk[1][i] + 1):
                        score = hyp.score + float(logp)

                        if self.use_lm:
                            score += self.lm_weight * float(hyp.lm_scores[k])

                        V.append(
                            ExtendedHypothesis(
                                yseq=hyp.yseq[:] + [int(k)],
                                score=score,
                                dec_out=hyp.dec_out[:],
                                dec_state=hyp.dec_state,
                                lm_state=hyp.lm_state,
                                lm_scores=hyp.lm_scores,
                            )
                        )

                V.sort(key=lambda x: x.score, reverse=True)
                V = subtract(V, hyps)[:beam]

                beam_state = self.decoder.create_batch_states(
                    beam_state,
                    [v.dec_state for v in V],
                    [v.yseq for v in V],
                )
                beam_dec_out, beam_state, beam_lm_tokens = self.decoder.batch_score(
                    V,
                    beam_state,
                    cache,
                    self.use_lm,
                )

                if self.use_lm:
                    beam_lm_scores, beam_lm_states = self.lm.batch_score(
                        beam_lm_tokens, [v.lm_state for v in V], None
                    )

                if n < (self.nstep - 1):
                    for i, v in enumerate(V):
                        v.dec_out.append(beam_dec_out[i])

                        v.dec_state = self.decoder.select_state(beam_state, i)

                        if self.use_lm:
                            v.lm_state = beam_lm_states[i]
                            v.lm_scores = beam_lm_scores[i]

                    hyps = V[:]
                else:
                    beam_logp = torch.log_softmax(
                        self.joint_network(beam_enc_out, beam_dec_out),
                        dim=-1,
                    )

                    for i, v in enumerate(V):
                        if self.nstep != 1:
                            v.score += float(beam_logp[i, 0])

                        v.dec_out.append(beam_dec_out[i])

                        v.dec_state = self.decoder.select_state(beam_state, i)

                        if self.use_lm:
                            v.lm_state = beam_lm_states[i]
                            v.lm_scores = beam_lm_scores[i]

            kept_hyps = sorted((S + V), key=lambda x: x.score, reverse=True)[:beam]

        return self.sort_nbest(kept_hyps)

    def modified_adaptive_expansion_search(
        self, enc_out: torch.Tensor
    ) -> List[ExtendedHypothesis]:
        """It's the modified Adaptive Expansion Search (mAES) implementation.

        Based on/modified from https://ieeexplore.ieee.org/document/9250505 and NSC.

        Args:
            enc_out: Encoder output sequence. (T, D_enc)

        Returns:
            nbest_hyps: N-best hypothesis.

        """
        beam = min(self.beam_size, self.vocab_size)
        beam_state = self.decoder.init_state(beam)

        init_tokens = [
            ExtendedHypothesis(
                yseq=[self.blank_id],
                score=0.0,
                dec_state=self.decoder.select_state(beam_state, 0),
            )
        ]

        cache = {}

        beam_dec_out, beam_state, beam_lm_tokens = self.decoder.batch_score(
            init_tokens,
            beam_state,
            cache,
            self.use_lm,
        )

        state = self.decoder.select_state(beam_state, 0)

        if self.use_lm:
            beam_lm_scores, beam_lm_states = self.lm.batch_score(
                beam_lm_tokens, [i.lm_state for i in init_tokens], None
            )

            lm_state = beam_lm_states[0]
            lm_scores = beam_lm_scores[0]
        else:
            lm_state = None
            lm_scores = None

        kept_hyps = [
            ExtendedHypothesis(
                yseq=[self.blank_id],
                score=0.0,
                dec_state=state,
                dec_out=[beam_dec_out[0]],
                lm_state=lm_state,
                lm_scores=lm_scores,
            )
        ]

        for enc_out_t in enc_out:
            hyps = self.prefix_search(
                sorted(kept_hyps, key=lambda x: len(x.yseq), reverse=True),
                enc_out_t,
            )
            kept_hyps = []

            beam_enc_out = enc_out_t.unsqueeze(0)

            list_b = []
            duplication_check = [hyp.yseq for hyp in hyps]

            for n in range(self.nstep):
                beam_dec_out = torch.stack([h.dec_out[-1] for h in hyps])

                beam_logp, beam_idx = torch.log_softmax(
                    self.joint_network(beam_enc_out, beam_dec_out),
                    dim=-1,
                ).topk(self.max_candidates, dim=-1)

                k_expansions = select_k_expansions(
                    hyps,
                    beam_idx,
                    beam_logp,
                    self.expansion_gamma,
                )

                list_exp = []
                for i, hyp in enumerate(hyps):
                    for k, new_score in k_expansions[i]:
                        new_hyp = ExtendedHypothesis(
                            yseq=hyp.yseq[:],
                            score=new_score,
                            dec_out=hyp.dec_out[:],
                            dec_state=hyp.dec_state,
                            lm_state=hyp.lm_state,
                            lm_scores=hyp.lm_scores,
                        )

                        if k == 0:
                            list_b.append(new_hyp)
                        else:
                            if new_hyp.yseq + [int(k)] not in duplication_check:
                                new_hyp.yseq.append(int(k))

                                if self.use_lm:
                                    new_hyp.score += self.lm_weight * float(
                                        hyp.lm_scores[k]
                                    )

                                list_exp.append(new_hyp)

                if not list_exp:
                    kept_hyps = sorted(list_b, key=lambda x: x.score, reverse=True)[
                        :beam
                    ]

                    break
                else:
                    beam_state = self.decoder.create_batch_states(
                        beam_state,
                        [hyp.dec_state for hyp in list_exp],
                        [hyp.yseq for hyp in list_exp],
                    )

                    beam_dec_out, beam_state, beam_lm_tokens = self.decoder.batch_score(
                        list_exp,
                        beam_state,
                        cache,
                        self.use_lm,
                    )

                    if self.use_lm:
                        beam_lm_scores, beam_lm_states = self.lm.batch_score(
                            beam_lm_tokens, [k.lm_state for k in list_exp], None
                        )

                    if n < (self.nstep - 1):
                        for i, hyp in enumerate(list_exp):
                            hyp.dec_out.append(beam_dec_out[i])
                            hyp.dec_state = self.decoder.select_state(beam_state, i)

                            if self.use_lm:
                                hyp.lm_state = beam_lm_states[i]
                                hyp.lm_scores = beam_lm_scores[i]

                        hyps = list_exp[:]
                    else:
                        beam_logp = torch.log_softmax(
                            self.joint_network(beam_enc_out, beam_dec_out),
                            dim=-1,
                        )

                        for i, hyp in enumerate(list_exp):
                            hyp.score += float(beam_logp[i, 0])

                            hyp.dec_out.append(beam_dec_out[i])
                            hyp.dec_state = self.decoder.select_state(beam_state, i)

                            if self.use_lm:
                                hyp.lm_states = beam_lm_states[i]
                                hyp.lm_scores = beam_lm_scores[i]

                        kept_hyps = sorted(
                            list_b + list_exp, key=lambda x: x.score, reverse=True
                        )[:beam]

        return self.sort_nbest(kept_hyps)

    def multi_blank_greedy_search(self, enc_out: torch.Tensor) -> List[Hypothesis]:
        """Greedy Search for Multi-Blank Transducer (Multi-Blank Greedy, MBG).

        In this implementation, we assume:
        1. the index of standard blank is the last entry of self.multi_blank_indices
           rather than self.blank_id (to avoid too much change on original transducer)
        2. other entries in self.multi_blank_indices are big blanks that account for
           multiple frames.

        Based on https://arxiv.org/abs/2211.03541

        Args:
            enc_out: Encoder output sequence. (T, D_enc)

        Returns:
            hyp: 1-best hypothesis.

        """

        big_blank_duration = 1
        blank_start = self.multi_blank_indices[0]
        blank_end = self.multi_blank_indices[-1]

        dec_state = self.decoder.init_state(1)
        hyp = Hypothesis(score=0.0, yseq=[blank_end], dec_state=dec_state)
        cache = {}

        for enc_out_t in enc_out:
            # case 1: skip frames until big_blank_duration == 1
            if big_blank_duration > 1:
                big_blank_duration -= 1
                continue

            symbols_added = 0
            while symbols_added <= 3:
                dec_out, state, _ = self.decoder.score(hyp, cache)
                logp = torch.log_softmax(self.joint_network(enc_out_t, dec_out), dim=-1)
                top_logp, k = torch.max(logp, dim=-1)

                # case 2: predict a blank token
                if blank_start <= k <= blank_end:
                    big_blank_duration = self.multi_blank_durations[k - blank_start]
                    hyp.score += top_logp
                    break

                # case 3: predict a non-blank token
                else:
                    symbols_added += 1
                    hyp.yseq.append(int(k))
                    hyp.score += float(top_logp)
                    hyp.dec_state = state

        return [hyp]
