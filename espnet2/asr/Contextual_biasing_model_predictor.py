import logging
import math
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.layers.gnn import GCN
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.transducer.error_calculator import ErrorCalculatorTransducer
from espnet2.asr_transducer.utils import get_transducer_task_io
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.text.Butils import BiasProc
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy, to_device
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (  # noqa: H301
    LabelSmoothingLoss,
)

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetContextualBiasingPredictorASRModel(ESPnetASRModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        decoder: Optional[AbsDecoder],
        ctc: CTC,
        joint_network: Optional[torch.nn.Module],
        aux_ctc: dict = None,
        ctc_weight: float = 0.5,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        transducer_multi_blank_durations: List = [],
        transducer_multi_blank_sigma: float = 0.05,
        # In a regular ESPnet recipe, <sos> and <eos> are both "<sos/eos>"
        # Pretrained HF Tokenizer needs custom sym_sos and sym_eos
        sym_sos: str = "<sos/eos>",
        sym_eos: str = "<sos/eos>",
        extract_feats_in_collect_stats: bool = True,
        lang_token_id: int = -1,
        biasing: bool = False,
        biasingsche: int = 0,
        battndim: int = 0,
        deepbiasing: bool = False,
        biasingGNN: str = "",
        biasinglist: str = "",
        bmaxlen: int = 100,
        bdrop: float = 0.0,
        bpemodel: str = "",
        freeze_enc_dec: bool = False,
        biasing_type: str = "contextualbiasing",
    ):
        super().__init__(
            vocab_size,
            token_list,
            frontend,
            specaug,
            normalize,
            preencoder,
            encoder,
            postencoder,
            decoder,
            ctc,
            joint_network,
            aux_ctc,
            ctc_weight,
            interctc_weight,
            ignore_id,
            lsm_weight,
            length_normalized_loss,
            report_cer,
            report_wer,
            sym_space,
            sym_blank,
            transducer_multi_blank_durations,
            transducer_multi_blank_sigma,
            sym_sos,
            sym_eos,
            extract_feats_in_collect_stats,
            lang_token_id,
        )
        # biasing
        if biasinglist != "":
            self.bprocessor = BiasProc(
                biasinglist,
                maxlen=bmaxlen,
                bdrop=bdrop,
                bpemodel=bpemodel,
                charlist=token_list,
            )
        self.biasing      = biasing
        self.biasing_type = biasing_type
        if self.biasing:
            self.attndim = battndim
            self.Qproj_semantic = torch.nn.Linear(
                self.encoder.output_size(),
                self.attndim,
            )
            self.Kproj   = torch.nn.Linear(self.decoder.dunits, self.attndim)
            self.Vproj   = torch.nn.Linear(self.decoder.dunits, self.attndim)
            self.proj    = torch.nn.Linear(self.attndim, self.joint_network.joint_space_size)
            self.ooKBemb = torch.nn.Embedding(1, self.decoder.dunits)
            self.CbRNN   = torch.nn.LSTM(
                self.decoder.dunits, 
                self.attndim // 2, 
                1, 
                batch_first=True, 
                bidirectional=True
            )
            self.Bdrop   = torch.nn.Dropout(0.1)

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
            kwargs: "utt_id" is among the input.
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]
        
        if self.biasing:
            (
                biasingwords, 
                lextree, 
                cb_tokens, 
                cb_tokens_len
            ) = self.bprocessor.select_biasing_words(
                text.tolist(), 
                cb=True
            )
        
        text[text == -1] = self.ignore_id

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        loss_transducer, cer_transducer, wer_transducer = None, None, None
        stats = dict()

        # 1. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        # Intermediate CTC (optional)
        loss_interctc = 0.0
        if self.interctc_weight != 0.0 and intermediate_outs is not None:
            for layer_idx, intermediate_out in intermediate_outs:
                # we assume intermediate_out has the same length & padding
                # as those of encoder_out

                # use auxillary ctc data if specified
                loss_ic = None
                if self.aux_ctc is not None:
                    idx_key = str(layer_idx)
                    if idx_key in self.aux_ctc:
                        aux_data_key = self.aux_ctc[idx_key]
                        aux_data_tensor = kwargs.get(aux_data_key, None)
                        aux_data_lengths = kwargs.get(aux_data_key + "_lengths", None)

                        if aux_data_tensor is not None and aux_data_lengths is not None:
                            loss_ic, cer_ic = self._calc_ctc_loss(
                                intermediate_out,
                                encoder_out_lens,
                                aux_data_tensor,
                                aux_data_lengths,
                            )
                        else:
                            raise Exception(
                                "Aux. CTC tasks were specified but no data was found"
                            )
                if loss_ic is None:
                    loss_ic, cer_ic = self._calc_ctc_loss(
                        intermediate_out, encoder_out_lens, text, text_lengths
                    )
                loss_interctc = loss_interctc + loss_ic

                # Collect Intermedaite CTC stats
                stats["loss_interctc_layer{}".format(layer_idx)] = (
                    loss_ic.detach() if loss_ic is not None else None
                )
                stats["cer_interctc_layer{}".format(layer_idx)] = cer_ic

            loss_interctc = loss_interctc / len(intermediate_outs)

            # calculate whole encoder loss
            loss_ctc = (
                1 - self.interctc_weight
            ) * loss_ctc + self.interctc_weight * loss_interctc

        if self.use_transducer_decoder:
            # 2a. Transducer decoder branch
            (
                loss_transducer,
                cer_transducer,
                wer_transducer,
            ) = self._calc_transducer_loss(
                encoder_out,
                encoder_out_lens,
                text,
                cb_tokens,
                cb_tokens_len
            )

            if loss_ctc is not None:
                loss = loss_transducer + (self.ctc_weight * loss_ctc)
            else:
                loss = loss_transducer

            # Collect Transducer branch stats
            stats["loss_transducer"] = (
                loss_transducer.detach() if loss_transducer is not None else None
            )
            stats["cer_transducer"] = cer_transducer
            stats["wer_transducer"] = wer_transducer

        else:
            # 2b. Attention decoder branch
            if self.ctc_weight != 1.0:
                loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )

            # 3. CTC-Att loss definition
            if self.ctc_weight == 0.0:
                loss = loss_att
            elif self.ctc_weight == 1.0:
                loss = loss_ctc
            else:
                loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

            # Collect Attn branch stats
            stats["loss_att"] = loss_att.detach() if loss_att is not None else None
            stats["acc"] = acc_att
            stats["cer"] = cer_att
            stats["wer"] = wer_att

        # Collect total loss stats
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _calc_transducer_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        labels: torch.Tensor,
        cb_tokens: torch.Tensor,
        cb_tokens_len: torch.Tensor,
    ):
        """Compute Transducer loss.

        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc)
            encoder_out_lens: Encoder output sequences lengths. (B,)
            labels: Label ID sequences. (B, L)

        Return:
            loss_transducer: Transducer loss value.
            cer_transducer: Character error rate for Transducer.
            wer_transducer: Word Error Rate for Transducer.

        """
        decoder_in, target, t_len, u_len = get_transducer_task_io(
            labels,
            encoder_out_lens,
            ignore_id=self.ignore_id,
            blank_id=self.blank_id,
        )

        self.decoder.set_device(encoder_out.device)
        decoder_out = self.decoder(decoder_in)

        # 1.2 Semantic biasing
        if self.biasing:
            cb_tokens     = cb_tokens.to(decoder_out.device)
            cb_tokens_len = cb_tokens_len.to(decoder_out.device)
            embed_matrix  = torch.cat(
                [self.decoder.embed.weight.data, self.ooKBemb.weight], dim=0
            )
            cb_tokens_embed = embed_matrix[cb_tokens]
            cb_seq_embed, _ = self.CbRNN(cb_tokens_embed)
            cb_embed        = torch.mean(cb_seq_embed, dim=1)
            
            sem_bias, atten = self.get_semantic_biasing_vector(decoder_out, cb_embed, return_atten=True)
            # decoder_out     = decoder_out + sem_bias

        lin_encoder_out = self.joint_network.lin_enc(encoder_out)
        lin_decoder_out = self.joint_network.lin_dec(decoder_out)
        lin_decoder_out = lin_decoder_out + sem_bias

        joint_out = self.joint_network.joint_activation(
            lin_encoder_out.unsqueeze(2) + lin_decoder_out.unsqueeze(1)
        )
        joint_out = self.joint_network.lin_out(joint_out)

        # joint_out, _ = self.joint_network(
        #     encoder_out.unsqueeze(2), decoder_out.unsqueeze(1)
        # )

        loss_transducer = self.criterion_transducer(
            joint_out,
            target,
            t_len,
            u_len,
        )

        cer_transducer, wer_transducer = None, None
        if not self.training and self.error_calculator_trans is not None:
            cer_transducer, wer_transducer = self.error_calculator_trans(
                encoder_out, target
            )

        return loss_transducer, cer_transducer, wer_transducer

    def get_semantic_biasing_vector(
        self,
        decoder_out,
        biasing_embed,
        return_atten=False
    ):
        query = self.Qproj_semantic(decoder_out)
        key   = self.Bdrop(self.Kproj(biasing_embed))
        value = self.Vproj(biasing_embed)
        # attention
        qk    = torch.einsum("tk,ijk->ijt", key, query)
        qk    = qk / math.sqrt(query.size(-1))
        score = torch.nn.functional.softmax(qk, dim=-1)
        attn  = torch.einsum("tk,ijt->ijk", value, score)
        if return_atten:
            return self.proj(attn), score
        else:
            return self.proj(attn)