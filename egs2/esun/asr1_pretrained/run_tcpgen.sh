#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=1
stop_stage=13

train_set="train"
valid_set="dev"
test_sets="test"

asr_config=conf/exp/train_rnnt_freeze_tcpgen.yaml
inference_config=conf/exp/decode_tcpgen_medium.yaml

lm_config=conf/exp/train_lm_transformer.yaml
use_lm=false
use_wordlm=false

pretrained_model="/share/nas165/amian/experiments/speech/tcpgen/espnet/egs2/esun/asr1_pretrained/exp/asr_train_asr_transducer_conformer_raw_bpe5000_use_wandbtrue_sp_suffix/valid.loss.ave_10best.pth"

if [ ! -f "data/train/token.man.2" ]; then
    # must preprocess data first to get Mandarin character tokens
    if [ ${stage} -eq 1 ]; then
        ./asr.sh --stage 1 --stop_stage 1
        stage=2
    else
        echo "Error: data/train/token.man.2 does not exist! Run from stage=1 again."
        exit 1
    fi
fi

man_chars=3955
bpe_nlsyms=""

source data/train/token.man.2  # for bpe_nlsyms & man_chars
# nbpe=$((3000 + man_chars + 4))  # 5626
nbpe=5000
# English BPE: 3000 / Mandarin: 2622 / other symbols: 4

CUDA_VISIBLE_DEVICES=0 ./asr.sh \
    --ngpu 1 \
    --nj 1 \
    --gpu_inference false \
    --inference_nj 10 \
    --stage ${stage} \
    --stop_stage ${stop_stage} \
    --nbpe ${nbpe} \
    --suffixbpe suffix \
    --bpe_nlsyms "${bpe_nlsyms}" \
    --use_lm ${use_lm}            \
    --use_word_lm ${use_wordlm}   \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --max_wav_duration 30 \
    --audio_format "flac.ark" \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text.eng.bpe" \
    --score_opts "-e utf-8 -c NOASCII" \
    --asr_args "--use_wandb true" \
    --inference_asr_model valid.loss.ave_10best.pth \
    --biasing true \
    --pretrained_model $pretrained_model \
    --ignore_init_mismatch true \
    "$@"
