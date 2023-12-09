#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_clean_100"
valid_set="dev"
# test_sets="test_clean test_other dev_clean dev_other"
test_sets="test_clean"

asr_config=conf/tuning/train_asr_transducer_conformer_e15_linear1024.yaml
# inference_config=conf/tuning/decode_transducer.yaml
# inference_config=conf/decode_asr.yaml
inference_config=conf/decode_asr_lm.yaml

CUDA_VISIBLE_DEVICES=3 ./asr.sh \
    --lang en \
    --ngpu 1 \
    --nj 8 \
    --gpu_inference false \
    --inference_nj 10 \
    --nbpe 600 \
    --suffixbpe suffix \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm true \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@" \
    --asr_args "--use_wandb true" \
    --inference_asr_model "valid.loss.ave_10best.pth"
