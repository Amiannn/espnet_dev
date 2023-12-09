#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=zh_train
valid_set=zh_dev
test_sets="zh_test"

asr_config=conf/exp/train_asr_transducer_conformer_e15_linear1024.yaml
inference_config=conf/exp/decode_asr.yaml

lm_config=conf/train_lm_transformer.yaml
use_lm=false
use_wordlm=false

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"

CUDA_VISIBLE_DEVICES=0 ./asr.sh \
    --nj 32 \
    --inference_nj 32 \
    --gpu_inference false \
    --ngpu 1 \
    --lang zh \
    --nbpe 4500 \
    --suffixbpe suffix \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm ${use_lm}                                 \
    --use_word_lm ${use_wordlm}                        \
    --lm_config "${lm_config}"                         \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --asr_args "--use_wandb true" \
    --inference_asr_model "valid.loss.ave_10best.pth" \
    --lm_fold_length 150 \
    --lm_train_text "data/${train_set}/text" "$@"
