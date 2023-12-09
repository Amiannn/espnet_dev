#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=zh_train
valid_set=zh_dev
test_sets="zh_test"

asr_config=conf/tuning/train_rnnt_freeze_tcpgen_zh.yaml
# inference_config=conf/decode_asr.yaml
inference_config=conf/decode_asr.yaml
asr_tag=finetune_freeze_conformer_transducer_tcpgen500_deep_sche30_rep_zh

pretrained_model=exp/asr_train_asr_transducer_conformer_e15_linear1024_raw_zh_bpe4500_use_wandbtrue_sp_suffix/valid.loss.ave_10best.pth

CUDA_VISIBLE_DEVICES=0 ./asr.sh \
    --lang zh \
    --ngpu 1 \
    --nj 16 \
    --gpu_inference false \
    --inference_nj 10 \
    --nbpe 4500 \
    --suffixbpe suffix \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --asr_tag ${asr_tag} \
    --inference_asr_model valid.loss.best.pth \
    --biasing true \
    --bpe_train_text "data/${train_set}/text" \
    --asr_args "--use_wandb true" \
    --pretrained_model $pretrained_model \
    --ignore_init_mismatch true \
    "$@"
