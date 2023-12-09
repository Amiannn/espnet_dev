#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_clean_100"
valid_set="dev"
test_sets="test_clean"

asr_config=conf/tuning/train_rnnt_freeze_contextual_biasing.yaml
inference_config=conf/decode_asr_trie.yaml
asr_tag=finetune_freeze_conformer_transducer_contextual_biasing_proj

pretrained_model=/share/nas165/amian/experiments/speech/tcpgen/espnet/egs2/librispeech_100/asr1/exp/asr_train_asr_transducer_conformer_e15_linear1024_raw_en_bpe600_use_wandbtrue_sp_suffix/valid.loss.ave_10best.pth

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 ./asr.sh \
    --lang en \
    --ngpu 1 \
    --nj 16 \
    --gpu_inference false \
    --inference_nj 10 \
    --nbpe 600 \
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
    --inference_asr_model valid.loss.ave_10best.pth \
    --biasing true \
    --bpe_train_text "data/${train_set}/text" \
    --pretrained_model $pretrained_model \
    --ignore_init_mismatch true \
    "$@"

# --asr_args "--use_wandb true" \