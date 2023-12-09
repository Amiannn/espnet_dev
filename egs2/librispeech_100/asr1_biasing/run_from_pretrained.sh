MODEL_PATH="espnet/guangzhisun_librispeech100_asr_train_conformer_transducer_tcpgen500_deep_sche30_GCN6L_rep_suffix"

./run_original.sh                \
    --skip_data_prep true        \
    --skip_train true            \
    --download_model $MODEL_PATH
