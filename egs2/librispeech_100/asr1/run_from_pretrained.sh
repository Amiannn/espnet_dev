MODEL_PATH="pyf98/librispeech_100_transducer_conformer"

./run_original.sh \
    --skip_data_prep true  \
    --skip_train true      \
    --download_model $MODEL_PATH