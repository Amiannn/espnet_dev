import os

from local.utils import read_file

root_path = "exp/asr_finetune_freeze_conformer_transducer_contextual_biasing_sche30_rep_suffix/decode_asr_asr_model_valid.loss.ave_10best/test_clean/logdir"

def caluate_exp_time(root_path):
    total_time = 0
    for i in range(10):
        path  = os.path.join(root_path, f'asr_inference.{i + 1}.log')
        texts = [d[0] for d in read_file(path, sp='　')]
        for text in texts:
            if '# Accounting: time=' in text:
                total_time += int(text.split('# Accounting: time=')[-1].split(' ')[0])
    return total_time

if __name__ == '__main__':
    root_path   = "exp/{a}/{b}/test_clean/logdir"
    exp_folders = [
        ['encoder_biasing', 'asr_finetune_freeze_conformer_transducer_contextual_biasing_sche30_rep_suffix'],
        # ['predictor_biasing', 'asr_finetune_freeze_conformer_transducer_contextual_biasing_predictor_proj_suffix'],
    ]
    decode_folders = [
        ['1000', 'decode_asr_asr_model_valid.loss.ave_10best'],
        ['500', 'decode_asr_medium_asr_model_valid.loss.ave_10best'],
        ['100', 'decode_asr_small_asr_model_valid.loss.ave_10best'],
        ['50', 'decode_asr_tinysmall_asr_model_valid.loss.ave_10best'],
        ['10', 'decode_asr_tiny_asr_model_valid.loss.ave_10best'],
    ]

    for exp_tag, exp_folder in exp_folders:
        for dec_tag, decode_folder in decode_folders:
            path = root_path.format(a=exp_folder, b=decode_folder)
            total_time = caluate_exp_time(path)
            print(f'{exp_tag}, {dec_tag}, {total_time}')
            print('_' * 30)
