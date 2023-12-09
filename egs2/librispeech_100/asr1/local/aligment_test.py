import soundfile

from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_align import CTCSegmentation

d = ModelDownloader(cachedir="./modelcache")
wsjmodel = d.download_and_unpack("kamo-naoyuki/wsj")

asr_train_config_path = "/share/nas165/amian/experiments/speech/tcpgen/espnet/egs2/librispeech_100/asr1/exp/pyf98/librispeech_100_transducer_conformer/config.yaml"
asr_model_file_path   = "/share/nas165/amian/experiments/speech/tcpgen/espnet/egs2/librispeech_100/asr1/exp/pyf98/librispeech_100_transducer_conformer/valid.loss.ave_10best.pth"

speech, rate = soundfile.read("/share/corpus/LibriSpeech/LibriSpeech/test-clean/1089/134686/1089-134686-0003.flac")

# aligner = CTCSegmentation( 
#     asr_train_config=asr_train_config_path,
#     asr_model_file=asr_model_file_path,
#     kaldi_style_text=False 
# )
aligner = CTCSegmentation( 
    **wsjmodel,
    kaldi_style_text=False 
)
print(aligner)

text = "HELLO BERTIE ANY GOOD IN YOUR MIND".split(' ')


segments = aligner(speech, text)
print(segments)