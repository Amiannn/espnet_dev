import os
import torch

MODEL_PATH = "/share/nas165/amian/experiments/speech/tcpgen/espnet/egs2/librispeech_100/asr1/exp/pyf98/librispeech_100_transducer_conformer/valid.loss.ave_10best.pth"

ckpt = torch.load(MODEL_PATH)
