"""
Encoder-decoder model for SAME series:
https://github.com/lkulowski/LSTM_encoder_decoder/blob/master/code/lstm_encoder_decoder.py
https://github.com/pytorch/tutorials/blob/master/intermediate_source/seq2seq_translation_tutorial.py
attn:
https://buomsoo-kim.github.io/attention/2020/04/27/Attention-mechanism-21.md/
"""

import sys
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR
from quarter_car_model_machine_learning.utils.various_utils import *

# Get logger for module
log_ed = get_mogule_logger("encoder_decoder")

def print():
    log_ed.debug('mjau')