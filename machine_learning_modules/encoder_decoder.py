"""
@author: Milena Bajic (DTU Compute)

encoder-decoder model for SAME series:
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
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR
from quarter_car_model_machine_learning.utils.various_utils import *

# Get logger for module
ed_log = get_mogule_logger("encoder_decoder")

class lstm_encoder(nn.Module):
    ''' Encodes time-series sequence '''

    def __init__(self, input_size = 1, hidden_size = 64, num_layers = 1, device = 'cuda'):

        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        # define LSTM layer
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers)


    def forward(self, x_input):

        '''
        : param x_input:               input of shape (seq_len, # in batch, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence
        '''

        self.lstm_out, self.hidden = self.lstm(x_input.view(x_input.shape[0], x_input.shape[1], self.input_size))

        #print('Encoder forward - lstm_out: ',self.lstm_out.shape)
        #print('Encoder forward - hidden: ',self.hidden[0].shape)
        return self.lstm_out, self.hidden

    def init_hidden(self, batch_size):

        '''
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state
        '''

        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


class lstm_decoder(nn.Module):

    ''' Decodes hidden state output by encoder '''

    def __init__(self, input_size = 32, hidden_size = 64, output_size = 1, num_layers = 1, device = 'cuda'):

        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x_input, encoder_hidden_states):

        '''
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence

        '''
        #x_input = x_input.unsqueeze(0) #(batch_size, input_features) -> (1, batch_size, input_features))
        #print('Decoder forward - lstm input: ',x_input.shape)
        x_input.to(self.device)
        lstm_out, self.hidden = self.lstm(x_input, encoder_hidden_states)
        #print('Decoder forward - lstm_out: ',lstm_out.shape)
        lstm_out = lstm_out.squeeze(0) #-> [batch size, hidden dim]
        #print('Decoder forward - linear input: ',lstm_out.shape)
        prediction = self.linear(lstm_out)
        #print('Decoder forward - prediction: ',prediction.shape)
        return prediction, self.hidden



class lstm_seq2seq(nn.Module):
    ''' train LSTM encoder-decoder and make predictions '''

    def __init__(self, input_size  = 1, hidden_size = 64, target_len = 1000, 
                 use_teacher_forcing = True, device = 'cuda'):

        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        '''

        super(lstm_seq2seq, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.target_len = target_len
        self.use_teacher_forcing = use_teacher_forcing
        self.device = device

        self.encoder = lstm_encoder(device = self.device, hidden_size = self.hidden_size)
        
        # decoder input: target sequence, features only taken as input hidden state
        self.decoder = lstm_decoder(input_size = input_size, hidden_size = self.hidden_size, 
                                    device = self.device)


    def forward(self, input_batch, target_batch):

        '''
        : param input_batch:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence

        '''

        batch_size = input_batch.shape[1]

        # ======== ENCODER ======== #
        # Initialize hidden state
        encoder_hidden = self.encoder.init_hidden(batch_size)

        # Encoder outputs
        encoder_output, encoder_hidden = self.encoder(input_batch)
        self.encoder_output = encoder_output
        self.encoder_hidden = encoder_hidden
        

        # ====== DDECODER ======= #
        # First decoder input: '0' (1, batch_size, 1)
        # First decoder hidden state: last encoder hidden state (batch_size, input_size)
        decoder_input = torch.zeros([1, batch_size, 1]).to(self.device)
        decoder_hidden = encoder_hidden

        # To cuda
        decoder_hidden[0].to(self.device)
        decoder_hidden[1].to(self.device)

        # Outputs tensor
        outputs = torch.zeros([self.target_len,  batch_size, 1]).to(self.device)

        # Decoder output
        if self.use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for t in range(self.target_len):
                #print('Seq2Seq forward - decoder input: ',decoder_input.shape)
                #print('Seq2Seq forward - decoder hidden input: ',decoder_hidden[0].shape)
                decoder_input.to(self.device)
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[t] = decoder_output
                decoder_input = target_batch[t,:,:].unsqueeze(0) # current target will be the input in the next timestep
                #print('Seq2Seq forward after - decoder input: ',decoder_input.shape)
                #print('Seq2Seq forward after - decoder hidden input: ',decoder_hidden[0].shape)

        else:
            # Without teacher forcing: use its own predictions as the next input
            for t in range(self.target_len):
                #print('Seq2Seq forward - decoder input: ',decoder_input.shape)
                #print('Seq2Seq forward - decoder hidden input: ',decoder_hidden[0].shape)
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[t] = decoder_output
                decoder_input = decoder_output.unsqueeze(0)
                #print('Seq2Seq forward after - decoder input: ',decoder_input.shape)
                #print('Seq2Seq forward after - decoder hidden input: ',decoder_hidden[0].shape)

        return outputs


