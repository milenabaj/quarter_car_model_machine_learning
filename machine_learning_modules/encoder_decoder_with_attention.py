"""
@author: Milena Bajic (DTU Compute)
Nice explanations of attention:
https://blog.floydhub.com/attention-mechanism/
"""
import sys
import logging
import torch
import random
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR
from quarter_car_model_machine_learning.utils.various_utils import *
from scipy.stats import norm

# Get logger for module
ed_log = get_mogule_logger("encoder_decoder")

class lstm_encoder(nn.Module):
    ''' Encodes time-series sequence '''

    def __init__(self, input_size = 1, hidden_size = 64, num_layers = 1, device = 'cuda',bidirectional = True):

        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.device = device
        self.bidirectional = bidirectional 
        self.ndirs=2 if self.bidirectional else 1
        
        hidden_size_a = hidden_size
        #hidden_size_b = hidden_size_a*2
        hidden_size_last = hidden_size_a *2

        # Define LSTM layers
        #self.lstm_a = nn.LSTM(input_size = input_size, hidden_size = hidden_size_a, num_layers = 1, bidirectional = self.bidirectional)
        #self.lstm_b = nn.LSTM(input_size = self.ndirs *hidden_size_a, hidden_size = hidden_size_b, num_layers = 1, bidirectional = self.bidirectional)
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = 1, bidirectional = self.bidirectional) #last
        self.last_hidden_size = self.ndirs * self.lstm.hidden_size

        # Initialize weights with zeros
        #self.init_hidden_zeros(self.lstm_a)
        self.init_hidden_zeros(self.lstm)
        
    def forward(self, x_input):

        '''
        : param x_input:               input of shape (seq_len, # in batch, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence
        '''

        #self.lstm_out_a, self.hidden_a = self.lstm_a(x_input)
        #self.lstm_out_b, self.hidden_b = self.lstm_b(self.lstm_out_a)
        self.lstm_out, self.hidden = self.lstm(x_input)
     
        # Concat states from the forward and the backward states of the last encoder layer
        if self.bidirectional:
            h0 = self.hidden[0]
            h0_cat = torch.cat( (h0[0],h0[1]), dim=1).unsqueeze(0)
            
            h1 = self.hidden[1]
            h1_cat = torch.cat( (h1[0],h1[1]), dim=1).unsqueeze(0)
    
            self.hidden = (h0_cat, h1_cat)

        return self.lstm_out, self.hidden

    def init_hidden_zeros(self, layer):

        '''
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state
        '''
        torch.nn.init.zeros_(layer.weight_ih_l0)
        torch.nn.init.zeros_(layer.weight_hh_l0)
        if self.bidirectional:
            torch.nn.init.zeros_(layer.weight_ih_l0_reverse)
            torch.nn.init.zeros_(layer.weight_hh_l0_reverse)


class lstm_decoder(nn.Module):

    ''' Decodes hidden state output by encoder '''

    def __init__(self, input_size = 32, encoder_model =  None , output_size = 1, num_layers = 1, device = 'cuda', attn='general'):

        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.encoder_model = encoder_model
        self.num_layers = num_layers
        self.device = device
        self.attn = attn

        # LSTM Layer
        #self.lstm_a = nn.LSTM(input_size = input_size, hidden_size = encoder_model.lstm_a.hidden_size*encoder_model.ndirs, num_layers = num_layers)
        self.lstm = nn.LSTM(input_size =  input_size, hidden_size = encoder_model.lstm.hidden_size * encoder_model.ndirs, num_layers = num_layers)
                                     
        # Attention Layer if general
        if self.attn =='general':
            self.attention_layer = nn.Linear(self.lstm.hidden_size, self.lstm.hidden_size, bias=True)
            torch.nn.init.zeros_(self.attention_layer.weight)
            
        # Final LSTM to Linear Layer
        self.linear = nn.Linear(2*self.lstm.hidden_size, output_size) #2 because the context vector is concatenated with the lstm out
        

    def forward(self, x_input, hidden_states, encoder_output, acc, t):

        '''
        : param x_input:                    should be 2D (batch_size, input_size)
        : param  hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence

        '''
        #x_input = x_input.unsqueeze(0) #(batch_size, input_features) -> (1, batch_size, input_features))
        #print('Decoder forward - lstm input: ',x_input.shape)
        
        # TODO: if acceleration
        # LSTM
        x_input.to(self.device)
        #self.lstm_out_a, self.hidden_a = self.lstm_a(x_input)  
        self.lstm_out, self.hidden = self.lstm(x_input, hidden_states)  
      
        # Dot attention 
        if self.attn =='dot':
            self.scores = torch.bmm(encoder_output.permute(1,0,2),self.lstm_out.permute(1,2,0)) #(batch, input_hidden_size, output_timestep(=1))
            # put batch first to get correct bmm and then put 2D mat. in correct shapes for mm
           
        # In general attention, decoder hidden state is passed through linear layers to introduce a weight matrix             
        elif self.attn =='general':
            decoder_out = self.attention_layer( self.lstm_out.permute(1,0,2).squeeze(1) )# [batch size, hidden dim])
            self.scores = torch.bmm(encoder_output.permute(1,0,2), decoder_out.unsqueeze(2))
            t_min=t-20
            t_max=t+20
            self.scores[:,:t_min,:]=-100
            self.scores[:,t_max:,:]=-100

        # Attention weights after softmax
        self.attn_weights = F.softmax(self.scores, dim=1) #(batch, number of ts, 1)
        self.attn_weights.to(self.device)

         # Context vector
        self.context_vector = torch.bmm(encoder_output.permute(1,2,0), self.attn_weights)

        # Concat. context vector and lstm out along the hidden state dimension
        self.cat = torch.cat( (self.context_vector.permute(2,0,1),self.lstm_out), dim=2)
        self.cat = self.cat.squeeze(0) #-> [batch size, hidden dim]

        # Final Linear layer
        prediction = self.linear(self.cat)
        
        return prediction, self.hidden, self.attn_weights



class lstm_seq2seq_with_attn(nn.Module):
    ''' train LSTM encoder-decoder and make predictions '''

    def __init__(self, input_size  = 1, hidden_size = 1, target_len = 1000, 
                 use_teacher_forcing = True, device = 'cuda', bidirectional = True, attn = 'general'):

        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        '''

        super(lstm_seq2seq_with_attn, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.target_len = target_len
        self.attn = attn
        self.use_teacher_forcing = use_teacher_forcing
        self.device = device
        self.bidirectional = bidirectional         # Bidectional LSTM

        # Encoder
        self.encoder = lstm_encoder(device = self.device, hidden_size = self.hidden_size, bidirectional = self.bidirectional)
        
        # Decoder, decoder input: target sequence, features taken as input hidden state
        self.decoder = lstm_decoder(input_size = input_size, encoder_model =  self.encoder , device = self.device, attn = self.attn)

    def forward(self, input_batch, target_batch = None, teacher_forcing_ratio = None):

        '''
        : param input_batch:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence

        '''
        # Reverse acc
        #input_batch = input_batch.flip(1)
        
        if target_batch is None:
            self.use_teacher_forcing = False # can't use teacher forcing if output sequence is not given
        batch_size = input_batch.shape[1]

        # ======== ENCODER ======== #
        # Pass trough the encoder
        self.encoder_output, self.encoder_hidden = self.encoder(input_batch)

        # ====== DECODER ======= #
        # First decoder input: '0' (1, batch_size, 1)
        # First decoder hidden state: last encoder hidden state (batch_size, input_size)
        self.decoder_input = torch.zeros([1, batch_size, 1]).to(self.device) # start of the output seq.
        self.decoder_hidden = self.encoder_hidden
        #self.decoder_hidden = (torch.zeros(self.encoder_hidden[0].shape), torch.zeros(self.encoder_hidden[1].shape))
        
        # To cuda
        self.decoder_hidden[0].to(self.device)
        self.decoder_hidden[1].to(self.device)

        # Initialize vector to store the decoder output
        self.outputs = torch.zeros([self.target_len,  batch_size, 1]).to(self.device)
        
        #Initialize vector to store the attentions weights (output timestep, input timestep, batch_size)
        self.attention_weights = torch.zeros([self.target_len, self.target_len, batch_size, ]).to(self.device)

        # If asked for teaching forcing, use it in teacher_forcing_ratio% of cases
        use_teacher_forcing = False
        if self.use_teacher_forcing:
            if random.random() < teacher_forcing_ratio:
                use_teacher_forcing = True
        
        # Teacher forcing: Feed the target as the next input
        if use_teacher_forcing:
            for t in range(self.target_len):
                self.decoder_output, self.decoder_hidden, self.attn_weights_ts = self.decoder(self.decoder_input, self.decoder_hidden, self.encoder_output, input_batch, t)
               
                self.decoder_input = target_batch[t,:,:].unsqueeze(0).to(self.device) # current target will be the input in the next timestep
               
                # Save attention weights
                self.attn_weights_ts = self.attn_weights_ts.view(batch_size,self.target_len).permute(1,0)
                self.attention_weights[t] = self.attn_weights_ts
                
                # Save output
                self.outputs[t] = self.decoder_output
                
        else:
            # Without teacher forcing: use its own predictions as the next input
            for t in range(self.target_len):
                self.decoder_output, self.decoder_hidden, self.attn_weights_ts = self.decoder(self.decoder_input, self.decoder_hidden,  self.encoder_output, input_batch, t)
                self.decoder_input = self.decoder_output.unsqueeze(0).to(self.device) # current prediction will be the input in the next timestep
                            
                # Save attention weights
                self.attn_weights_ts = self.attn_weights_ts.view(batch_size,self.target_len).permute(1,0)
                self.attention_weights[t] = self.attn_weights_ts
                
                # Save Output
                self.outputs[t] = self.decoder_output


        return self.outputs


# IDEA: generate gaussian hard coded attention with max in the same point and around a bit
        # mean equal to the point where it is
#plt.plot(scipy.stats.norm(0, 1).pdf(x))