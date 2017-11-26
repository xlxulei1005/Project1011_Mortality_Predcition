import torch.optim as optim
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
import logging


class AttentionRNN(nn.Module):
    def __init__(self, config):
        super(AttentionRNN, self).__init__()
        self.batch_size = config['batch_size']
        n_classes = config['target_class']
        output_channel = config['output_channel']
        self.note_gru_hidden = config['note_gru_hidden']
        self.bidirection_gru = config['bidirection_gru']
        self.note_embedding = Convolutional_Embedding(config)
        self.note_gru = nn.GRU(output_channel, self.note_gru_hidden, bidirectional= self.bidirection_gru)
        if self.bidirection_gru:
            self.lin_out = nn.Linear(self.note_gru_hidden * 2, n_classes)
        else:
            self.lin_out = nn.Linear(self.note_gru_hidden, n_classes)
        #self.softmax_note = nn.Softmax()
        
    def forward(self, mini_batch, hidden_state):
        num_of_notes, num_of_words, batch_size = mini_batch.size()
        s = None
        for i in range(num_of_notes):
            _s = self.note_embedding(mini_batch[i,:,:].transpose(0,1))
            if (s is None):
                s = _s.unsqueeze(0)
            else:
                s = torch.cat((s,_s.unsqueeze(0)),0)
        #print('CNN: ', s.size())
                
        #packed_s = torch.nn.utils.rnn.pack_padded_sequence(s, length, batch_first=True)
        
        # (seq_len, batch, input_size),  (num_layers * num_directions, batch, hidden_size)
        out_note, _ =  self.note_gru(s, hidden_state) 
        #print('RNN: ',out_note.size())
        x = out_note[-1,:,:].squeeze()
        x = self.lin_out(x)
        return F.softmax(x)
    
    def init_hidden(self):
        if self.bidirection_gru == True:
            return Variable(torch.zeros(2, self.batch_size, self.note_gru_hidden))
        else:
            return Variable(torch.zeros(1, self.batch_size, self.note_gru_hidden))

        
class Convolutional_Embedding(nn.Module):
    def __init__(self, config):
        super(Convolutional_Embedding, self).__init__()
        words_dim = config['words_dim']
        self.embed_mode = config['embed_mode']
        
        output_channel = config['output_channel'] #
        filter_width = config['filter_width']
        
        
        vocab_size = config['vocab_size']
        self.word_embed = nn.Embedding(vocab_size, words_dim)
        #self.static_word_embed = nn.Embedding(vocab_size, words_dim)
        #self.nonstatic_word_embed = nn.Embedding(vocab_size, words_dim)

        #self.static_word_embed.weight.requires_grad = False
 
        input_channel = 1
    
        self.conv= nn.Conv2d(input_channel, output_channel, (filter_width, words_dim), padding=(filter_width - 1, 0))
        self.dropout = nn.Dropout(config['dropout'])
        
        #n_hidden = output_channel 

        #self.combined_feature_vector = nn.Linear(n_hidden, n_hidden_conv )
        #self.hidden = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        if self.embed_mode == 'random':
            x = self.word_embed(x) 
        x = x.unsqueeze(1)
        x = self.conv(x)
        #print(x.size())
        x = F.tanh(x).squeeze(3)
        #print(x.size())
        x = F.max_pool1d(x, x.size(2))
        #print(x.size())
        x = x.squeeze(2)  # max-over-time pooling
        # append external features and feed to fc
        #x = F.tanh(self.combined_feature_vector(x))
        #x = self.dropout(x)
        #x = self.hidden(x)
        #print(x.size())
        return x

