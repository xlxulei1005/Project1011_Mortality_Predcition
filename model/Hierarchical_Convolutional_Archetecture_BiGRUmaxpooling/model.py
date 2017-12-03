"""
Nan Wu
All rights reserved
Report bugs to Nan Wu nw1045@nyu.edu
"""
import torch.optim as optim
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
import logging



def batch_matmul_bias(seq, weight, bias, nonlinearity=''):
    s = None
    print(weight.size(),  seq.size())

    bias_dim = bias.size()
    for i in range(seq.size(0)):
        _s = torch.matmul(seq[i], weight) 
        _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).transpose(0,1)
        if(nonlinearity=='tanh'):
            _s_bias = torch.tanh(_s_bias)
        _s_bias = _s_bias.unsqueeze(0)
        if(s is None):
            s = _s_bias
        else:
            s = torch.cat((s,_s_bias),0)
    return s

def batch_matmul(seq, weight, nonlinearity=''):
    s = None
    for i in range(seq.size(0)):
        _s = torch.matmul(seq[i], weight)
        if (nonlinearity=='tanh'):
            _s = torch.tanh(_s)
        _s = _s.unsqueeze(0)
        if(s is None):
            s = _s
        else:
            s = torch.cat((s,_s),0)
    return s.squeeze(2)

def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if(attn_vectors is None):
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors,h_i),0)
    return torch.sum(attn_vectors, 0)

class Hierachical_BiGRU_max(nn.Module):
    def __init__(self, config):
        super(Hierachical_BiGRU_max, self).__init__()
        self.batch_size = config['batch_size']
        n_classes = config['target_class']
        
        self.bigru_max_sub_hidden = config['bigru_max_sub_hidden']
        self.bigru_max_note_hidden = config['bigru_max_note_hidden']

        words_dim = config['words_dim']
        self.embed_mode = config['embed_mode']
        
        vocab_size = config['vocab_size']
        self.word_embed = nn.Embedding(vocab_size, words_dim)

        self.note_bigru = nn.GRU(words_dim, self.bigru_max_note_hidden, bidirectional= True )
        
        self.subject_gru = nn.GRU(2*self.bigru_max_note_hidden, self.bigru_max_sub_hidden, bidirectional= True)
        

        self.lin_out = nn.Linear(self.bigru_max_sub_hidden * 2, n_classes)

        self.softmax_note = nn.Softmax()
        
    def forward(self, mini_batch, hidden_state_note, hidden_state_sub):
        num_of_notes, num_of_words, batch_size = mini_batch.size()
        s = None
        for i in range(num_of_notes):

            if self.embed_mode == 'random':
                x = self.word_embed(mini_batch[i,:,:].transpose(0,1)) 
            #print(x.size())
            #x = x.tran
            #print(x.size())
            x, hidden_state_note = self.note_bigru(x.transpose(0,1), hidden_state_note)
            #print(x.size())
            x = x.transpose(0,1).transpose(1,2)
            _s = F.max_pool1d(x, x.size(2)).squeeze(2)
            #print(_s.size())
            if (s is None):
                s = _s.unsqueeze(0)
                #print(s.size())
            else:
                s = torch.cat((s,_s.unsqueeze(0)),0)

        
        out_note, _ =  self.subject_gru(s, hidden_state_sub)
        out_note = out_note.transpose(0,1).transpose(1,2) 
        note_embedding = F.max_pool1d(out_note, out_note.size(2)).squeeze(2)
        
        #x = self.lin_out(note_embedding)

        return self.lin_out(note_embedding)
    
    def init_hidden(self):
        return Variable(torch.zeros(2, self.batch_size, self.bigru_max_note_hidden)), Variable(torch.zeros(2, self.batch_size, self.bigru_max_sub_hidden))
               

class AttentionRNN(nn.Module):
    def __init__(self, config):
        super(AttentionRNN, self).__init__()
        self.attention = config['attention']
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
        self.softmax_note = nn.Softmax()
        if self.attention == True:
            #self.weight_W_note = nn.Parameter(torch.Tensor(2* note_gru_hidden ,2* note_gru_hidden))
            self.weight_W_note = nn.Parameter(torch.randn(2* self.note_gru_hidden ,2* self.note_gru_hidden))
            #self.bias_note = nn.Parameter(torch.Tensor(2* note_gru_hidden,1))
            self.bias_note = nn.Parameter(torch.randn(2* self.note_gru_hidden,1))
            #self.weight_proj_note = nn.Parameter(torch.Tensor(2* note_gru_hidden, 1))
            self.weight_proj_note = nn.Parameter(torch.randn(2* self.note_gru_hidden, 1))
        
    def forward(self, mini_batch, hidden_state):
        num_of_notes, num_of_words, batch_size = mini_batch.size()
        s = None
        for i in range(num_of_notes):
            _s = self.note_embedding(mini_batch[i,:,:].transpose(0,1))
            if (s is None):
                s = _s.unsqueeze(0)
            else:
                s = torch.cat((s,_s.unsqueeze(0)),0)
        
        out_note, _ =  self.note_gru(s, hidden_state) 

        if self.attention:          
            #print("-------------------------weight_W_note")
            #print(self.weight_W_note.size())
            #print(" -------------------------bias_note")
            #print(self.bias_note.size())
            note_squish = batch_matmul_bias(out_note, self.weight_W_note, self.bias_note, nonlinearity='tanh')
            #print("-----------------------------note_squish")
            #print(note_squish.size())
            #print("----------------------weight_proj_note")
            #print(self.weight_proj_note.size())
            note_attn = batch_matmul(note_squish, self.weight_proj_note)
            #note_attn = note_attn.unsqueeze(0)
            #print("----------------------------note_attn")
            #print(note_attn.size())
            note_attn_norm = self.softmax_note(note_attn.transpose(1,0))    
            #print("-----------------------------note_attn_norm")
            #print(note_attn_norm)
            note_attn_vectors = attention_mul(out_note, note_attn_norm.transpose(1,0))
            #print("------------------------------note_attn_vectors")
            #print(note_attn_vectors)
            
            # size note_attn_vectors: batch size x note_hidden * num_directions           
            final_map = self.lin_out(note_attn_vectors)
            #print(final_map)
            return out_note, note_attn_norm, note_attn_vectors, final_map

        else:
            x = out_note[-1,:,:].squeeze()
            x = self.lin_out(x)

            return x
    
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

