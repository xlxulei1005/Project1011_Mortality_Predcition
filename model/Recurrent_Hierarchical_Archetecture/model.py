"""
Layers of Recurrent Hierarchical model.
Sheng Liu
All rights reserved
Report bugs to ShengLiu shengliu@nyu.edu
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
import training_tools
import model
import logging

def batch_matmul_bias(seq, weight, bias, nonlinearity=''):
	s = None
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

class AttentionWordRNN(nn.Module):
	
	
	def __init__(self, batch_size, num_tokens, embed_size, word_gru_hidden, bidirectional= True):        
		
		super(AttentionWordRNN, self).__init__()
		
		self.batch_size = batch_size
		self.num_tokens = num_tokens
		self.embed_size = embed_size
		self.word_gru_hidden = word_gru_hidden
		self.bidirectional = bidirectional
		
		self.lookup = nn.Embedding(num_tokens, embed_size)
		if bidirectional == True:
			self.word_gru = nn.GRU(embed_size, word_gru_hidden, bidirectional= True)
			self.weight_W_word = nn.Parameter(torch.randn(2* word_gru_hidden,2*word_gru_hidden))
			self.bias_word = nn.Parameter(torch.randn(2* word_gru_hidden,1))
			self.weight_proj_word = nn.Parameter(torch.randn(2* word_gru_hidden, 1))
		else:
			self.word_gru = nn.GRU(embed_size, word_gru_hidden, bidirectional= False)
			self.weight_W_word = nn.Parameter(torch.Tensor(word_gru_hidden, word_gru_hidden))
			self.bias_word = nn.Parameter(torch.Tensor(word_gru_hidden,1))
			self.weight_proj_word = nn.Parameter(torch.Tensor(word_gru_hidden, 1))
			
		self.softmax_word = nn.Softmax()
		self.weight_W_word.data.uniform_(-0.1, 0.1)
		self.weight_proj_word.data.uniform_(-0.1,0.1)

		
		
	def forward(self, embed, state_word):
		# embeddings
		embedded = self.lookup(embed)
		# word level gru
		output_word, state_word = self.word_gru(embedded, state_word)
		word_squish = batch_matmul_bias(output_word, self.weight_W_word,self.bias_word, nonlinearity='')
		#print("-----------------------------word_squish")
		#print(word_squish)
		word_attn = batch_matmul(word_squish, self.weight_proj_word)
		#print("-----------------------------weight")
		#print(self.weight_proj_word)
		#print("-----------------------------word_attn")
		#print(word_attn)
		word_attn_norm = self.softmax_word(word_attn.transpose(1,0))
		#print("-----------------------------word_attn_norm")
		#print(word_attn_norm)
		word_attn_vectors = attention_mul(output_word, word_attn_norm.transpose(1,0))        
		#print("-----------------------------word_attn_vectors")
		#print(word_attn_vectors)
		return word_attn_vectors, state_word, word_attn_norm
	
	def init_hidden(self):
		if self.bidirectional == True:
			return Variable(torch.zeros(2, self.batch_size, self.word_gru_hidden))
		else:
			return Variable(torch.zeros(1, self.batch_size, self.word_gru_hidden))  

class AttentionSentRNN(nn.Module):
	
	
	def __init__(self, batch_size, sent_gru_hidden, word_gru_hidden,  bidirectional= True):        
		
		super(AttentionSentRNN, self).__init__()
		
		self.batch_size = batch_size
		self.sent_gru_hidden = sent_gru_hidden
		self.word_gru_hidden = word_gru_hidden
		self.bidirectional = bidirectional
		
		
		if bidirectional == True:
			self.sent_gru = nn.GRU(2 * word_gru_hidden, sent_gru_hidden, bidirectional= True)        
			#self.weight_W_sent = nn.Parameter(torch.Tensor(2* sent_gru_hidden ,2* sent_gru_hidden))
			self.weight_W_sent = nn.Parameter(torch.randn(2* sent_gru_hidden ,2* sent_gru_hidden))
			#self.bias_sent = nn.Parameter(torch.Tensor(2* sent_gru_hidden,1))
			self.bias_sent = nn.Parameter(torch.randn(2* sent_gru_hidden,1))
			#self.weight_proj_sent = nn.Parameter(torch.Tensor(2* sent_gru_hidden, 1))
			self.weight_proj_sent = nn.Parameter(torch.randn(2* sent_gru_hidden, 1))
			#self.final_linear = nn.Linear(2* sent_gru_hidden, n_classes)
		else:
			self.sent_gru = nn.GRU(word_gru_hidden, sent_gru_hidden, bidirectional= True)        
			self.weight_W_sent = nn.Parameter(torch.Tensor(sent_gru_hidden ,sent_gru_hidden))
			self.bias_sent = nn.Parameter(torch.Tensor(sent_gru_hidden,1))
			self.weight_proj_sent = nn.Parameter(torch.Tensor(sent_gru_hidden, 1))
			#self.final_linear = nn.Linear(sent_gru_hidden, n_classes)
		self.softmax_sent = nn.Softmax()
		self.final_softmax = nn.Softmax()
		self.weight_W_sent.data.uniform_(-0.1, 0.1)
		self.weight_proj_sent.data.uniform_(-0.1,0.1)
		
		
	def forward(self, word_attention_vectors, state_sent):
		output_sent, state_sent = self.sent_gru(word_attention_vectors, state_sent)   
		#print("----------------------output_sent")
		#print(output_sent)
		sent_squish = batch_matmul_bias(output_sent, self.weight_W_sent,self.bias_sent, nonlinearity='tanh')
		#print("----------------------sent_squish")
		#print(sent_squish)
		sent_attn = batch_matmul(sent_squish, self.weight_proj_sent)
		#print("----------------------sent_attn")
		#print(sent_attn)
		sent_attn_norm = self.softmax_sent(sent_attn.transpose(1,0))
		#print("----------------------sent_attn_norm")
		#print(sent_attn_norm)
		sent_attn_vectors = attention_mul(output_sent, sent_attn_norm.transpose(1,0)) 
		#print("----------------------sent_attn_vectors")
		#print(sent_attn_vectors)
		# final classifier
		#final_map = self.final_linear(sent_attn_vectors.squeeze(0))
		return sent_attn_vectors, state_sent, sent_attn_norm
		#F.log_softmax(final_map), state_sent, sent_attn_norm
	
	def init_hidden(self):
		if self.bidirectional == True:
			return Variable(torch.zeros(2, self.batch_size, self.sent_gru_hidden))
		else:
			return Variable(torch.zeros(1, self.batch_size, self.sent_gru_hidden)) 

class AttentionNoteRNN(nn.Module):

	def __init__(self, batch_size, num_tokens, embed_size, word_gru_hidden, sent_gru_hidden, note_gru_hidden, n_classes= 2, note_attention = True):        
		
		super(AttentionNoteRNN, self).__init__()
		self.word_attention =  AttentionWordRNN(batch_size, num_tokens, embed_size,\
												word_gru_hidden, bidirectional= True) 
		self.sent_attention = AttentionSentRNN(batch_size, sent_gru_hidden, \
											   word_gru_hidden, bidirectional= True)
		self.attention = note_attention
		self.batch_size = batch_size
		self.word_gru_hidden = word_gru_hidden
		self.sent_gru_hidden = sent_gru_hidden
		self.note_gru_hidden = note_gru_hidden
		if self.attention == True:
			self.note_gru = nn.GRU(2 * sent_gru_hidden, note_gru_hidden, bidirectional= True)
			#self.weight_W_note = nn.Parameter(torch.Tensor(2* note_gru_hidden ,2* note_gru_hidden))
			self.weight_W_note = nn.Parameter(torch.randn(2* note_gru_hidden ,2* note_gru_hidden))
			#self.bias_note = nn.Parameter(torch.Tensor(2* note_gru_hidden,1))
			self.bias_note = nn.Parameter(torch.randn(2* note_gru_hidden,1))
			#self.weight_proj_note = nn.Parameter(torch.Tensor(2* note_gru_hidden, 1))
			self.weight_proj_note = nn.Parameter(torch.randn(2* note_gru_hidden, 1))
			self.final_linear = nn.Linear(2* note_gru_hidden, n_classes)
		else:
			self.note_gru = nn.GRU(sent_gru_hidden, note_gru_hidden, bidirectional= False)
			self.weight_W_note = nn.Parameter(torch.Tensor(note_gru_hidden ,2* note_gru_hidden))
			self.bias_note = nn.Parameter(torch.Tensor(note_gru_hidden,1))
			self.weight_proj_note = nn.Parameter(torch.Tensor(note_gru_hidden, 1))
			self.final_linear = nn.Linear(note_gru_hidden, n_classes)
		self.softmax_note = nn.Softmax()
		self.weight_W_note.data.uniform_(-0.1, 0.1)
		self.weight_proj_note.data.uniform_(-0.1,0.1)
		#self.state_word = self.word_attention.init_hidden()
		#self.state_sent = self.sent_attention.init_hidden()
		
		


	def forward(self, mini_batch,state_word,state_sent,state_note):
		num_of_notes,num_of_sents,_,_ = mini_batch.size()
		n = None
		for i in range(num_of_notes):
			s = None
			for j in range(num_of_sents):
				_s, state_word, _ = self.word_attention(mini_batch[i,j,:,:].transpose(0,1), state_word)
				if (s is None):
					s = _s.unsqueeze(0)
				else:
					s = torch.cat((s,_s.unsqueeze(0)),0)               
			# s size: num_of_sents x batch_size x (word_hidden*2)
			_n, state_sent, _ = self.sent_attention(s, state_sent)
			_n = _n.unsqueeze(0)            
			if (n is None):
				n = _n
			else:
				n = torch.cat((n,_n),0)
			# n size: num_of_notes x batch size x sent_hidden * num_directions            
		out_note,state_note =  self.note_gru(n,state_note)
		#print("------------------------------out_note")
		#print(out_note.size())
		
		#out_note size : num_of_notes, batch size, note_hidden * num_directions
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
			final_map = self.final_linear(note_attn_vectors)
			#print(final_map)
			return out_note, note_attn_norm, note_attn_vectors, final_map
		else:
			return out_note
		
	def init_hidden(self):
		if self.attention == True:
			return Variable(torch.zeros(2, self.batch_size, self.word_gru_hidden)),\
		Variable(torch.zeros(2, self.batch_size, self.sent_gru_hidden)),\
		Variable(torch.zeros(2, self.batch_size, self.note_gru_hidden))
		else:
			return Variable(torch.zeros(1, self.batch_size, self.word_gru_hidden)),\
		Variable(torch.zeros(1, self.batch_size, self.sent_gru_hidden)),\
		Variable(torch.zeros(1, self.batch_size, self.note_gru_hidden))