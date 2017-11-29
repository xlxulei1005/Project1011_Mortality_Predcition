"""
training tools for Recurrent Hierarchical model.
Sheng Liu
All rights reserved
Report bugs to shengliu@nyu.edu
"""
import time
import math
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
import sklearn.metrics as metrics
#%matplotlib inline

def train_data(mini_batch, targets, word_attn_model, sent_attn_model, word_optimizer, sent_optimizer, criterion):
	state_word = word_attn_model.init_hidden()
	state_sent = sent_attn_model.init_hidden()
	state_word = state_word.cuda()
	state_sent = state_sent.cuda()
	max_sents, batch_size, max_tokens = mini_batch.size()
	word_optimizer.zero_grad()
	sent_optimizer.zero_grad()
	#print("state_word_size: ", state_word.size())
	#print("state_sent_size: ", state_sent.size())
	#print("word_attn_input_size: ", mini_batch[0,:,:].transpose(0,1).size())
	s = None
	for i in range(max_sents):
		_s, state_word, _ = word_attn_model(mini_batch[i,:,:].transpose(0,1), state_word)
		if(s is None):
			#s = _s
			s = _s.unsqueeze(0)
		else:
			#s = torch.cat((s,_s),0)  
			s = torch.cat((s,_s.unsqueeze(0)),0)
   # print("sent_attn_input_size: ", s.size())
	y_pred, state_sent, _ = sent_attn_model(s, state_sent)
	loss = criterion(y_pred, targets.long()) 
	loss.backward()
	
	word_optimizer.step()
	sent_optimizer.step()
	
	return loss.data[0]

def test_data(mini_batch, targets, word_attn_model, sent_attn_model, criterion):    
	state_word = word_attn_model.init_hidden()
	state_sent = sent_attn_model.init_hidden()
	state_word = state_word.cuda()
	state_sent = state_sent.cuda()
	max_sents, batch_size, max_tokens = mini_batch.size()
	s = None
	for i in range(max_sents):
		_s, state_word, _ = word_attn_model(mini_batch[i,:,:].transpose(0,1), state_word)
		if(s is None):
			s = _s.unsqueeze(0)
		else:
			s = torch.cat((s,_s.unsqueeze(0)),0)            
	y_pred, state_sent,_ = sent_attn_model(s, state_sent)
	y_pred_death = F.softmax(y_pred).data.cpu().numpy()[:,1]
	loss = criterion(y_pred.cuda(), targets.long())     
	return loss.data[0], y_pred_death


def get_predictions(val_tokens, word_attn_model, sent_attn_model):
	max_sents, batch_size, max_tokens = val_tokens.size()
	state_word = word_attn_model.init_hidden()
	state_sent = sent_attn_model.init_hidden()
	state_word = state_word.cuda()
	state_sent = state_sent.cuda()
	s = None
	for i in range(max_sents):
		_s, state_word, _ = word_attn_model(val_tokens[i,:,:].transpose(0,1), state_word)
		if(s is None):
			s = _s.unsqueeze(0)
		else:
			s = torch.cat((s,_s.unsqueeze(0)),0)            
	y_pred, state_sent, _ = sent_attn_model(s, state_sent)    
	return y_pred


def pad_batch(mini_batch):
	mini_batch_size = len(mini_batch)
	max_sent_len = int(np.max([len(x) for x in mini_batch]))
	max_token_len = int(np.mean([len(val) for sublist in mini_batch for val in sublist]))
	main_matrix = np.zeros((mini_batch_size, max_sent_len, max_token_len), dtype= np.int)
	for i in range(main_matrix.shape[0]):
		for j in range(main_matrix.shape[1]):
			for k in range(main_matrix.shape[2]):
				try:
					main_matrix[i,j,k] = mini_batch[i][j][k]
				except IndexError:
					pass
	return Variable(torch.from_numpy(main_matrix).transpose(0,1))


def test_accuracy_mini_batch(tokens, labels, word_attn, sent_attn):
	y_pred = get_predictions(tokens, word_attn, sent_attn)
	_, y_pred = torch.max(y_pred, 1)
	correct = np.ndarray.flatten(y_pred.data.cpu().numpy())
	labels = np.ndarray.flatten(labels.data.cpu().numpy())
	num_correct = sum(correct == labels)
	return float(num_correct) / len(correct)

def test_accuracy_full_batch(tokens, labels, mini_batch_size, word_attn, sent_attn):
	p = []
	l = []
	a = []
	g = gen_minibatch(tokens, labels, mini_batch_size)
	for token, label in g:
		y_pred = get_predictions(token, word_attn, sent_attn)
		y_pred_death = F.softmax(y_pred).data.cpu().numpy()[:,1]
		_, y_pred = torch.max(y_pred, 1)
		a.extend(list(y_pred_death))   
		p.append(np.ndarray.flatten(y_pred.data.cpu().numpy()))
		l.append(np.ndarray.flatten(label.data.cpu().numpy()))
	AUC_score = metrics.roc_auc_score(labels, a)
	p = [item for sublist in p for item in sublist]
	l = [item for sublist in l for item in sublist]
	p = np.array(p)
	l = np.array(l)
	num_correct = sum(p == l)
	return float(num_correct)/ len(p), AUC_score 

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
	assert inputs.shape[0] == targets.shape[0]
	if shuffle:
		indices = np.arange(inputs.shape[0])
		np.random.shuffle(indices)
	for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batchsize]
		else:
			excerpt = slice(start_idx, start_idx + batchsize)
		yield inputs[excerpt], targets[excerpt]

def gen_minibatch(tokens, labels, mini_batch_size, shuffle= True):
	for token, label in iterate_minibatches(tokens, labels, mini_batch_size, shuffle= shuffle):
		token = pad_batch(token)
		yield token.cuda(), Variable(torch.from_numpy(label), requires_grad= False).cuda()
		#yield token, Variable(torch.from_numpy(label), requires_grad= False)


def check_val_loss(val_tokens, val_labels, mini_batch_size, word_attn_model, sent_attn_model, criterion):
	val_loss = []
	y_pred_death_list = []
	target_list = []
	for token, label in iterate_minibatches(val_tokens, val_labels, mini_batch_size, shuffle= True):
		loss, y_pred_death = test_data(pad_batch(token).cuda(), Variable(torch.from_numpy(label), requires_grad= False).cuda(), 
								  word_attn_model, sent_attn_model, criterion)
		val_loss.append(loss)
		y_pred_death_list.extend(list(y_pred_death))
		target_list.extend(list(label))
	AUC_score = metrics.roc_auc_score(target_list, y_pred_death_list)
	return np.mean(val_loss), AUC_score

def timeSince(since):
	now = time.time()
	s = now - since
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)


def train_early_stopping(mini_batch_size, X_train, y_train, X_test, y_test, word_attn_model, sent_attn_model, 
						 word_attn_optimiser, sent_attn_optimiser, loss_criterion, num_epoch, 
						 print_val_loss_every = 50, print_loss_every = 20):
	start = time.time()
	loss_full = []
	loss_epoch = []
	accuracy_epoch = []
	loss_smooth = []
	accuracy_full = []
	epoch_counter = 0
	g = gen_minibatch(X_train, y_train, mini_batch_size)
	for i in range(1, num_epoch + 1):
		try:
			tokens, labels = next(g)
			loss = train_data(tokens, labels, word_attn_model, sent_attn_model, word_attn_optimiser, sent_attn_optimiser, loss_criterion)
			acc = test_accuracy_mini_batch(tokens, labels, word_attn_model, sent_attn_model)
			accuracy_full.append(acc)
			accuracy_epoch.append(acc)
			loss_full.append(loss)
			loss_epoch.append(loss)
			# print loss every n passes
			if i % print_loss_every == 0:
				print ('Loss at %d minibatches, %d epoch,(%s) is %f' %(i, epoch_counter, timeSince(start), np.mean(loss_epoch)))
				print ('Accuracy at %d minibatches is %f' % (i, np.mean(accuracy_epoch)))
			# check validation loss every n passes
			if i % print_val_loss_every == 0:
				val_loss, auc = check_val_loss(X_test, y_test, mini_batch_size, word_attn_model, sent_attn_model, loss_criterion)
				print('Average training loss at this epoch..minibatch..%d..is %f' % (i, np.mean(loss_epoch)))
				print('Validation loss after %d passes is %f' %(i, val_loss))
				print('Validation auc after %d passes is %f' %(i, auc))
				#if val_loss > np.mean(loss_full):
				#    print('Validation loss is higher than training loss at %d is %f , stopping training!' % (i, val_loss))
				#    print('Average training loss at %d is %f' % (i, np.mean(loss_full)))
		except StopIteration:
			epoch_counter += 1
			print('Reached %d epocs' % epoch_counter)
			print('i %d' % i)
			g = gen_minibatch(X_train, y_train, mini_batch_size)
			loss_epoch = []
			accuracy_epoch = []
	return loss_full
