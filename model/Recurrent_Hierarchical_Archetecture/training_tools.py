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


def train_data(mini_batch, targets, model, optimizer, criterion):
	state_word,state_sent,state_note = model.init_hidden()
	state_word = state_word.cuda()
	state_sent = state_sent.cuda()
	state_note = state_note.cuda()
	_, _, _, y_pred = model(mini_batch,state_word,state_sent,state_note)
	loss = criterion(y_pred.cuda(), targets.long()) 
	loss.backward()    
	optimizer.step()
	return loss.data[0]

def test_data(mini_batch, targets, model, criterion):  
	state_word,state_sent,state_note = model.init_hidden()
	state_word = state_word.cuda()
	state_sent = state_sent.cuda()
	state_note = state_note.cuda()
	_, _, _, y_pred = model(mini_batch,state_word,state_sent,state_note)
	y_pred_death = F.softmax(y_pred).data.cpu().numpy()[:,1]       
	loss = criterion(y_pred.cuda(), targets.long())
	#loss = criterion(y_pred, targets.long()) 
	return loss.data[0], y_pred_death

def get_predictions(val_tokens, model):
	state_word,state_sent,state_note = model.init_hidden()
	state_word = state_word.cuda()
	state_sent = state_sent.cuda()
	state_note = state_note.cuda()
	_, _, _, y_pred = model(val_tokens, state_word, state_sent,state_note)    
	return y_pred

def pad_batch(mini_batch):
	mini_batch_size = len(mini_batch)
	max_note_len = int(np.mean([len(x) for x in mini_batch]))
	max_sent_len = int(np.mean([len(val) for sublist in mini_batch for val in sublist]))
	max_token_len = int(np.mean([len(tmp) for sublist in mini_batch for val in sublist for tmp in val]))
	main_matrix = np.zeros((mini_batch_size, max_note_len, max_sent_len, max_token_len), dtype= np.int)
	for i in range(main_matrix.shape[0]):
		for j in range(main_matrix.shape[1]):
			for k in range(main_matrix.shape[2]):
				for l in range(main_matrix.shape[3]):
					try:
						main_matrix[i,j,k,l] = mini_batch[i][j][k][l]
					except IndexError:
						pass
	return Variable(torch.from_numpy(main_matrix).permute(1,2,0,3))

def test_accuracy_mini_batch(tokens, labels, model):
	y_pred = get_predictions(tokens.cuda(), model)
	_, y_pred = torch.max(y_pred, 1)
	correct = np.ndarray.flatten(y_pred.data.cpu().numpy())
	labels = np.ndarray.flatten(labels.data.cpu().numpy())
	num_correct = sum(correct == labels)
	return float(num_correct) / len(correct)


def test_accuracy_full_batch(tokens, labels, mini_batch_size, model):
	p = []
	l = []
	a = []
	g = gen_minibatch(tokens, labels, mini_batch_size)
	for token, label in g:
		y_pred = get_predictions(token, model)
		y_pred_death = F.softmax(y_pred).data.cpu().numpy()[:,1]
		a.extend(list(y_pred_death))   
		_, y_pred = torch.max(y_pred, 1)
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

def check_val_loss(val_tokens, val_labels, mini_batch_size, model, criterion):
	val_loss = []
	y_pred_death_list = []
	target_list =[]
	for token, label in iterate_minibatches(val_tokens, val_labels, mini_batch_size, shuffle= True):
		loss, y_pred_death = test_data(pad_batch(token).cuda(), Variable(torch.from_numpy(label), requires_grad= False).cuda(), 
								  model, criterion)
		val_loss.append(loss)
		y_pred_death_list.extend(list(y_pred_death))
		target_list.extend(list(label))
		#val_loss.append(test_data(pad_batch(token), Variable(torch.from_numpy(label), requires_grad= False),model)) 
	#print(val_labels.data)
	#print(y_pred_death_list)
	AUC_score = metrics.roc_auc_score(target_list, y_pred_death_list)                       
	return np.mean(val_loss),AUC_score


def timeSince(since):
	now = time.time()
	s = now - since
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)

def train_early_stopping(mini_batch_size, X_train, y_train, X_val, y_val, model, 
						 optimizer, loss_criterion, num_epoch, logger, 
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
			loss = train_data(tokens, labels, model, optimizer, loss_criterion)
			acc = test_accuracy_mini_batch(tokens, labels, model)
			accuracy_full.append(acc)
			accuracy_epoch.append(acc)
			loss_full.append(loss)
			loss_epoch.append(loss)
			# print loss every n passes
			if i % print_loss_every == 0:
				logger.info('Loss at %d minibatches, %d epoch,(%s) is %f' %(i, epoch_counter, timeSince(start), np.mean(loss_epoch)))
				logger.info('Accuracy at %d minibatches is %f' % (i, np.mean(accuracy_epoch)))
			# check validation loss every n passes
			if i % print_val_loss_every == 0:

				val_loss, auc = check_val_loss(X_val, y_val, mini_batch_size, model,loss_criterion)
				torch.save(model.state_dict(), str(i) + 'model.pt')
				logger.info('save model!') 
				logger.info('Average training loss at this epoch..minibatch..%d..is %f' % (i, np.mean(loss_epoch)))
				logger.info('Validation loss after %d passes is %f' %(i, val_loss))
				logger.info('Validation auc after %d passes is %f' %(i, auc))
				#if val_loss > np.mean(loss_full):
				#    logger.info('Validation loss is higher than training loss at %d is %f , stopping training!' % (i, val_loss))
				#    logger.info('Average training loss at %d is %f' % (i, np.mean(loss_full)))
		except StopIteration:
			epoch_counter += 1
			logger.info('Reached %d epocs' % epoch_counter)
			logger.info('i %d' % i)
			g = gen_minibatch(X_train, y_train, mini_batch_size)
			loss_epoch = []
			accuracy_epoch = []
	return loss_full
