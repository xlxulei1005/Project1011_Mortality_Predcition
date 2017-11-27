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
import sklearn.metrics as metrics

def data_iter(source, batch_size):
    dataset_size = len(source)
    start = -1 * batch_size
    order = list(range(dataset_size))
    random.shuffle(order)

    while True:
        start += batch_size
        if start > dataset_size - batch_size:
            # Start another epoch.
            start = 0
            random.shuffle(order)   
        batch_indices = order[start:start + batch_size]
        batch = [source[index] for index in batch_indices]
        yield [source[index] for index in batch_indices]

# This is the iterator we use when we're evaluating our model. 
# It gives a list of batches that you can then iterate through.
def eval_iter(source, batch_size):
    batches = []
    dataset_size = len(source)
    start = -1 * batch_size
    order = list(range(dataset_size))
    random.shuffle(order)

    while start < dataset_size - batch_size:
        start += batch_size
        batch_indices = order[start:start + batch_size]
        batch = [source[index] for index in batch_indices]
        if len(batch) == batch_size:
            batches.append(batch)
        else:
            continue
        
    return batches

# The following function gives batches of vectors and labels, 
# these are the inputs to your model and loss function
def get_batch(batch):
    vectors = []
    labels = []
    for dict in batch:
        vectors.append(dict["text"])
        labels.append(dict["label"])
    return vectors, labels

def pad_minibatch(vectors):
    length = max([len(x) for x in vectors])
    padded_v = []
    note_len = len(vectors[0][0])
    for before_padded in vectors:
        #print(before_padded)
        v = before_padded[:]
        #print(type(v))
        for i in range(length - len(before_padded)):
            v.append([0]*note_len)
        padded_v.append(v)
    return torch.from_numpy(np.array(padded_v)).permute(1,2,0)

def evaluate(model, loss_, data_iter, config):
    model.eval()
    correct = 0
    total = 0
    labels_all = []
    output_all = []
    for i in range(len(data_iter)):
        vectors, labels = get_batch(data_iter[i])
        vectors = pad_minibatch(vectors)
        labels = torch.stack(labels).squeeze()
        if config['cuda']:
     		vectors = vectors.cuda()
     		labels = labels.cuda()
     	vectors = Variable(vectors)
        labels = Variable(labels)
        hidden = model.init_hidden()
        if config['cuda']:
        	hidden = hidden.cuda()
        if config['attention']:
            _, _, _, output = model(vectors, hidden)
        else:
            output = model(vectors, hidden)
        loss = loss_(output, labels)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        predicted = predicted.cpu()
        predicted = predicted.numpy()
        labels = labels.cpu()
        labels = labels.data.numpy()
        correct += (predicted == labels).sum()
        labels_all += list(labels)
        output = output.cpu()
        output_all += list(output.data.numpy())
    output_all = np.array(output_all)
    auc = metrics.roc_auc_score(labels_all, output_all[:,1])
    #loss_epoch = loss_(Variable(torch.from_numpy(output_all)), Variable(torch.from_numpy(np.array(labels))))
    loss_epoch = metrics.log_loss(labels_all, output_all[:,1])
    return loss_epoch, correct / float(total), auc

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def training_loop(config, model, loss_, optim, train_data, training_iter, dev_iter, logger, savepath):
    step = 0
    epoch = 0
    total_batches = int(len(train_data) / config['batch_size'])
    start = time.time()
    labels_all = []
    output_all = []
    auc_max = 0
    early_count = 0
    while epoch <= config['num_epochs']:
        model.train()
        vectors, labels = get_batch(next(training_iter)) 
        vectors = pad_minibatch(vectors)
        labels = torch.stack(labels).squeeze()
        print(labels)
        if config['cuda']:
     		vectors = vectors.cuda()
     		labels = labels.cuda()
     	vectors = Variable(vectors)
        labels = Variable(labels)
        optim.zero_grad()
        
        
        hidden = model.init_hidden()
        if config['cuda']:
            hidden = hidden.cuda()
        
        #model.zero_grad()
        if config['attention']:
            _, note_attn_norm, _, output = model(vectors, hidden)
        else:
            output = model(vectors, hidden)
        print('note_attn_norm:')
        print(note_attn_norm)
        lossy = loss_(output, labels)
        lossy.backward()
        #torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
        optim.step()
        
        labels = labels.cpu()
        labels = labels.data.numpy()
        labels_all += list(labels)
        output= output.cpu()
        output_all += list(output.data.numpy())
    
        if step % total_batches == 0:
            output_all = np.array(output_all)
            #print(output_all.shape, np.array(labels).shape)
            #loss_epoch = loss_(Variable(torch.from_numpy(output_all)), Variable(torch.from_numpy(np.array(labels))))
            loss_epoch = metrics.log_loss(labels_all, output_all[:,1])

            if epoch % config['val_per_epoch'] == 0 and epoch!=0:
            	#print(timeSince(start), end = '; ')
            	logger.info(timeSince(start))
                #print(" Epoch %i; Step %i; Train Loss %f; Val Loss: %f; Val acc %f; Val AUC %f" 
                #      %(epoch, step, lossy.data[0],\
                #        evaluate(model, loss_, dev_iter)))
                eval_loss, acc, auc = evaluate(model, loss_, dev_iter, config)
                logger.info(timeSince(start))
                logger.info("Epoch %i; Step %i / %i; Train Loss %f; Val Loss: %f; Val acc %f; Val AUC %f" 
                     %(epoch, step%total_batches, total_batches, loss_epoch, eval_loss, acc, auc))
                
                if auc < auc_max:
                    early_count += 1
                    if early_count > config['early_stop']:
                        logger.info('EARLY STOP:  Max auc %f at Epoch %i' % (auc_max,epoch))
                        torch.save(model.state_dict(), savepath + str(epoch) + 'model.pt')
                        logger.info('Model Saved')
                else:
                    auc_max = auc
                    early_count = 0


            
            #output_all = np.array(output_all)
            #print(output_all.shape, np.array(labels).shape)
            #loss_epoch = loss_(Variable(torch.from_numpy(output_all)), Variable(torch.from_numpy(np.array(labels))))
            #loss_epoch = metrics.log_loss(np.array(labels_all), output_all)
            logger.info("Epoch %i; Total Loss %f" %(epoch, loss_epoch))
            labels_all = []
            output_all = []
            epoch += 1
        else:
            logger.info(timeSince(start))
            logger.info("Epoch %i; Step %i / %i; Train Loss %f" %(epoch, step % total_batches, total_batches, lossy.data[0]))
            
        step += 1


