"""
training Recurrent Hierarchical model.
Sheng Liu
All rights reserved
Report bugs to shengliu@nyu.edu
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
from training_tools import *
from model import AttentionNoteRNN
import logging
#%matplotlib inline

def data_split(patient_list, downsampling_rate = 0.3):
    # Argments:
    #       patient_list : in format [[patient_id, label]] where death label = 1 otherwise label = 0
    #       downsampling_rate : the percentage of negative sample in the final training data
    # Output:
    #        train_data, validation data and test_data
    
    while True:
        
        np.random.rand(patient_list.shape[0]).argsort()
        np.take(patient_list,np.random.rand(patient_list.shape[0]).argsort(),axis=0,out=patient_list)
        patient_list = np.array(patient_list)
        num_patients = len(patient_list[:,0])
        train_data = patient_list[:int(0.7*num_patients)]
        val_data   = patient_list[int(0.7*num_patients):int(0.8*num_patients)]
        test_data  = patient_list[int(0.8*num_patients):num_patients]
        
        #downsampling
        if sum(train_data[:,1]) > 10:
            break
        else:
            np.random.rand(patient_list.shape[0]).argsort()
            np.take(patient_list,np.random.rand(patient_list.shape[0]).argsort(),axis=0,out=patient_list)
    
    if sum(train_data[:,1])/len(train_data[:,1]) <= 0.3:

        downsampling_size = int(sum(train_data[:,1])*(1 - downsampling_rate)/downsampling_rate)

        train_data_survive = train_data[train_data[:,1] != 1][:downsampling_size]
        train_data_dead = train_data[train_data[:,1] == 1]
        train_data = np.vstack((train_data_survive,train_data_dead))
        #random.shuffle(train_data)
        print('The percentage of negative sample after downsampling is {:.1%}'.format(sum(train_data[:,1])/len(train_data[:,1])))
        return train_data[:,0], val_data[:,0], test_data[:,0]

    else:
        print('The percentage of negative sample is {:.1%}'.format(sum(train_data[:,1])/len(train_data[:,1])))
        return train_data[:,0], val_data[:,0], test_data[:,0]

batch_size = 32
num_tokens = 181444
embed_size = 200
word_gru_hidden = 200
sent_gru_hidden = 200
note_gru_hidden = 200
number_of_classes = 2
num_epoch = 4000
print_val_loss_every = 100
print_loss_every = 50

model = AttentionNoteRNN(batch_size, num_tokens, embed_size, word_gru_hidden, \
                         sent_gru_hidden, note_gru_hidden, n_classes= number_of_classes, note_attention = True)

model.cuda()

logger_name = "mortality_prediction"
logger = logging.getLogger(logger_name)
logger.setLevel(logging.INFO)

# file handler
fh = logging.FileHandler('./' + 'Mortality_prediction')
fh.setLevel(logging.INFO)
logger.addHandler(fh)

# stream handler
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)

#    torch.cuda.set_device(args.gpu_id)

#for arg in vars(args):
#    logger.info(str(arg) + ' ' + str(getattr(args, arg)))

# load train/dev/test data
# train data
logger.info('loading data...')
train = np.load('./train_15m.npy')
val = np.load('./val_15m.npy')
test = np.load('./test_15m.npy')
X_train = train.item()['DATA']
y_train = train.item()['MORTALITY_LABEL']
X_val = val.item()['DATA']
y_val = val.item()['MORTALITY_LABEL']
X_test = test.item()['DATA']
y_test = test.item()['MORTALITY_LABEL']
#y = np.load('../mordality_pred/label.npy')
#y = y.item()['MORTALITY_LABEL']
#patient_index = np.vstack((range(len(y)), y)).T.astype(int)
#train, val, test = data_split(patient_index, downsampling_rate = 0.3)
#X_train = X[train]
#y_train = y[train]
#X_val   = X[val]
#y_val   = y[val]
#X_test  = X[test]
#y_test  = y[test]
logger.info('train size # sent ' + str(len(X_train)))
logger.info('dev size # sent ' + str(len(X_val)))
logger.info('test size # sent ' + str(len(X_test)))

learning_rate = 1e-2
optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

logger.info('Start to train...')

loss_full = train_early_stopping(batch_size, X_train, y_train, X_val, y_val, model, optimizer, 
                            criterion, num_epoch , logger, print_val_loss_every, print_loss_every)

logger.info('Training end!')
logger.info('Test!')
test_acc,test_auc = test_accuracy_full_batch(X_test, y_test, 64, model)
logger.info('Test accuracy is %.2f' % test_acc)
logger.info('Test auc is %.2f' % test_auc)
np.save('loss_full','loss_full')
