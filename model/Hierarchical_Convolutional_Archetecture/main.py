"""
Nan Wu
All rights reserved
Report bugs to Nan Wu nw1045@nyu.edu
"""
from model import AttentionRNN
from train import data_iter, eval_iter, training_loop
from data import data_formatting
import logging
import os 
import numpy as np
import torch.nn as nn
import torch

DATAPATH = './data_cnn'
vocabulary = np.load(os.path.join(DATAPATH, 'voc_100.npy'))
config = {'vocab_size': len(vocabulary),
          'words_dim': 100,
          'embed_mode': 'random',
          'output_channel': 50,
          'dropout':0,
          'target_class':2,
          'note_gru_hidden': 100,
          'bidirection_gru': False,
          'batch_size': 8,
          'learning_rate': 0.001,
          'num_epochs':100,
          'filter_width':5,
          'cuda': True,
          'data_portion': 1,
          'savepath': './model/15m_words_dim_100_output_cha_50_hidden_100_filter_width_5_batch_8_Adam_drop_0/',
          'time_name': '15m'
}


logger_name = "mortality_prediction"
logger = logging.getLogger(logger_name)
logger.setLevel(logging.INFO)
# file handler
if os.path.exists(config['savepath']):
   pass
else:    
   os.mkdir(config['savepath'])
fh = logging.FileHandler(config['savepath'] + 'output.log')
fh.setLevel(logging.INFO)
logger.addHandler(fh)

# stream handler
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)

logger.info('loading data...')

#DATAPATH = './data_cnn'
#vocabulary = np.load(os.path.join(DATAPATH, 'voc_100.npy'))
index = range(len(vocabulary))
voca_dict = dict(zip(vocabulary, index))
train_data, val_data, test_data = data_formatting(path = DATAPATH, time_name = config['time_name'])

logger.info('train size # sent ' + str(len(train_data)))
logger.info('dev size # sent ' + str(len(val_data)))
logger.info('test size # sent ' + str(len(test_data)))


logger.info(str(config))

model = AttentionRNN(config)
if config['cuda']:
     model.cuda()

# Loss and Optimizer
loss = nn.CrossEntropyLoss()  
#optimizer = torch.optim.Adadelta(model.parameters(), lr=config['learning_rate'])
optimizer = torch.optim.Adam(model.parameters())
print(model.parameters())
# Train the model
training_iter = data_iter(train_data[:int(config['data_portion']*len(train_data))], config['batch_size'])

dev_iter = eval_iter(val_data[:int(config['data_portion']*len(val_data))], config['batch_size'])
logger.info('Start to train...')
#os.mkdir(config['savepath'])
training_loop(config, model, loss, optimizer, train_data[:int(config['data_portion']*len(train_data))], training_iter, dev_iter, logger, config['savepath'])

