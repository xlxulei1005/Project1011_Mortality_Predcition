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
import pickle

from model import AttentionRNN, Hierachical_BiGRU_max
from train import data_iter, eval_iter, training_loop, test_iter
from data import data_formatting
import logging
import os
import numpy as np
import torch.nn as nn
import torch
import config_Hierachical_BiGRU_max 
import config_AttentionRNN 

the_model = Hierachical_BiGRU_max(config)
the_model.load_state_dict(torch.load(PATH))

config_dict = {'cnn_rnn': config_AttentionRNN.config_loading, 'bigru_max': config_Hierachical_BiGRU_max.config_loading}
    
config = config_dict[args.model]()
config['model'] = args.model

DATAPATH = config['DATAPATH']
vocabulary = np.load(os.path.join(DATAPATH, 'voc_100.npy'))
index = range(len(vocabulary))
voca_dict = dict(zip(vocabulary, index))
config['vocab_size'] = len(index)

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

train_data, val_data, test_data, max_length = data_formatting(config = config, path = DATAPATH, time_name = config['time_name'], concat = config['concat'])

logger.info('loading data...')

logger.info('train size # sent ' + str(len(train_data)))
logger.info('dev size # sent ' + str(len(val_data)))
logger.info('test size # sent ' + str(len(test_data)))

logger.info(str(config))

if config['model'] == 'cnn_rnn':
    model = AttentionRNN(config)
elif config['model'] == 'bigru_max':
    model = Hierachical_BiGRU_max(config)


        