"""
Train CNN-RNN or BiGRU-maxpooling with MIMIC-III data
agrs: model (['cnn_rnn', 'bigru_max'])


Nan Wu
All rights reserved
Report bugs to Nan Wu nw1045@nyu.edu
"""

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
import argparse
    
def main(args):
    
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

    print(model.parameters())


    if config['cuda']:
         model.cuda()
    
    # Loss and Optimizer
    loss = nn.CrossEntropyLoss()
    if config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'])
    elif config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
        
    
    # Train the model
    training_iter = data_iter(train_data[:int(config['data_portion']*len(train_data))], config['batch_size'])

    dev_iter = eval_iter(val_data[:int(config['data_portion']*len(val_data))], config['batch_size'])

    testing_iter = test_iter(test_data[:int(config['data_portion']*len(test_data))], 1)


    logger.info('Start to train...')
    #os.mkdir(config['savepath'])
    training_loop(config, model, loss, optimizer, train_data[:int(config['data_portion']*len(train_data))], training_iter, dev_iter, testing_iter, logger, config['savepath'])



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Mortality Prediction')
    parser.add_argument("--model", type=str, default='cnn_rnn', choices=['cnn_rnn', 'bigru_max']) # 
    args = parser.parse_args()
    main(args)

