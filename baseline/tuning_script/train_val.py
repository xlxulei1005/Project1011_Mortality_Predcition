import fasttext
import itertools
import os
from sklearn.metrics import roc_auc_score
#import matplotlib.pyplot as plt
#from tqdm import tqdm
import pandas as pd
#from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
import numpy as np
import pickle
import argparse
import logging
#helpper function
def readin_split(path):
    '''
    path: get the val/test set
    return: list of texts, corresponding labels
    '''
    text_val = []
    with open(path,'r') as f:
        for line in f:
            text_val.append(line)

    labels_val = []
    for i in range(len(text_val)):
        temp = text_val[i].strip('\n').split('__')
        text_val[i] = temp[0]
        labels_val.append(temp[-1])
    return text_val, labels_val


def tunning_function(time,epoch_li, learning_rate_li, window_size_li, dim_li):
	'''
	This is the function for tuning the model
	time: time period
	*_list: the parameter to be tuned
	return: nothing
	'''
	logger.info('This is for time period {}'.format(time))
	
	#configure parameters
	x = []
	x.append(epoch_li)
	x.append(learning_rate_li)
	x.append(window_size_li)
	x.append(dim_li)
	para_pair = list(itertools.product(*x))
	score_val_list = []
	score_test_list = []

	#configuration data path
	training_path = path + '_' + time + '_train.txt'
	val_path = path + '_' + time + '_val.txt'
	test_path = path + '_' + time + '_test.txt'
	
	#configure validation and test data
	text_val, label_val = readin_split(val_path)
	text_test, label_test = readin_split(test_path)
	binary_label_val = [0 if i == 'LIVE' else 1 for i in label_val]
	binary_label_test = [0 if i == 'LIVE' else 1 for i in label_test]
	
	#model
	for epoch_ in epoch_li:
		for lr_ in learning_rate_li:
			for wd_ in window_size_li:
				for dim_ in dim_li:
					logger.info('The epoch is '+ str(epoch_))
					logger.info('The learning rate is ' + str(lr_))
					logger.info('The window size is ' + str(wd_))
					logger.info('The word dimension is ' + str(dim_))	
					classifier = fasttext.supervised(training_path, 'model_{}_{}_{}_{}_{}'.format(time, epoch_, lr_, wd_, dim_), epoch = epoch_, lr = lr_, ws = wd_, dim = dim_,label_prefix='__label__')
					predict_list_val = classifier.predict(text_val)
					predict_list_val = list(itertools.chain.from_iterable(predict_list_val))
					predict_list_test = classifier.predict(text_test)
					predict_list_test = list(itertools.chain.from_iterable(predict_list_test))
					binary_predict_li_val = [0  if i == 'LIVE' else 1 for i in predict_list_val]
					binary_predict_li_test = [0  if i == 'LIVE' else 1 for i in predict_list_test]
					score_val = roc_auc_score(binary_label_val, binary_predict_li_val)
					score_test = roc_auc_score(binary_label_test, binary_predict_li_test)
					logger.info('validation score is' + str(score_val))
					logger.info('test score is' + str(score_test))
					score_val_list.append(score_val)
					score_test_list.append(score_test)
					print('finish one')
	
	#save the data
	print('finish tuning')
	para_pair_df = pd.DataFrame(para_pair)
	para_pair_df.columns = ['epoch_para', 'lr_para', 'ws_para','dim_para']
	para_pair_df['score_val'] = score_val
	para_pair_df['score_test'] = score_test
	pickle.dump(para_pair_df, open( "../result/save_{}.p".format(time), "wb" ) )


if __name__ == "__main__":
	
    #arguments
    parser = argparse.ArgumentParser(description = 'tune baseline for 1011')
    parser.add_argument('--epoch_num', default = '30 50', type = str, help = 'epoch number')
    parser.add_argument('-learning_rate', default = '0.001 0.005', type = str, help = 'learning_rate (default: 1)')
    #parser.add_argument('--batch_sz',default = 50, type = str, help = 'size of batch')
    parser.add_argument('--window_size', default = '3 5', type = str, help = 'the window_size of context window')
    parser.add_argument('--time_period', default = '15m 6h 12h 24h', type = str, help = 'time')
    parser.add_argument('--dim_list', default = '200 300',type = str, help = 'embedding dimension')
    args = parser.parse_args()

    #logger
    logger_name = "Parameter Tuning FastText"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler('../result/log_fasttext_result')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    #configure
    path = '../data/voc_100_downsample'
    time_list = args.time_period.split(' ')
    learning_rate_list = [float(i) for i in args.learning_rate.split(' ')]
    window_size_list = [float(i) for i in args.window_size.split(' ')]
    epoch_list = [int(i) for i in args.epoch_num.split(' ')]
    dim_list = [int(i) for i in args.dim_list.split(' ')]	

    #tuning
    for time in time_list:
	tunning_function(time,epoch_list, learning_rate_list, window_size_list, dim_list)
	
			
