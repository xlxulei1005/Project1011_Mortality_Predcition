# config of AttentionRNN
def config_loading():
	word_padded_length_in_notes_dict = {'12h': 0, }

	config_cnn_rnn = {
	          'DATAPATH' : './data_unconcated',
	          'concat': True, # if the loaded data is splitted by sentences, concat = True, will concat all sentences into a note.
	          'to_sentences':True, # if you would like to keep only the sentence and patient level 
	          'words_dim': 300, 
	          'embed_mode': 'random', 
	          'output_channel': 100, # number of feature maps for CNN 
	          'dropout':0,
	          'target_class':2,
	          'note_gru_hidden': 200,
	          'bidirection_gru': True,
	          'batch_size': 16,
	          'learning_rate': 0.01,
	          'num_epochs':150,
	          'filter_width':8,
	          'cuda': True,
	          'attention': True,
	          'early_stop': 3,
	          'val_per_epoch': 5,
	          'data_portion': 1,
	          'optimizer': 'SGD', # Adam or SGD, the learning rate for Adam is default by 0.001
	          'padding_before_batch': True, 
	          'padding_max': True, 
	          'word_padded_length_in_notes': None, #None for default 
	          'savepath': './model/24h_words_dim_300_output_cha_100_hidden_200_filter_width_8_batch_16_SGD_lr0.01_drop_0_attention/',
	          'time_name': '24h',
	          'split_points':   [12*60, 24*60 ,48*60, 72*60, 96*60, 120*60, 240*60 ]
	}

	if config_cnn_rnn['word_padded_length_in_notes'] == None:
	    config_cnn_rnn['word_padded_length_in_notes'] = word_padded_length_in_notes_dict[config['time_name']]

	return config_cnn_rnn



