# config of AttentionRNN
def config_loading():
	word_padded_length_in_notes_dict = {'12h': 0, }

	config_cnn_rnn = {
	          'DATAPATH' : './data_unconcated',
	          'words_dim': 300,
	          'embed_mode': 'random',
	          'output_channel': 100,
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
	          'concat': True,
	          'optimizer': 'SGD',
	          'padding_before_batch': True,
	          'padding_max': True,
	          'to_sentences':True,
	          'word_padded_length_in_notes': None, #None for default 
	          'savepath': './model/24h_words_dim_300_output_cha_100_hidden_200_filter_width_8_batch_16_SGD_lr0.01_drop_0_attention/',
	          'time_name': '24h'
	}

	if config_cnn_rnn['word_padded_length_in_notes'] == None:
	    config_cnn_rnn['word_padded_length_in_notes'] = word_padded_length_in_notes_dict[config['time_name']]

	return config_cnn_rnn



