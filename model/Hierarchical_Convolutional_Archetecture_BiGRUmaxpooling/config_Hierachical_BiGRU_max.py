# config of Hierachical_BiGRU_max
def config_loading():
	config_bigru_max = {
	          'DATAPATH' : './data_unconcated',
                  'concat': True,
	          'words_dim': 300,
	          'embed_mode': 'random',
	          'bigru_max_sub_hidden': 200,
	          'target_class':2,
	          'bigru_max_note_hidden': 50,
	          'batch_size': 8,
	          'learning_rate': 0.01,
	          'num_epochs':150,
	          'cuda': True,
	          'early_stop': 3,
	          'val_per_epoch': 1,
	          'data_portion': 1,
	          'optimizer': 'Adam',
	          'padding_before_batch': False,
	          'to_sentences':True, 
              'word_padded_length_in_notes': None,
	          'savepath': './model/6h_bigru_max_words_dim_300_subhidden_50_notehidden_200_batch_8_Adam/',
	          'time_name': '6h'
	}
	return config_bigru_max
