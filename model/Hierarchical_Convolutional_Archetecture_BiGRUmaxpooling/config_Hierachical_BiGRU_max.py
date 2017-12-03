# config of Hierachical_BiGRU_max

def config_loading():
	config_bigru_max = {
	          'DATAPATH' : './data_notdown',
              'concat': False, # if the input data only have note and patient level, concat = False
	          'words_dim': 300, # word vector length
	          'embed_mode': 'random', 
	          'bigru_max_sub_hidden': 200, # Number of hidden neurons for the second gru 
	          'target_class':2, 
	          'bigru_max_note_hidden': 80, # Number of hidden neurons for the first gru
	          'batch_size': 16, 
	          'learning_rate': 0.001,
	          'num_epochs':150,
	          'cuda': True,
	          'early_stop': 3,
	          'val_per_epoch': 1,
	          'data_portion': 1, # portion of training data to use
	          'optimizer': 'Adam',
	          'padding_before_batch': False, # Only for CNN
	          'to_sentences': False,  # for sentence and patient level models 
              'word_padded_length_in_notes': None, # only for CNN
              'max_note_length' : 200, # maximun note length (if the maximum length in a minbatch is smaller than this number, no notes will be cropped.)
              'random_cutting': True, # random cropping
	          'savepath': './model/24h_bigru_max_maxnotelen_200_wordsdim_300_subhidden_200_notehidden_80_batch_16_Adam/',
	          'time_name': '24h',
                  'save_test_result':False
	}
	return config_bigru_max
