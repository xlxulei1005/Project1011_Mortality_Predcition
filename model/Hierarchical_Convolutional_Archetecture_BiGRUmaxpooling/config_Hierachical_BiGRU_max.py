# config of Hierachical_BiGRU_max

def config_loading():
	config_bigru_max = {
	          'DATAPATH' : './voc_100_downsample_concat_sent',
              'concat': False, # if the input data only have note and patient level, concat = False
	          'words_dim': 300, # word vector length
	          'embed_mode': 'random', 
	          'bigru_max_sub_hidden': 200, # Number of hidden neurons for the second gru 
	          'target_class':2, 
	          'bigru_max_note_hidden': 80, # Number of hidden neurons for the first gru
	          'batch_size': 16, 
	          'learning_rate': 0.01,
	          'num_epochs':150,
	          'cuda': True,
	          'early_stop': 10,
	          'val_per_epoch': 1,
	          'data_portion': 1, # portion of training data to use
	          'optimizer': 'SGD',
	          'padding_before_batch': False, # Only for CNN
	          'to_sentences': False,  # for sentence and patient level models 
              'word_padded_length_in_notes': None, # only for CNN
              'max_note_length' : 300, # maximun note length (if the maximum length in a minbatch is smaller than this number, no notes will be cropped.)
              'random_cutting': True, # random cropping
	          'savepath': './model/15m_bigru_max_regulizer_maxnotelen_300_wordsdim_300_subhidden_200_notehidden_80_batch_16_SGD_0.01/',
	          'time_name': '15m',
                  'save_test_result':False,
                'regulization_by_note': True
	}
	return config_bigru_max
