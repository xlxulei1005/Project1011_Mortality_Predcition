# config of Hierachical_BiGRU_max

def config_loading():
	config_bigru_max = {
	          'DATAPATH' : '../../debugdata/voc_100_downsample_concat_sent',
              'concat': False, # if the input data only have note and patient level, concat = False
	          'words_dim': 300, # word vector length
	          'embed_mode': 'not random',
	          'bigru_max_sub_hidden': 100, # Number of hidden neurons for the second gru 
	          'target_class':2, 
	          'bigru_max_note_hidden': 80, # Number of hidden neurons for the first gru
	          'batch_size': 16, 
	          'learning_rate': 0.0001,
	          'num_epochs':150,
	          'cuda': True,
	          'early_stop': 1,
	          'val_per_epoch': 1,
	          'data_portion': 1, # portion of training data to use
	          'optimizer': 'Adam',
	          'padding_before_batch': False, # Only for CNN
	          'to_sentences': False,  # for sentence and patient level models 
              'word_padded_length_in_notes': None, # only for CNN
              'attention': True,
              'max_note_length' : 300, # maximun note length (if the maximum length in a minbatch is smaller than this number, no notes will be cropped.)
              'random_cutting': True, # random cropping
	          'savepath': './model/',
	          #'savepath': './model/Test_15m_bigru_max_regulizer_maxnotelen_300_wordsdim_300_subhidden_200_notehidden_80_batch_16_Adam_0.0001_correct/',
	          'time_name': '15m',
                  'save_test_result':True,
                'regulization_by_note': True,
                'split_points':   [12*60, 24*60 ,48*60, 72*60, 96*60, 120*60, 240*60 ],
                'regulization_by_time': True
                'embed_tuned':True
	}
	return config_bigru_max
