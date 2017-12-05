"""
Nan Wu
All rights reserved
Report bugs to Nan Wu nw1045@nyu.edu
"""
import torch
import numpy as np
import os

def bin_time_label_generation(train_data, split_points = [5, 10 ,15]):
    '''
    
    '''
    def label_generate(time):
        label_index = 0
        for point in split_points:
            if time<= point:
                return label_index
            label_index +=1
        return label_index+1
    label_all = []           
   
    for time_list in train_data['TIME_TODEATH']:
        label = []
        for t in time_list:
            label.append(label_generate[t])
        label_all.append(label)
    return label_all
    
def continues_time_label_generation():
    pass

def notes_combine(infor_all, concat = True):
    data = infor_all['DATA'] 
    notes_per_sub = []
    max_len = []
    for sub in data:
        notes = []
        for note in sub:
            if concat:
            	d = [item for sublist in note for item in sublist]
            else:
            	d = note
            notes.append(d )
            max_len.append(len(d))
        notes_per_sub.append(notes)
    #del notes
    #del d
    return notes_per_sub, max_len

def notes_padding(notes_all, note_len):
    notes_per_sub = []
    for sub in notes_all:
        notes = []
        for note in sub:
            num_padding = note_len - len(note)
            new_sentence = note + [0] * num_padding
            notes.append(new_sentence )
        notes_per_sub.append(notes)
    del notes
    return notes_per_sub

def data_dict_generate(label, data):
    result = []
    for (d, l) in zip(data, label):
        #print(l)
        result.append({'text':d, 'label':torch.LongTensor([int(l)])})
    return result

def data_loading(path, time_name, test_file = None, train_file =None,val_file= None ):
    if test_file == None:
        test_file = 'test_'+ time_name +'.npy'
    if train_file == None:
        train_file = 'train_'+ time_name +'.npy' 
    if val_file == None:
        val_file = 'val_'+ time_name +'.npy'

    test_time = np.load(os.path.join(path, test_file)).item()
    train_time = np.load(os.path.join(path, train_file)).item()
    val_time = np.load(os.path.join(path, val_file)).item()
    return train_time, test_time, val_time

def notes_to_sentences(infor_all):
    result = []
    for sub in infor_all['DATA'] :
        sub_sentences = []
        for note in sub:
            sub_sentences = sub_sentences + note
        result.append(sub_sentences)
    infor_all['DATA'] = result
    return infor_all


def data_formatting(config, path, time_name,  concat = True, sort_by_number_of_notes = True, test_file = None, train_file =None,val_file= None):
    
    train_time, test_time, val_time = data_loading(path, time_name, test_file, train_file, val_file)
    
    if config['to_sentences']:
        train_time, test_time, val_time = notes_to_sentences(train_time), notes_to_sentences(test_time), notes_to_sentences(val_time)
        test_notes, len_test = notes_combine(test_time, False )
        train_notes , len_train= notes_combine(train_time, False)
        val_notes, len_val = notes_combine(val_time, False)
        
    else:
        test_notes, len_test = notes_combine(test_time, concat )
        train_notes , len_train= notes_combine(train_time, concat)
        val_notes, len_val = notes_combine(val_time, concat)

    len_all = np.array(len_test + len_train + len_val)
    
    if sort_by_number_of_notes:
        number_of_notes_train = [len(x) for x in train_notes]
        index = np.argsort(np.array(number_of_notes_train))
        train_notes= np.array(train_notes)[list(index)]

    if config['padding_before_batch']:
        if config['padding_max']:
            padding_length = len_all.max()
        else:
            padding_length = config['word_padded_length_in_notes']

        test_notes_padded = notes_padding(test_notes, padding_length)
        train_notes_padded = notes_padding(train_notes, padding_length)
        val_notes_padded = notes_padding(val_notes, padding_length)
        val_data = data_dict_generate(val_time['MORTALITY_LABEL'], val_notes_padded)
        train_data = data_dict_generate(np.array(train_time['MORTALITY_LABEL'])[index], train_notes_padded)
        test_data = data_dict_generate(test_time['MORTALITY_LABEL'], test_notes_padded)

    else:
        val_data = data_dict_generate(val_time['MORTALITY_LABEL'], val_notes)
        train_data = data_dict_generate(np.array(train_time['MORTALITY_LABEL'])[index], train_notes)
        test_data = data_dict_generate(test_time['MORTALITY_LABEL'], test_notes)

    return train_data, val_data, test_data, len_all.max()









