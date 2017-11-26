"""
Nan Wu
All rights reserved
Report bugs to Nan Wu nw1045@nyu.edu
"""
import torch
import numpy as np
import os

def notes_combine(infor_all):
    data = infor_all['DATA'] 
    notes_per_sub = []
    max_len = []
    for sub in data:
        notes = []
        for note in sub:
            d = [item for sublist in note for item in sublist]
            notes.append(d )
            max_len.append(len(d))
        notes_per_sub.append(notes)
    del notes
    del d
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

def data_formatting(path, time_name, test_file = None, train_file =None,val_file= None ):
    if test_file == None:
    	test_file = 'test_'+ time_name +'.npy'
    if train_file == None:
        train_file = 'train_'+ time_name +'.npy' 
    if val_file == None:
        val_file = 'val_'+ time_name +'.npy'

    test_time = np.load(os.path.join(path, test_file)).item()
    train_time = np.load(os.path.join(path, train_file)).item()
    val_time = np.load(os.path.join(path, val_file)).item()

    test_notes, len_test = notes_combine(test_time)
    train_notes , len_train= notes_combine(train_time)
    val_notes, len_val = notes_combine(val_time)

    len_all = np.array(len_test + len_train + len_val)

    number_of_notes_train = [len(x) for x in train_notes]
    index = np.argsort(np.array(number_of_notes_train))
    train_notes= np.array(train_notes)[list(index)]

    test_notes_padded = notes_padding(test_notes, len_all.max())
    train_notes_padded = notes_padding(train_notes, len_all.max())
    val_notes_padded = notes_padding(val_notes, len_all.max())

    val_data = data_dict_generate(val_time['MORTALITY_LABEL'], val_notes_padded)
    train_data = data_dict_generate(np.array(train_time['MORTALITY_LABEL'])[index], train_notes_padded)
    test_data = data_dict_generate(test_time['MORTALITY_LABEL'], test_notes_padded)
    return train_data, val_data, test_data









