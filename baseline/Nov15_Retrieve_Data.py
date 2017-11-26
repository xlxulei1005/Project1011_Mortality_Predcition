
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import pickle
from datetime import timedelta
import sys
sys.path.append('./data_preprocessing')
from data_cleaning import word_tokenize_by_string
import random
import os


time_length = input('Please enter the time interval which you want to generate notes for?\n 15m, 6h, 12h, 24h')


time_stamp, measurement = time_length[:-1], time_length[-1]


#all should be changed to sys
word_list_path = './data/label.npy'
patient_timesheet_path = './data/patient_timesheet_final.pickle'
notes_path = './data/cleaned_notes.csv'
mortality_label_path = './data/id_inhospital_deathtime.csv'
SAVE_PATH = './data/'


### read_in dataset

#read_in the vocabulary of the dataset chosen by tfidf
label = np.load(word_list_path)
voca=label.item()['WORD_LIST']



#the dictionary for patient timesheet which includes, patient_id, all charts creation time he has
patient_timesheet = pd.read_pickle(patient_timesheet_path)
#take subject ID, charts creation time as key. notes are cleaned
notes_cleaned_pickle = pd.read_csv(notes_path)

#filter out mortality labels for our subjects
id_inhospital_deathtime = pd.read_csv(mortality_label_path, index_col = 1)
id_inhospital_deathtime = id_inhospital_deathtime[id_inhospital_deathtime.index.isin(SUBJECT_ID)]
mortality_label= ~id_inhospital_deathtime.DEATHTIME.isnull()*1
#index is patient and label is dead or not




#function area
def data_selection(timesheet, time_period):
    '''
    this is to select notes keys from the whole sheet data
    timesheet: patient_timesheet
    time_period: time interval which you want to extract
    return: selected notes in the form of [(subject_id, chartcreation time)]
    '''
    selected_notes = []
    for sub in timesheet.keys():
        i = 0
        try:
            for interval in timesheet[sub]['CHARTTIME_interval']:
                if interval<=time_period:
                    selected_notes.append((sub, timesheet[sub]['CHARTTIME'][i]))
                i+=1
        except KeyError:
            print(sub)
         
    return selected_notes



def notes_take(dictionary_for_notes, table_selecte):
    '''
    dictionary_for_notes: (subject_id, chart-time) documents_content
    table_selected: pandas series with sub_id as its index and list of creation time stamps for notes
    return: notes_selected_by_sub: concatnated notes for patients, summ: which is the no. of wrong keys
    '''
    notes_selected_by_sub = []
    label = []
    summ = 0
    missed=[]
    for sub in table_selecte.index:
        notes = []
        
        for i in range(len(table_selecte[sub])):
            try:
                notes.append(dictionary_for_notes[(sub, table_selecte[sub][i])])
                #print((sub, table_selecte[sub][i]))
            except KeyError:
                #print((sub, table_selecte[sub][i]))
                missed.append((sub, table_selecte[sub][i]))
                summ+=1
        notes_selected_by_sub.append([' '.join(notes), sub])

    return notes_selected_by_sub, summ, missed


def filter_by_set(notes_list, voca_set):
    '''
    notes_list: each note in a list
    voca: the dictionary we use
    return: filtered notes
    '''
    notes_set = set(notes_list)
    notes_vaild = notes_set & voca_set
    new_li=[i for i in notes_list if i in notes_vaild]
    return new_li



def save_splited_data(train_index, val_index,test_index, data, save_file):
    
    f_train = open(os.path.join(SAVE_PATH, 'train_'+save_file+'.txt'),"w")
    f_val = open(os.path.join(SAVE_PATH, 'val_'+save_file+'.txt'),"w")
    f_test = open(os.path.join(SAVE_PATH, 'test_'+save_file+'.txt'),"w")

    for i in range(merged.shape[0]):
        row = merged.iloc[i,:]
        if row['sub_id'] in train_index:
            f_train.write(' '.join(row['notes'])+' '+row['DEATHTIME']+'\n')
        elif row['sub_id'] in val_index:
            f_val.write(' '.join(row['notes'])+' '+row['DEATHTIME']+'\n')
        else:
            f_test.write(' '.join(row['notes'])+' '+row['DEATHTIME']+'\n')
        
        
def split_data_by_time(train_sub,val_sub,test_sub, sub_id ):
    train_sub_t = [x for x in sub_id if x in train_sub]
    val_sub_t = [x for x in sub_id if x in val_sub]
    test_sub_t = [x for x in sub_id if x in test_sub]
    return train_sub_t,val_sub_t, test_sub_t



selected_notes = data_selection(patient_timesheet, np.timedelta64(time_stamp, measurement))
selected_notes = sorted(selected_notes, key = lambda i: (i[0],i[1]))


sub_id = [x[0] for x in selected_notes]
creation_time = [x[1] for x in selected_notes]
table_selected_notes = pd.DataFrame({'sub_id': sub_id, 'creation_time':creation_time})
table_selected = table_selected_notes.groupby(['sub_id'])['creation_time'].unique()


CHARTTIME = notes_cleaned_pickle['CHARTTIME']
DOCUMENTS = notes_cleaned_pickle['DOCUMENTS']
SUBJECT_ID = notes_cleaned_pickle['SUBJECT_ID']
CHARTTIME = [np.datetime64(x, 'ns') for x in CHARTTIME ]

keys_for_notes = tuple(zip(SUBJECT_ID, CHARTTIME))

dictionary_for_notes = dict(zip(keys_for_notes, DOCUMENTS))


notes_selected_by_sub, summ, missed = notes_take(dictionary_for_notes,table_selected )




notes_selected_by_sub = pd.DataFrame(notes_selected_by_sub)
notes_selected_by_sub.columns=['notes','sub_id']
notes_selected_by_sub['notes'] = notes_selected_by_sub['notes'].map(lambda x: word_tokenize_by_string(x))




notes_after = notes_selected_by_sub['notes'].map(lambda x: filter_by_set(x,voca_set))
notes_selected_by_sub['notes']=notes_after



merged = notes_selected_by_sub.merge(pd.DataFrame(mortality_label), how= 'left', left_on='sub_id', right_index=True)
merged['DEATHTIME'] = merged['DEATHTIME'].map(lambda x: '__label__DEAD' if x == 1 else '__label__LIVE')




patient_list = np.array(list(zip(list(mortality_label.index), list(mortality_label))))
train_sub, val_sub, test_sub = data_split(patient_list)
train_sub_,val_sub_, test_sub_ = split_data_by_time(train_sub, val_sub, test_sub, merged.sub_id)


#save the dataset
save_splited_data(train_sub_,val_sub_,test_sub_, merged,time_length)
sub_id_dict = {}
sub_id_dict[time_length]['train_{}'.format(time_length)] = train_sub_
sub_id_dict[time_length]['val_{}'.format(time_length)] = val_sub_
sub_id_dict[time_length]['test_{}'.format(time_length)] = test_sub_
pickle.dump(sub_id_dict, open( "patient_id_splited_data_{}.p".format(time_length), "wb" ) )



