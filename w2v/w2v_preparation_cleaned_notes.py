#is the file for tokenization 
import pickle
import numpy as np
import time
import pandas as pd
from nltk.tokenize.treebank import TreebankWordTokenizer
import re
import string
import argparse
import gensim.models.word2vec as w2v
import numpy as np

#from data_preprocessing import * #invalid when in the terminal since it is no module(no initial file)

notes_path = './cleaned_notes.csv'
word_list_path = './label.npy'

def find_unique_identification(documents, unique_identification_dictionary):
    '''
    final all unique identification 

    Returns:
       A list of unique identications
    '''
    identifications = []
    new_documents = []
    print(len(documents))
    all_l = len(documents)
    num = 0
    start = time.time()
    for note in documents:
        num+=1
        if num%1000 == 0:
            print('Processed: ', time.time() - start, num/all_l)
        i = 0
        new = note
        while i <len(note):
            if note[i] == '[' and note[i+1] == '*':
                j = i+1
                while note[j-1] != '*' or note[j] != ']':
                    j+=1
                identifications.append(note[i:j+1])
                #print(unique_identification_dictionary[note[i:j+1]])
                #new = new[:i] + [x for x in unique_identification_dictionary[note[i:j+1]]] + new[j+2:]
                if len(unique_identification_dictionary) != 0:
                    new = new.replace(note[i:j+1] ,unique_identification_dictionary[note[i:j+1]])
                else:
                    new = new.replace(note[i:j+1] ,find_standrad_identification(note[i:j+1][3:-3]))
                #print(note[i:j+1], unique_identification_dictionary[note[i:j+1]])
                #print(new)
            i+=1
        new_documents.append(new)

def word_tokenize_by_string(note):
    translator = str.maketrans('', '', string.punctuation.replace(".",""))
    _treebank_word_tokenizer = TreebankWordTokenizer()
    note = note.translate(translator)
    note = note.replace('0','#')
    note = note.replace('1','#')
    note = note.replace('2','#')
    note = note.replace('3','#')
    note = note.replace('4','#')
    note = note.replace('5','#')
    note = note.replace('6','#')
    note = note.replace('7','#')
    note = note.replace('8','#')
    note = note.replace('9','#')
    tokenized_note = _treebank_word_tokenizer.tokenize(note)
    return tokenized_note



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='word2vec')
    parser.add_argument('--embedding_dim', default=300, type=int, help='embedding dim (default: 300)')
    parser.add_argument('--mode', default=1, type=int, help='mode (default: 1)')
    #parser.add_argument()
    args = parser.parse_args()


    notes_cleaned_pickle = pd.read_csv(notes_path,)

    #print('complete one')
    #print(notes_cleaned_pickle['DOCUMENTS'])
    documents = notes_cleaned_pickle['DOCUMENTS']
    #print(documents)
    #print(type(documents))
    #print(documents)  
  
    print('finish reading, start clean')
    #_, cleaned_doc= find_unique_identification(documents,[])
    #cleaned_doc = clean_spaces(cleaned_doc)
     
    print('start tokenize')
    cleaned_doc = documents.tolist()
    print(type(cleaned_doc))
    cleaned_token = []
    for i in range(len(cleaned_doc)):
        try:
            cleaned_doc[i] = ' '.join(word_tokenize_by_string(cleaned_doc[i])).replace(".", " . ").split('.') #list of sentences
            cleaned_token_temp = [i.split(' ') for i in cleaned_doc[i]] #list of lists
            cleaned_token.extend(cleaned_token_temp)
        except AttributeError:
            pass
        
    #print('start training word2vec')
    #pickle.dump(tokens, open( "tokened_sentence.p", "wb" ) )


    embed_dim = args.embedding_dim
    min_word_count = 1
    #num_workers = multiprocessing.cpu_count()
    context_size = 7
    downsampling = 1e-3
    seed = 134
    print('start model')
    if args.mode == 1:
        medical2vec = w2v.Word2Vec(cleaned_token, sg = 1, seed = seed, workers=3, size = embed_dim, min_count = min_word_count, \
                           window = context_size, sample = downsampling)
        medical2vec.save('model_word2vec_mode1_cleaned_notes.bin')
        print('model saved')
        #words = list(medical2vec.wv.vocab)
        #pickle.dump(words, open( "voca.p", "wb" ) )
        #model.save
    elif args.mode == 2:
        #only use words in the vocabulary already
        #tokens = [filter_by_set(i) for i in tokens]
        medical2vec = w2v.Word2Vec(tokens,sg = 1, seed = seed, workers=3, size = embed_dim, min_count = 1, \
                           window = context_size, sample = downsampling)

        medical2vec.save('model_word2vec_mode2.bin')
        #words = list(medical2vec.wv.vocab)
        #pickle.dump(words, open( "tokened_sentence.p", "wb" ) )
