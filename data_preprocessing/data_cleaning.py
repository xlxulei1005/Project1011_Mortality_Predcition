from nltk.tokenize.treebank import TreebankWordTokenizer
import re
import string
import identification_transformer 

def find_unique_identification(documents):
    '''
    final all unique identification 

    Returns:
       A list of unique identications
    '''
    identifications = []
    for note in documents:
        i = 0
        while i <len(note):
            if note[i] == '[' and note[i+1] == '*':
                j = i+1
                while note[j-1] != '*' or note[j] != ']':
                    j+=1
                identifications.append(note[i:j+1])
            i+=1
    return list(set(identifications))


def find_standrad_identification(identification):
    return identification_transformer.transformer(identification)

def create_unique_identification_dictionary(documents):
    '''
    Find all unique identification and return a replacement table
    '''
    unique_identifications = find_unique_identification(documents)
    standrad_identifications = []
    for identification in  unique_identifications:
       standrad_identifications.append(find_standrad_identification(identification[3:-3]))
    return dict(zip(unique_identifications, standrad_identifications)) 


def identification_reform(documents, unique_identification_dictionary = None):
    '''
    Args:
        documents: orignal data
        unique_identification_dictionary: {'**name1**': 'FIRSTNAME' ....}
    Returns:
             cleaned_doc: all identification replace by a standrad symbol
     '''

    identifications = []
    new_documents = []
    for note in documents:
        i = 0
        new = note
        while i <len(note):
            if note[i] == '[' and note[i+1] == '*':
                j = i+1
                while note[j-1] != '*' or note[j] != ']':
                    j+=1
                identifications.append(note[i:j+1])
                if unique_identification_dictionary == None:
                    new = new.replace(note[i:j+1] ,unique_identification_dictionary[note[i:j+1]])
                else:    
                    new = new.replace(note[i:j+1] ,identification_transformer.transformer(note[i:j+1]))
                
            i+=1
        new_documents.append(new)
            
    return list(set(identifications)), new_documents

def clean_spaces(documents):
    cleaned_doc = []
    for note in documents:
        single_space_note = ' '.join(note.split())
        cleaned_doc.append(single_space_note)
    return cleaned_doc

def word_tokenize(documents):
    cleaned_doc = []
    translator = str.maketrans('', '', string.punctuation)
    _treebank_word_tokenizer = TreebankWordTokenizer()
    for note in documents:
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
        cleaned_doc.append(tokenized_note)
    return cleaned_doc

def word_tokenize_by_string(note):
    translator = str.maketrans('', '', string.punctuation)
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



