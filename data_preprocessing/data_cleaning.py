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


def find_standrad_identification(identification, transformer_list):
    for transformer in transformer_list:
        standrad_identifications = transformer(identification)
    return standrad_identifications

def create_unique_identification_dictionary(documents):
    '''
    Find all unique identification and return a replacement table
    '''
    unique_identifications = find_unique_identification(documents)
    standrad_identifications = []
    for identification in  unique_identifications:
       standrad_identifications.append(find_standrad_identification(identification[2:-2]))
    return dict(zip(unique_identifications, standrad_identifications)) 


def identification_reform(documents, unique_identification_dictionary):
    '''
    Args:
        documents: orignal data
        unique_identification_dictionary: {'**name1**': 'FIRSTNAME' ....}
    Returns:
             cleaned_doc: all identification replace by a standrad symbol
     '''
     
    cleaned_doc = []
    for note in documents:
        for identification in unique_identification_dictionary.keys():
           note = note.replace(identification, unique_identification_dictionary[identification])
        cleaned_doc.append(note)
   return cleaned_doc

def clean_spaces(documents):
    cleaned_doc = []
    for note in documents:
        single_space_note = ' '.join(note.split())
        cleaned_doc.append(single_space_note)
    return cleaned_doc



