from sklearn.feature_extraction.text import TfidfVectorizer


def build_vocabulary(documents):
    # save the table as a file
    return vocab

def shrink_by_dfidf(documents):
    return vocabulary

'''
Args:
   notes - list of notes from patients, [notes from patient1, notes from patient2, ...]
   k - the number of words to choose from tf-idf
Return:
    each line is top k words of a patient 
'''
def tf_idf(notes, k):
    vectorizer = TfidfVectorizer()
    vec = vectorizer.fit_transform(notes).toarray()
    #index_list = list(vectorizer.idf_.argsort()[:number])
    vec = [list(vec[i].argsort()[-k:][::-1]) for i in range(vec.shape[0])]
    word_list = vectorizer.get_feature_names()
    tf_idf = []
    for i in range(len(vec)):
        words_list = []
        for j in range(len(vec[0])):
            words_list.append(word_list[vec[i][j]])
        tf_idf.append(words_list)
    return tf_idf
