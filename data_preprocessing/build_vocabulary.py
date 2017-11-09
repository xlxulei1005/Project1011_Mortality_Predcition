from sklearn.feature_extraction.text import TfidfVectorizer


def build_vocabulary(documents):
    # save the table as a file
    return vocab

def shrink_by_dfidf(documents):
    return vocabulary

'''
Args:
   notes - list of notes from a single patient
   number - the number of words to choose from tf-idf
Return:
    list of words

'''
def tf_idf(notes, number):
    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(notes).toarray()
    index_list = list(vectorizer.idf_.argsort()[:number])
    word_list = vectorizer.get_feature_names()
    return ([word_list[index] for index in index_list])
