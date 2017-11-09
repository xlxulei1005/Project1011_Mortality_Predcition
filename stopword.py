'''
	Stopwords class - part of sentlex
	This class responds to True/False queries about whether a word is a stop word, based on a source file.
	The default stop words file is derived from Cornell Univ. SMART information system, and van Rijsbergen:
	http://nlp.uned.es/~ircourse/examples/stoplist.html
	http://www.lextek.com/manuals/onix/stopwords2.html
'''
import os


class Stopword(object):
	'''
	 Stopword class - encapsulates dict containing all known stop words in lowercase with a function removing the stopword
	'''
	def __init__(self, filename=None):
		self.worddict = {}
		if not filename:
				curpath = os.path.dirname(os.path.abspath(__file__))
				filename = os.path.join(curpath, 'stopwords.txt')
				self.load(filename)

	def load(self, filename):
		f = open(filename)
		for word in f.readlines():
				self.worddict[word[:-1].lower()] = 1

	def is_stop(self, word):
		return (word.lower() in self.worddict)

	def tokentext_no_stopw(self, tokenized_text):

		return [t for t in tokenized_text if not self.is_stop(t)]

if __name__ == "__main__":
	print(Stopword().tokentext_no_stopw(['A','apple','is','On','the','Table']))


	
