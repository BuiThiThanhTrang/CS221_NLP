'''
Sample predictive model.
You must supply at least 2 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
'''
import numpy as np   
import nltk
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('punkt')
nltk.download("stopwords")

class model:
	def __init__(self):
		self.classifier = None
		self.vectorizer = None
    
	def fit(self, XTrain, YTrain):
		self.vectorizer = CountVectorizer()
		vTrain = self.vectorizer.fit_transform(XTrain).toarray()
		self.classifier = MultinomialNB()
		self.classifier.fit(vTrain, YTrain)


	def predict(self, XTest):	
		vTest = self.vectorizer.transform(XTest).toarray()
		YTest = self.classifier.predict(vTest)
		return YTest
